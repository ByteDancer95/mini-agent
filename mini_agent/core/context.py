"""Context 管理：决定「哪些信息进入 LLM context、以什么形式、放在什么位置」。

召回时机（when）
---------------
1. 每次 loop 迭代调用 ``build_messages()`` 时 —— 短期/工作记忆被召回；
2. Session 首次被 ``get_or_create`` 命中（含从磁盘恢复）时 —— 长期会话记忆被召回；
3. 纯对话追问与带工具追问走的是**同一份** ``_messages``，区别只在于本轮是否追加 tool 消息。

放置方式（where / what）
------------------------
放入 LLM context：
  * system 提示词（工具目录 + 输出协议）—— 每轮重建，位于最前，**不存入历史**；
  * 压缩摘要 digest —— system 角色，紧跟提示词；
  * 会话状态快照（todo 清单）—— system 角色，紧跟 digest，按需注入；
  * 用户输入 user / 助手最终答案 assistant / 工具决策 assistant（去掉 thought）/ 工具结果 tool。

刻意 **不** 放进 LLM context：
  * Agent 思考过程（thought）—— 它是中间推理，回放会显著增加 token 且容易让模型复读；
    改为落到 ``_audit``（会话审计，可持久化）与 Trace 日志，用于可观测性与调试。
    可通过 ``ContextPolicy.include_thought_in_llm_context=True`` 打开。
  * 工具结果里的超大 payload —— 按 ``tool_result_max_chars`` 截断。
  * 历史里过旧的回合 —— 压缩成 digest 一行。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any

from .types import AgentOutput, Message, Role, ToolCall, ToolResult

__all__ = ["ContextPolicy", "ContextManager", "tool_result_text"]


@dataclass
class ContextPolicy:
    include_thought_in_llm_context: bool = False
    compress: bool = True
    max_messages: int = 40
    max_chars: int = 12000
    keep_recent: int = 12
    digest_max_lines: int = 20
    inject_state_snapshot: bool = True
    tool_result_max_chars: int = 1500


def tool_result_text(name: str, call_id: str, payload: dict[str, Any], limit: int = 1500) -> str:
    """工具结果进入 context 的文本形态（LLM 只看得到这个字符串）。"""
    body = json.dumps(payload, ensure_ascii=False, default=str)
    if len(body) > limit:
        body = body[:limit] + "…(truncated)"
    return f'<tool_result name="{name}" call_id="{call_id}">{body}</tool_result>'


class ContextManager:
    """单个会话的对话上下文。所有写操作都会即时更新，压缩由 Runtime 显式触发。"""

    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy()
        self._messages: list[Message] = []
        self._digest: list[str] = []
        self._audit: list[dict[str, Any]] = []
        self._compactions = 0

    # ---- 写入 -------------------------------------------------------------
    def add_user(self, text: str) -> Message:
        msg = Message(role=Role.USER.value, content=text, meta={"kind": "user_input"})
        self._messages.append(msg)
        return msg

    def add_decision(self, output: AgentOutput) -> Message:
        """记录「调用工具」这一决策。

        默认只写入 action，**剥离 thought** —— 思考过程不进入 LLM context。
        """
        obj: dict[str, Any] = {
            "action": {
                "type": "tool",
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in output.tool_calls],
            }
        }
        if self.policy.include_thought_in_llm_context and output.thought:
            obj["thought"] = output.thought
        msg = Message(
            role=Role.ASSISTANT.value,
            content=json.dumps(obj, ensure_ascii=False, default=str),
            meta={"kind": "tool_decision"},
        )
        self._messages.append(msg)
        return msg

    def add_tool_result(self, call: ToolCall, result: ToolResult) -> Message:
        payload = result.to_payload()
        msg = Message(
            role=Role.TOOL.value,
            content=tool_result_text(call.name, call.call_id, payload,
                                     self.policy.tool_result_max_chars),
            name=call.name,
            tool_call_id=call.call_id,
            meta={"kind": "tool_result", "tool": call.name,
                  "call_id": call.call_id, "payload": payload,
                  "ok": result.ok, "elapsed_ms": result.elapsed_ms},
        )
        self._messages.append(msg)
        return msg

    def add_final(self, answer: str) -> Message:
        msg = Message(role=Role.ASSISTANT.value, content=answer, meta={"kind": "final"})
        self._messages.append(msg)
        return msg

    def record_thought(self, thought: str, turn: int, source: str = "") -> None:
        """思考过程只进审计，不进 LLM context。"""
        if not thought:
            return
        self._audit.append({"turn": turn, "thought": thought, "source": source, "ts": time.time()})

    # ---- 召回 -------------------------------------------------------------
    def build_messages(self, system_prompt: str, state_snapshot: str | None = None) -> list[Message]:
        """组装本次 LLM 调用的完整 messages —— 这就是短期记忆的召回时刻。"""
        out = [Message(role=Role.SYSTEM.value, content=system_prompt,
                       meta={"kind": "system_prompt", "volatile": True})]
        if self._digest:
            body = "[Earlier conversation digest]\n" + "\n".join(f"- {line}" for line in self._digest)
            out.append(Message(role=Role.SYSTEM.value, content=body,
                               meta={"kind": "digest", "volatile": True}))
        if self.policy.inject_state_snapshot and state_snapshot:
            body = "[Session state snapshot]\n" + state_snapshot
            out.append(Message(role=Role.SYSTEM.value, content=body,
                               meta={"kind": "state_snapshot", "volatile": True}))
        out.extend(self._messages)
        return out

    # ---- 基础压缩 ---------------------------------------------------------
    @property
    def total_chars(self) -> int:
        return sum(len(m.content) for m in self._messages)

    def needs_compaction(self) -> bool:
        return self.policy.compress and (
            len(self._messages) > self.policy.max_messages
            or self.total_chars > self.policy.max_chars
        )

    @staticmethod
    def _digest_line(msg: Message) -> str:
        text = (msg.content or "").replace("\n", " ")[:120]
        label = msg.meta.get("tool") if msg.meta.get("kind") == "tool_result" else None
        prefix = f"tool[{label}]" if label else msg.role_str
        return f"{prefix}: {text}"

    def maybe_compact(self) -> bool:
        """超过阈值就把最旧的消息折成 digest 一行，保留最近 keep_recent 条原样。"""
        if not self.needs_compaction() or len(self._messages) <= 1:
            return False
        keep = max(1, min(self.policy.keep_recent, len(self._messages) - 1))
        old = self._messages[:-keep]
        recent = self._messages[-keep:]
        for msg in old:
            self._digest.append(self._digest_line(msg))
        self._digest = self._digest[-self.policy.digest_max_lines:]
        # 保证最近一轮的 user 消息不会被压掉，否则追问会失去指代对象
        if not any(m.role == "user" for m in recent):
            last_user = next((m for m in reversed(old) if m.role == "user"), None)
            if last_user is not None:
                recent = [last_user] + recent
        self._messages = recent
        self._compactions += 1
        return True

    # ---- 观测 / 持久化 ----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "messages": len(self._messages),
            "chars": self.total_chars,
            "digest_lines": len(self._digest),
            "audit_entries": len(self._audit),
            "compactions": self._compactions,
            "include_thought_in_llm_context": self.policy.include_thought_in_llm_context,
        }

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def audit(self) -> list[dict[str, Any]]:
        return list(self._audit)

    @property
    def digest(self) -> list[str]:
        return list(self._digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": asdict(self.policy),
            "messages": [m.to_dict() for m in self._messages],
            "digest": list(self._digest),
            "audit": list(self._audit),
            "compactions": self._compactions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextManager":
        policy_data = data.get("policy") or {}
        policy = ContextPolicy(**{k: v for k, v in policy_data.items()
                                  if k in ContextPolicy.__dataclass_fields__})
        ctx = cls(policy)
        ctx._messages = [Message.from_dict(m) for m in data.get("messages", [])]
        ctx._digest = list(data.get("digest", []))
        ctx._audit = list(data.get("audit", []))
        ctx._compactions = int(data.get("compactions", 0))
        return ctx
