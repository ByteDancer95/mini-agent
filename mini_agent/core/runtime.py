"""Agent Runtime —— 本项目的核心，全部主流程自己实现，不依赖任何 agent 框架。

主循环（ReAct 风格）
-------------------
    user_input
       -> context.add_user()
       -> [loop] build_messages() 召回记忆
              -> LLM.generate(messages, tool_schemas)   基于 Schema 自主决策
              -> OutputParser.parse()                   提取 thought / tool_calls / final
              ├── final  -> context.add_final() -> 返回答案
              └── tool   -> context.add_decision() -> registry.execute() 逐个执行
                            -> context.add_tool_result() -> maybe_compact()
                            -> 继续下一轮（LLM 依据工具结果再决定继续或收尾）
       -> 超过 max_tool_rounds -> 强制收尾（force_final）或基于已有结果兜底
       -> 落盘 session + 输出 trace

异常处理分层
------------
* 工具层：未知工具 / 参数非法 / handler 抛错 -> ToolResult(ok=False) 回喂 LLM，loop 不中断；
* 解析层：非法 JSON -> 降级为纯文本答案，或注入 REPAIR_HINT 重试一次；
* LLM 层：网络/HTTP 错误 -> 重试 1 次，仍失败则整体降级为 RunResult(status='error')；
* Runtime 边界：任何未预期异常都被捕获，返回结构化错误结果，不向调用方抛栈。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context import ContextManager, ContextPolicy
from .errors import LLMError
from .llm import LLMClient, build_llm
from .parser import OutputParser
from .prompt import REPAIR_HINT, build_system_prompt
from .registry import ToolRegistry
from .session import SessionManager
from .trace import EventType, Trace, TraceSink
from .types import AgentOutput, LLMResponse, Message, Role, RunResult, ToolContext

logger = logging.getLogger("mini_agent.runtime")

__all__ = ["RuntimeConfig", "AgentRuntime", "build_default_runtime"]


@dataclass
class RuntimeConfig:
    max_tool_rounds: int = 5          # 单次用户输入最多执行多少轮工具
    max_llm_retries: int = 1          # LLM 调用失败重试次数
    max_parse_repairs: int = 1        # 输出无法解析时要求模型重写的次数
    extra_instructions: str = ""      # 追加到 system prompt 的补充约定


class AgentRuntime:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry | None = None,
        sessions: SessionManager | None = None,
        config: RuntimeConfig | None = None,
        context_policy: ContextPolicy | None = None,
        docs_root: str | Path | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        from ..tools.base import build_default_registry  # 延迟导入避免循环

        self.llm = llm
        self.registry = registry or build_default_registry()
        self.sessions = sessions or SessionManager(policy=context_policy)
        self.config = config or RuntimeConfig()
        self.docs_root = str(docs_root) if docs_root else None
        self.trace_sink = trace_sink or TraceSink(to_logger=True)
        self.parser = OutputParser()
        self.protocol = getattr(self.llm, "protocol_name", "json")

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, session_id: str, user_input: str) -> RunResult:
        trace = Trace(session_id)
        result = RunResult(session_id=session_id, answer="", run_id=trace.run_id, trace=trace)
        session = None
        text = str(user_input)
        schemas = self.registry.schemas()

        try:
            session = self.sessions.get_or_create(session_id)
            ctx = session.context
            ctx.add_user(text)
            trace.add(EventType.RUN_START, max_tool_rounds=self.config.max_tool_rounds,
                      tools=self.registry.names(), llm_protocol=self.protocol)
            trace.add(EventType.USER_INPUT, text=text)
            # 压缩不能只在「调用工具」分支触发：纯对话长会话同样会无限增长。
            # 在每轮 run 的开头检查一次，保证跨轮次 context 消息数始终有界。
            if ctx.maybe_compact():
                trace.add(EventType.COMPACT, turn=0, stats=ctx.stats(), phase="run_start")

            tool_ctx = ToolContext(session_id=session_id, state=session.state,
                                   docs_root=self.docs_root)
            final_answer: str | None = None
            turns_used = 0

            for round_no in range(1, self.config.max_tool_rounds + 1):
                turns_used = round_no
                messages = self._build(ctx, session, schemas, force_final=False)
                trace.add(EventType.LLM_CALL, turn=round_no, messages=len(messages),
                          context_chars=ctx.total_chars, protocol=self.protocol)

                output = self._think(messages, schemas, force_final=False,
                                     turn=round_no, trace=trace)
                ctx.record_thought(output.thought, round_no, output.source)
                if output.thought:
                    trace.add(EventType.THOUGHT, turn=round_no, thought=output.thought)
                trace.add(EventType.PARSE, turn=round_no, source=output.source,
                          warning=output.parse_warning, tool_calls=len(output.tool_calls))
                trace.add(EventType.DECISION, turn=round_no, decision=output.decision(),
                          tool_count=len(output.tool_calls))

                # ---- 分支 1: 直接回复 ----
                if output.is_final:
                    final_answer = output.final_answer or ""
                    ctx.add_final(final_answer)
                    trace.add(EventType.FINAL, turn=round_no, answer=final_answer)
                    break

                # ---- 分支 2: 无法解析，降级成纯文本答案 ----
                if not output.has_tool_calls:
                    final_answer = (output.raw or "").strip() or "（模型未返回可用内容）"
                    ctx.add_final(final_answer)
                    trace.add(EventType.WARNING, turn=round_no,
                              reason="unparsable_output_fallback", warning=output.parse_warning)
                    trace.add(EventType.FINAL, turn=round_no, answer=final_answer)
                    break

                # ---- 分支 3: 调用工具，再进入下一轮 ----
                ctx.add_decision(output)
                for call in output.tool_calls:
                    trace.add(EventType.TOOL_CALL, turn=round_no, name=call.name,
                              arguments=call.arguments, call_id=call.call_id)
                    tool_result = self.registry.execute(call, tool_ctx)
                    trace.add(EventType.TOOL_RESULT, turn=round_no, name=call.name,
                              call_id=call.call_id, ok=tool_result.ok,
                              elapsed_ms=tool_result.elapsed_ms,
                              error=tool_result.error,
                              preview=self._preview(tool_result.to_payload()))
                    ctx.add_tool_result(call, tool_result)
                if ctx.maybe_compact():
                    trace.add(EventType.COMPACT, turn=round_no, stats=ctx.stats())

            else:
                # for-else: 没有 break == 用尽所有工具轮次
                result.status = "loop_limit"
                trace.add(EventType.LOOP_LIMIT, turn=turns_used,
                          max_tool_rounds=self.config.max_tool_rounds)
                final_answer = self._force_final(ctx, session, schemas,
                                                 turn=turns_used + 1, trace=trace)
                ctx.add_final(final_answer)
                trace.add(EventType.FINAL, turn=turns_used + 1,
                          answer=final_answer, status=result.status)

            result.answer = final_answer or "（未生成答案）"
            result.turns = turns_used

        except Exception as exc:  # noqa: BLE001 - Runtime 边界兜底
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
            result.answer = (
                f"本轮执行出现异常（{type(exc).__name__}），已安全终止：{exc}\n"
                "会话历史已保留，可以直接继续提问。"
            )
            trace.add(EventType.ERROR, turn=0, error=result.error)
            logger.exception("AgentRuntime.run 异常")
        finally:
            if session is not None:
                session.touch()
                self.sessions.maybe_save(session_id)
                trace.add(EventType.SESSION_SAVE, persisted=bool(self.sessions.persist_dir))
            try:
                self.trace_sink.emit(trace)
            except Exception:  # noqa: BLE001 - 日志失败不能影响结果
                logger.warning("trace sink 输出失败", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------
    def _build(self, ctx: ContextManager, session, schemas, force_final: bool) -> list[Message]:
        system_prompt = build_system_prompt(
            schemas,
            force_final=force_final,
            extra_instructions=self.config.extra_instructions,
            protocol=self.protocol,
        )
        return ctx.build_messages(system_prompt, session.state_snapshot())

    def _think(
        self,
        messages: list[Message],
        schemas: list[dict[str, Any]],
        force_final: bool,
        turn: int,
        trace: Trace,
    ) -> AgentOutput:
        """调用 LLM + 解析 + LLM 重试 + 解析修复。"""
        response: LLMResponse | None = None
        last_error: Exception | None = None
        for attempt in range(self.config.max_llm_retries + 1):
            try:
                response = self.llm.generate(messages, schemas, force_final=force_final)
                trace.add(EventType.LLM_RESPONSE, turn=turn, attempt=attempt,
                          raw_chars=len(response.raw),
                          native_tool_calls=len(response.native_tool_calls))
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                trace.add(EventType.WARNING, turn=turn, attempt=attempt,
                          reason="llm_retry", error=f"{type(exc).__name__}: {exc}")
                if attempt >= self.config.max_llm_retries:
                    raise LLMError(f"LLM 调用失败（已重试 {attempt} 次）: {exc}") from None
        if response is None:  # pragma: no cover
            raise LLMError(f"LLM 未返回响应: {last_error}")

        output = self.parser.parse(response)
        repairs = 0
        while not output.is_usable and repairs < self.config.max_parse_repairs:
            repairs += 1
            trace.add(EventType.REPAIR, turn=turn, attempt=repairs,
                      warning=output.parse_warning)
            repair_messages = messages + [
                Message(role=Role.SYSTEM.value, content=REPAIR_HINT,
                        meta={"kind": "repair_hint", "volatile": True})
            ]
            try:
                fixed = self.llm.generate(repair_messages, schemas, force_final=force_final)
                candidate = self.parser.parse(fixed)
            except Exception as exc:  # noqa: BLE001
                trace.add(EventType.WARNING, turn=turn, reason="repair_failed", error=str(exc))
                break
            candidate.source = "repaired"
            output = candidate
        return output

    def _force_final(
        self, ctx: ContextManager, session, schemas, turn: int, trace: Trace
    ) -> str:
        """达到轮次上限：给模型一次「禁止调工具」的机会；失败则用已有结果兜底。"""
        try:
            messages = self._build(ctx, session, schemas, force_final=True)
            trace.add(EventType.LLM_CALL, turn=turn, messages=len(messages),
                      force_final=True)
            output = self._think(messages, schemas, force_final=True, turn=turn, trace=trace)
            if output.final_answer:
                return output.final_answer
        except Exception as exc:  # noqa: BLE001
            trace.add(EventType.WARNING, turn=turn, reason="force_final_failed", error=str(exc))
        return self._fallback_answer(ctx)

    @staticmethod
    def _fallback_answer(ctx: ContextManager) -> str:
        lines: list[str] = []
        for msg in ctx.messages[-10:]:
            if (msg.meta or {}).get("kind") != "tool_result":
                continue
            name = (msg.meta or {}).get("tool", "tool")
            payload = (msg.meta or {}).get("payload") or {}
            if payload.get("ok"):
                lines.append(f"- {name}: {str(payload.get('result'))[:200]}")
            else:
                lines.append(f"- {name} 执行失败: {payload.get('error')}")
        head = "已达到最大工具轮次，不再继续调用工具。已取得的信息：\n"
        return head + ("\n".join(lines) if lines else "（本轮未取得任何工具结果）")

    @staticmethod
    def _preview(payload: dict[str, Any], limit: int = 300) -> str:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        return text[:limit] + ("…" if len(text) > limit else "")


def build_default_runtime(
    *,
    llm_kind: str = "auto",
    persist_dir: str | Path | None = None,
    trace_path: str | Path | None = None,
    echo_trace: bool = False,
    config: RuntimeConfig | None = None,
    context_policy: ContextPolicy | None = None,
    registry: ToolRegistry | None = None,
    llm_kwargs: dict[str, Any] | None = None,
) -> AgentRuntime:
    """一键组装（CLI / demo 用）。llm_kind='auto' 时无密钥自动降级为演示模式。"""
    from ..tools.base import build_default_registry

    llm = build_llm(llm_kind, **(llm_kwargs or {}))
    sink = TraceSink(path=trace_path, echo=echo_trace, to_logger=True)
    sessions = SessionManager(persist_dir=persist_dir, policy=context_policy, autosave=True)
    return AgentRuntime(
        llm=llm,
        registry=registry or build_default_registry(),
        sessions=sessions,
        config=config,
        context_policy=context_policy,
        trace_sink=sink,
    )
