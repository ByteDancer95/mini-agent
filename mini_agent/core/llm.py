"""可替换的 LLM 客户端。Runtime 只依赖 ``generate()`` 这一个方法。

三种实现：

* ``OpenAICompatLLM`` — 真实模型。**API Key / Base URL / Model 全部只从环境变量或构造参数读取，
  源码中不含任何密钥、不含任何厂商地址字面量**；用标准库 urllib 发请求，零第三方依赖。
  支持 ``json``（自研协议）与 ``native``（OpenAI function calling）两种决策方式。
* ``HeuristicLLM``    — 零密钥本地确定性替身，让仓库 clone 下来就能跑完整 loop 与多 session 演示。
  它明确是 LLM 的替身，不伪装成模型。
* ``ScriptedLLM``     — 按脚本返回预设回复，供单元测试精确构造各种边界场景。

安全约定：任何情况下都不把 api_key 写进日志、异常消息或 trace。
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .errors import LLMConfigError, LLMError
from .types import LLMResponse, Message, ToolCall

__all__ = [
    "LLMClient", "OpenAICompatLLM", "HeuristicLLM", "ScriptedLLM",
    "build_llm", "load_env_file",
]

logger = logging.getLogger("mini_agent.llm")

_ENV_NAMES = {
    "api_key": ("MINI_AGENT_API_KEY", "OPENAI_API_KEY"),
    "base_url": ("MINI_AGENT_BASE_URL", "OPENAI_BASE_URL"),
    "model": ("MINI_AGENT_MODEL", "OPENAI_MODEL"),
    "protocol": ("MINI_AGENT_LLM_PROTOCOL",),
    "timeout": ("MINI_AGENT_TIMEOUT",),
}


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def load_env_file(path: str | Path = ".env") -> int:
    """极简 .env 读取（纯标准库）。只填充**尚未设置**的环境变量，返回加载的键数。

    绝不覆盖进程已有的环境变量；文件不存在时静默返回 0。
    """
    p = Path(path)
    if not p.is_file():
        return 0
    count = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


@runtime_checkable
class LLMClient(Protocol):
    """Runtime 唯一依赖的接口。"""

    protocol_name: str

    def generate(
        self,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]],
        force_final: bool = False,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# 1) 真实 LLM：OpenAI 兼容 Chat Completions
# ---------------------------------------------------------------------------
class OpenAICompatLLM:
    """通过环境变量配置的 OpenAI 兼容端点客户端。

    必需环境变量
    ------------
    MINI_AGENT_API_KEY (或 OPENAI_API_KEY)
    MINI_AGENT_BASE_URL (或 OPENAI_BASE_URL)  例如 https://<your-endpoint>/v1
    MINI_AGENT_MODEL    (或 OPENAI_MODEL)     模型名

    可选
    ----
    MINI_AGENT_LLM_PROTOCOL = json | native   (默认 json)
    MINI_AGENT_TIMEOUT      秒数               (默认 60)
    """

    protocol_name = "json"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        protocol: str | None = None,
        timeout: float | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or _env(*_ENV_NAMES["api_key"])
        self.base_url = (base_url or _env(*_ENV_NAMES["base_url"]))
        self.model = model or _env(*_ENV_NAMES["model"])
        self.protocol_name = (protocol or _env(*_ENV_NAMES["protocol"]) or "json").lower()
        if self.protocol_name not in {"json", "native"}:
            raise LLMConfigError("MINI_AGENT_LLM_PROTOCOL 只能是 json 或 native")
        env_timeout = _env(*_ENV_NAMES["timeout"])
        try:
            self.timeout = float(timeout if timeout is not None else (env_timeout or 60))
        except ValueError as exc:
            raise LLMConfigError("MINI_AGENT_TIMEOUT 必须是数字") from exc
        self.temperature = temperature
        self.max_tokens = max_tokens

        missing = [
            var for var, val in (
                ("MINI_AGENT_API_KEY / OPENAI_API_KEY", self.api_key),
                ("MINI_AGENT_BASE_URL / OPENAI_BASE_URL", self.base_url),
                ("MINI_AGENT_MODEL / OPENAI_MODEL", self.model),
            ) if not val
        ]
        if missing:
            raise LLMConfigError(
                "缺少 LLM 配置，请设置环境变量: " + "; ".join(missing)
                + "（参见 .env.example；不要把密钥写进代码）"
            )
        self.base_url = self.base_url.rstrip("/")

    def generate(
        self,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]],
        force_final: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_chat() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.protocol_name == "native" and tool_schemas:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s.get("parameters", {"type": "object", "properties": {}}),
                }}
                for s in tool_schemas
            ]
            if force_final:
                payload["tool_choice"] = "none"

        url = self.base_url + "/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            # 注意：绝不把 request headers / api_key 放进异常信息
            raise LLMError(f"LLM HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"LLM 网络错误: {type(exc).__name__}: {exc}") from None
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM 响应格式异常: {type(exc).__name__}: {exc}") from None

        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            raw = msg.get("content") or ""
            if not isinstance(raw, str):
                raw = json.dumps(raw, ensure_ascii=False)
            calls: list[ToolCall] = []
            for item in msg.get("tool_calls") or []:
                fn = item.get("function") or {}
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                calls.append(ToolCall(
                    name=str(fn.get("name") or ""),
                    arguments=args,
                    call_id=str(item.get("id") or ""),
                ))
            return LLMResponse(
                raw=raw,
                native_tool_calls=calls,
                finish_reason=choice.get("finish_reason"),
                model=data.get("model"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM 响应缺少 choices/message: {exc}") from None


# ---------------------------------------------------------------------------
# 2) 零密钥本地替身（明确标注为 heuristic，不伪装成 LLM）
# ---------------------------------------------------------------------------
_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京",
           "西安", "重庆", "长沙", "苏州", "天津", "郑州"]
_ARITH_RE = re.compile(r"[0-9][0-9\.\s\+\-\*\/\(\)\^\%×÷]{2,}")
_TODO_ADD_RE = re.compile(
    r"(?:记个待办|记一下|添加待办|新增待办|加到待办|待办事项|todo)\s*[:：,，]?\s*([^；;。\n]+)",
    re.IGNORECASE,
)
_NO_TOOL_RE = re.compile(r"不用工具|不要调用|直接回答|复述|纯聊天|不用查", re.IGNORECASE)
_RESULT_RE = re.compile(r"<tool_result[^>]*>(.*?)</tool_result>", re.DOTALL)


class HeuristicLLM:
    """本地确定性决策器：模拟「LLM 基于 Schema 自主决策」这一步。

    用途：无 API Key 时也能完整跑通 loop / 工具调用 / 多 session / 追问演示。
    它按关键词决定是否调用工具、调用哪些工具；看到 role=tool 结果后生成最终答案。
    """

    protocol_name = "json"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    @staticmethod
    def _json(obj: dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False)

    def generate(
        self,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]],
        force_final: bool = False,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        user_positions = [i for i, m in enumerate(messages) if m.role == "user"]
        if not user_positions:
            return LLMResponse(raw=self._json({
                "thought": "没有用户输入", "action": {"type": "final", "content": "请告诉我你想做什么。"}
            }))
        last_user = user_positions[-1]
        user_text = messages[last_user].content
        segment = messages[last_user + 1:]
        tool_msgs = [m for m in segment if m.role == "tool"]

        if force_final or tool_msgs:
            return LLMResponse(raw=self._json({
                "thought": "已收到工具结果，信息足够，可以给出最终答案。",
                "action": {"type": "final", "content": self._final_from_tools(user_text, tool_msgs, messages)},
            }))

        if _NO_TOOL_RE.search(user_text):
            return LLMResponse(raw=self._json({
                "thought": "用户明确要求不使用工具，属于纯对话追问，直接基于历史上下文回答。",
                "action": {"type": "final", "content": self._chitchat(user_text, messages[:last_user])},
            }))

        calls = self._decide(user_text)
        if calls:
            return LLMResponse(raw=self._json({
                "thought": "当前信息不足以直接回答，需要调用下列工具获取事实。",
                "action": {"type": "tool", "tool_calls": calls},
            }))
        return LLMResponse(raw=self._json({
            "thought": "这是普通对话，无需调用工具，直接回答。",
            "action": {"type": "final", "content": self._chitchat(user_text, messages[:last_user])},
        }))

    # ---- 决策 -------------------------------------------------------------
    @staticmethod
    def _decide(text: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        expr_match = _ARITH_RE.search(text)
        if expr_match and re.search(r"[\+\-\*\/\^\%×÷]|乘以|除以", expr_match.group(0)):
            expr = (expr_match.group(0).strip()
                    .replace("×", "*").replace("÷", "/").replace("^", "**"))
            calls.append({"name": "calculator", "arguments": {"expression": expr}})

        if "天气" in text or "weather" in text.lower():
            city = next((c for c in _CITIES if c in text), "广州")
            calls.append({"name": "weather", "arguments": {"city": city}})

        if re.search(r"搜索|检索|搜一下|查资料|找资料|search", text, re.IGNORECASE):
            query = re.sub(r"搜索|检索|搜一下|查资料|找资料|一下|帮我|请", "", text).strip(" ，,。:：") or "农业 病虫害"
            calls.append({"name": "search", "arguments": {"query": query, "top_k": 3}})

        if re.search(r"周报|文档模板|读文档|read_docs", text):
            calls.append({"name": "read_docs", "arguments": {"path": "weekly_report_template.md"}})

        add_match = _TODO_ADD_RE.search(text)
        if add_match and not re.search(r"列出|查看|我的待办|待办有哪些|刚才记的", text):
            item = add_match.group(1).strip(" ，,。:：")
            if item:
                calls.append({"name": "todo", "arguments": {"action": "add", "item": item}})
        elif re.search(r"待办|todo", text, re.IGNORECASE):
            calls.append({"name": "todo", "arguments": {"action": "list"}})

        return calls

    # ---- 生成最终答案 -----------------------------------------------------
    @staticmethod
    def _payload(msg: Message) -> dict[str, Any]:
        payload = msg.meta.get("payload") if isinstance(msg.meta, dict) else None
        if isinstance(payload, dict):
            return payload
        m = _RESULT_RE.search(msg.content or "")
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
        return {}

    @classmethod
    def _final_from_tools(
        cls, user_text: str, tool_msgs: list[Message], messages: list[Message]
    ) -> str:
        lines: list[str] = []
        weekly = bool(re.search(r"周报", user_text))
        for msg in tool_msgs:
            name = (msg.meta or {}).get("tool", "")
            payload = cls._payload(msg)
            result = payload.get("result")
            if not payload.get("ok"):
                lines.append(f"- {name or '工具'} 执行失败：{payload.get('error')}")
                continue
            if name == "calculator" and isinstance(result, dict):
                lines.append(f"- 计算结果：{result.get('expression')} = {result.get('result')}")
            elif name == "weather" and isinstance(result, dict):
                lines.append(
                    f"- {result.get('city')}天气：{result.get('condition')}，"
                    f"{result.get('temperature')}°C，{result.get('advice')}"
                )
            elif name == "search" and isinstance(result, dict):
                items = result.get("results") or []
                titles = "、".join(str(i.get("title")) for i in items[:3]) or "无"
                lines.append(f"- 检索到 {len(items)} 条资料：{titles}")
            elif name == "todo" and isinstance(result, dict):
                todos = result.get("todos") or []
                if todos:
                    items = "；".join(
                        f"{t.get('id')}. {t.get('item')}{'（已完成）' if t.get('done') else ''}"
                        for t in todos
                    )
                    lines.append(f"- 当前待办（{len(todos)} 项）：{items}")
                else:
                    lines.append("- 当前待办为空")
            elif name == "read_docs" and isinstance(result, dict):
                preview = str(result.get("content") or "")[:80].replace("\n", " ")
                lines.append(f"- 已读取文档 {result.get('path')}：{preview}…")
            else:
                lines.append(f"- {name or '工具'} 返回：{str(result)[:120]}")

        if weekly:
            head = "本周周报（草稿，依据文档模板与当前待办生成）：\n"
            body = (
                "  1. 本周进展：完成既定任务并同步记录待办；\n"
                "  2. 风险与阻塞：暂无；\n"
                "  3. 下周计划：按待办清单推进。\n"
            )
            return head + body + "工具侧信息：\n" + "\n".join(lines)
        if not lines:
            return "已完成处理，但没有拿到可用的工具结果。"
        return "已按你的要求完成：\n" + "\n".join(lines)

    @staticmethod
    def _chitchat(text: str, history: list[Message]) -> str:
        prev = [m.content for m in history if m.role == "assistant" and m.content]
        if prev:
            tail = prev[-1].replace("\n", " ")[:60]
            return (
                f"承接上一轮回答「{tail}…」：关于「{text.strip()[:40]}」，"
                "补充说明——这些信息来自本会话此前的上下文，本轮无需调用工具。"
            )
        return f"收到：「{text.strip()[:60]}」。这是本会话的第一轮，我先记录下来了，有具体任务可以随时告诉我。"


# ---------------------------------------------------------------------------
# 3) 脚本化 LLM（测试专用）
# ---------------------------------------------------------------------------
class ScriptedLLM:
    """按队列返回预设回复，用完后再调用默认返回 final。

    元素可以是 str / LLMResponse / callable(messages, schemas, force_final) -> str|LLMResponse。
    ``calls`` 会记录每次收到的 messages，方便断言「哪些信息进入了 context」。
    """

    protocol_name = "json"

    def __init__(self, responses: list[Any], default_final: str = "（脚本已用尽）完成。") -> None:
        self._responses = list(responses)
        self.default_final = default_final
        self.calls: list[list[Message]] = []
        self.force_final_flags: list[bool] = []

    def _next(self, messages, schemas, force_final):
        if self._responses:
            item = self._responses.pop(0)
            if callable(item):
                item = item(messages, schemas, force_final)
        else:
            item = json.dumps({
                "thought": "脚本已用尽", "action": {"type": "final", "content": self.default_final}
            }, ensure_ascii=False)
        return item if isinstance(item, LLMResponse) else LLMResponse(raw=str(item))

    def generate(
        self,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]],
        force_final: bool = False,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.force_final_flags.append(force_final)
        return self._next(messages, tool_schemas, force_final)


def build_llm(kind: str = "auto", **kwargs: Any) -> LLMClient:
    """工厂。kind: auto | openai | heuristic。

    ``auto``：环境变量齐全就用真实模型，否则降级为 HeuristicLLM 并打 warning，
    保证 clone 下来零配置即可运行（但会明确提示当前是演示模式）。
    """
    kind = (kind or "auto").lower()
    if kind == "heuristic":
        return HeuristicLLM()
    if kind == "openai":
        return OpenAICompatLLM(**kwargs)
    if kind == "auto":
        try:
            return OpenAICompatLLM(**kwargs)
        except LLMConfigError as exc:
            logger.warning("未配置 LLM 环境变量（%s），降级为零密钥 HeuristicLLM 演示模式。", exc)
            print("[mini-agent] 未检测到 LLM 配置，已启用 HeuristicLLM 零密钥演示模式。"
                  "配置方式见 README / .env.example。")
            return HeuristicLLM()
    raise ValueError(f"未知 LLM 类型: {kind}")
