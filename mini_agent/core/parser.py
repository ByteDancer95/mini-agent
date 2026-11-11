"""LLM 输出解析：从模型回复里提取 思考过程 / 工具调用 / 最终答案。

容错等级（从严格到宽松）：
1. 原生 function calling 的 tool_calls；
2. 完整 JSON 对象；
3. 去掉 ```json 代码围栏后的 JSON；
4. 从前后夹带解释文字中按「括号配对」抠出 JSON 对象；
5. 完全不是 JSON → 降级为「纯文本最终答案」，并打 parse_warning（不崩溃）。

同时兼容多种常见字段名别名（arguments/args/parameters、content/answer/final_answer...），
因为真实模型经常不严格遵守协议。
"""
from __future__ import annotations

import json
import re
from typing import Any

from .types import AgentOutput, LLMResponse, ToolCall

__all__ = ["OutputParser", "parse_llm_output", "iter_json_objects", "strip_code_fence"]

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON|text)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """去掉包住整个输出的 Markdown 代码围栏。"""
    m = _FENCE_RE.match(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def iter_json_objects(text: str):
    """按大括号配对扫描文本中所有可能的 JSON 对象（跳过字符串内的括号）。"""
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text or ""):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, TypeError):
            return None
    return None


def _normalize_tool_call(raw: Any) -> ToolCall | None:
    d = _as_dict(raw) if not isinstance(raw, dict) else raw
    if not isinstance(d, dict):
        return None

    # 兼容 OpenAI native function calling 的 {"function": {...}} 形状
    fn = d.get("function")
    if isinstance(fn, dict):
        name = fn.get("name") or d.get("name")
        args = fn.get("arguments", fn.get("args", fn.get("parameters")))
    else:
        name = d.get("name") or d.get("tool") or d.get("tool_name") or d.get("action")
        args = (
            d.get("arguments", d.get("args", d.get("parameters", d.get("input"))))
        )
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = _as_dict(args) if args is not None else {}
    if arguments is None:
        arguments = {}
    cid = d.get("id") or d.get("call_id")
    kwargs: dict[str, Any] = {"name": name.strip(), "arguments": arguments}
    if cid:
        kwargs["call_id"] = str(cid)
    return ToolCall(**kwargs)


def parse_llm_output(raw: str) -> AgentOutput:
    """解析文本输出。永远返回 AgentOutput，不抛异常。"""
    raw = raw or ""
    cleaned = strip_code_fence(raw)
    obj: dict[str, Any] | None = None
    warning: str | None = None

    try:
        candidate = json.loads(cleaned)
        if isinstance(candidate, dict):
            obj = candidate
    except (ValueError, TypeError):
        pass

    if obj is None:
        for chunk in iter_json_objects(cleaned):
            try:
                candidate = json.loads(chunk)
            except (ValueError, TypeError):
                continue
            if isinstance(candidate, dict) and (
                "action" in candidate or "tool_calls" in candidate
                or "final_answer" in candidate or "type" in candidate
            ):
                obj = candidate
                warning = "JSON 前后夹带了额外文字，已按括号配对提取"
                break

    if obj is None:
        # 降级：把整段文本当作最终答案，保证 agent 不会因为格式问题卡死
        text = cleaned.strip()
        return AgentOutput(
            thought="", final_answer=text or "（模型未返回内容）", raw=raw,
            parse_warning="输出不是协议 JSON，已降级为纯文本最终答案",
            source="plain",
        )

    return _from_object(obj, raw=raw, warning=warning)


def _from_object(
    obj: dict[str, Any], *, raw: str = "", warning: str | None = None,
    source: str = "json",
) -> AgentOutput:
    thought = str(obj.get("thought") or obj.get("reasoning") or obj.get("think") or "")

    action = obj.get("action")
    action_type = ""
    action_payload: dict[str, Any] = {}
    if isinstance(action, dict):
        action_payload = action
        action_type = str(action.get("type") or action.get("action") or "").lower()
    elif isinstance(action, str):
        action_type = action.lower()

    if not action_type:
        action_type = str(obj.get("type") or obj.get("decision") or "").lower()
        if not action_payload:
            action_payload = obj

    # ---- 收集工具调用 ----
    raw_calls = (
        action_payload.get("tool_calls")
        or action_payload.get("calls")
        or obj.get("tool_calls")
        or obj.get("tools")
        or []
    )
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    tool_calls: list[ToolCall] = []
    if isinstance(raw_calls, list):
        for item in raw_calls:
            call = _normalize_tool_call(item)
            if call is not None:
                tool_calls.append(call)

    if action_type in {"tool", "function", "call", "tool_call"} and not tool_calls:
        # 单工具简写: {"action": {"type":"tool","name":"x","arguments":{...}}}
        single = _normalize_tool_call(action_payload)
        if single and single.name not in {"tool", "function", "call"}:
            tool_calls.append(single)

    if tool_calls:
        return AgentOutput(
            thought=thought, tool_calls=tool_calls, raw=raw,
            parse_warning=warning, source=source,
        )

    # ---- 收集最终答案 ----
    content = None
    for key in ("content", "answer", "final_answer", "final", "text", "message", "response"):
        if key in action_payload and isinstance(action_payload[key], (str, int, float)):
            content = str(action_payload[key])
            break
    if content is None:
        for key in ("final_answer", "content", "answer", "final", "text", "message"):
            if key in obj and isinstance(obj[key], (str, int, float)):
                content = str(obj[key])
                break
    if content is None and action_type in {"final", "answer", "done", "finish"}:
        content = ""
    if content is not None:
        return AgentOutput(
            thought=thought, final_answer=content, raw=raw,
            parse_warning=warning, source=source,
        )

    return AgentOutput(
        thought=thought, raw=raw,
        parse_warning=warning or "JSON 中既没有 tool_calls 也没有 final 内容",
        source=source,
    )


class OutputParser:
    """解析入口。优先使用 native tool_calls，其次解析文本。"""

    def parse(self, response: "LLMResponse | str") -> AgentOutput:
        if isinstance(response, str):
            return parse_llm_output(response)
        if response.native_tool_calls:
            return AgentOutput(
                thought="", tool_calls=list(response.native_tool_calls),
                raw=response.raw, source="native",
            )
        return parse_llm_output(response.raw)
