"""全局数据结构。所有对象都是纯 dataclass，便于序列化到 session 持久化文件。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Role", "Message", "ToolCall", "AgentOutput", "ToolSpec", "ToolResult",
    "ToolContext", "LLMResponse", "RunResult",
]


def _short_id() -> str:
    return uuid.uuid4().hex[:10]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """内部统一的消息格式（与具体 LLM 厂商解耦）。

    ``meta`` 只用于本地审计/展示，不会进入发给 LLM 的 payload。
    """

    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_short_id)
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if isinstance(self.role, Role):
            self.role = self.role.value

    @property
    def role_str(self) -> str:
        return self.role.value if isinstance(self.role, Role) else str(self.role)

    def to_chat(self) -> dict[str, Any]:
        """转换成 OpenAI Chat Completions 兼容的一条消息。"""
        out: dict[str, Any] = {"role": self.role_str, "content": self.content}
        if self.name:
            out["name"] = self.name
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role_str,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "meta": self.meta,
            "id": self.id,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            name=d.get("name"),
            tool_call_id=d.get("tool_call_id"),
            meta=d.get("meta") or {},
            id=d.get("id") or _short_id(),
            ts=d.get("ts", time.time()),
        )


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=_short_id)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments), "call_id": self.call_id}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        return cls(
            name=str(d.get("name", "")),
            arguments=dict(d.get("arguments") or {}),
            call_id=str(d.get("call_id") or _short_id()),
        )


@dataclass
class ToolSpec:
    """工具注册单元: 名称 + 描述 + 参数 JSON Schema。

    ``schema_dict()`` 的产物会被序列化进系统提示词（json 协议），
    或作为 ``tools`` 参数传给支持 native function calling 的模型。
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def schema_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def native_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, "result": self.data}
        return {"ok": False, "error": self.error or "unknown tool error", "result": None}

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error": self.error,
                "elapsed_ms": self.elapsed_ms}


@dataclass
class ToolContext:
    """执行工具时的运行时上下文。

    ``state`` 是 **会话私有** 的可变字典；todo 等有状态工具只读写这里，
    从而天然实现多 session 隔离，不依赖任何全局单例。
    """

    session_id: str
    state: dict[str, Any] = field(default_factory=dict)
    docs_root: str | None = None


@dataclass
class AgentOutput:
    """LLM 输出解析结果: 思考过程 / 工具调用 / 最终答案 三选一或组合。"""

    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str | None = None
    raw: str = ""
    parse_warning: str | None = None
    source: str = "json"  # json | native | plain | repaired

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None and not self.tool_calls

    @property
    def is_usable(self) -> bool:
        return self.is_final or self.has_tool_calls

    def decision(self) -> str:
        if self.has_tool_calls:
            return "tool"
        if self.final_answer is not None:
            return "final"
        return "invalid"


@dataclass
class LLMResponse:
    """LLM 客户端统一返回。

    ``native_tool_calls`` 仅在启用原生 function calling 时非空，
    此时 parser 优先使用它，其余情况解析 ``raw`` 文本。
    """

    raw: str = ""
    native_tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None


@dataclass
class RunResult:
    session_id: str
    answer: str
    status: str = "ok"  # ok | loop_limit | error
    turns: int = 0
    run_id: str = ""
    error: str | None = None
    trace: Any = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"
