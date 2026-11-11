"""工具调用 trace 与执行日志。

* 每次 ``AgentRuntime.run`` 生成一个 ``Trace``，按时间顺序记录所有关键事件；
* ``TraceSink`` 把事件写成 JSONL 文件（一行一事件，便于机器分析），并可回显可读摘要；
* 同时通过标准库 ``logging`` 输出，方便接现有日志系统。
* 日志中**永不记录** api_key 等敏感字段（Runtime 不把凭据放进 payload）。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = ["EventType", "TraceEvent", "Trace", "TraceSink"]

logger = logging.getLogger("mini_agent.trace")


class EventType(str, Enum):
    RUN_START = "run_start"
    USER_INPUT = "user_input"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    PARSE = "parse"
    THOUGHT = "thought"
    DECISION = "decision"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMPACT = "compact"
    REPAIR = "repair"
    LOOP_LIMIT = "loop_limit"
    FINAL = "final"
    ERROR = "error"
    WARNING = "warning"
    SESSION_SAVE = "session_save"


@dataclass
class TraceEvent:
    type: str
    run_id: str
    session_id: str
    turn: int = 0
    ts: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trace:
    def __init__(self, session_id: str, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.session_id = session_id
        self.events: list[TraceEvent] = []
        self.started_at = time.time()

    def add(self, type: "EventType | str", turn: int = 0, **payload: Any) -> TraceEvent:
        name = type.value if isinstance(type, EventType) else str(type)
        event = TraceEvent(type=name, run_id=self.run_id,
                           session_id=self.session_id, turn=turn, payload=payload)
        self.events.append(event)
        return event

    # ---- 便捷查询（测试与 CLI 使用） ---------------------------------------
    def of_type(self, type: "EventType | str") -> list[TraceEvent]:
        name = type.value if isinstance(type, EventType) else str(type)
        return [e for e in self.events if e.type == name]

    def tool_calls(self) -> list[dict[str, Any]]:
        return [dict(e.payload) for e in self.of_type(EventType.TOOL_CALL)]

    def tool_results(self) -> list[dict[str, Any]]:
        return [dict(e.payload) for e in self.of_type(EventType.TOOL_RESULT)]

    def errors(self) -> list[TraceEvent]:
        return self.of_type(EventType.ERROR)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "events": len(self.events),
            "tool_calls": len(self.tool_calls()),
            "llm_calls": len(self.of_type(EventType.LLM_CALL)),
            "compactions": len(self.of_type(EventType.COMPACT)),
            "errors": len(self.errors()),
            "elapsed_ms": round((time.time() - self.started_at) * 1000, 2),
        }

    def to_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    def to_text(self) -> str:
        """人读的紧凑执行日志。"""
        lines = [f"trace run={self.run_id} session={self.session_id}"]
        for e in self.events:
            name = e.type
            p = e.payload
            if name == EventType.TOOL_CALL.value:
                detail = f"{p.get('name')}({json.dumps(p.get('arguments'), ensure_ascii=False)})"
            elif name == EventType.TOOL_RESULT.value:
                detail = (f"{p.get('name')} ok={p.get('ok')} "
                          f"{p.get('elapsed_ms')}ms {str(p.get('preview'))[:100]}")
            elif name == EventType.DECISION.value:
                detail = str(p.get("decision"))
            elif name == EventType.PARSE.value:
                detail = f"source={p.get('source')} warning={p.get('warning')}"
            elif name in {EventType.FINAL.value, EventType.ERROR.value,
                          EventType.LOOP_LIMIT.value, EventType.COMPACT.value}:
                detail = str(p.get("answer") or p.get("error") or p.get("stats") or "")[:120]
            else:
                detail = ""
            lines.append(f"  [t{e.turn:02d}] {name:<14} {detail}")
        return "\n".join(lines)


class TraceSink:
    """把 Trace 写到 JSONL 文件 / 标准日志 / 可选的 stdout 回显。"""

    def __init__(
        self,
        path: str | Path | None = None,
        echo: bool = False,
        to_logger: bool = True,
    ) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.to_logger = to_logger

    def emit(self, trace: Trace) -> None:
        if self.path:
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    for event in trace.events:
                        fh.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")
            except OSError as exc:
                logger.warning("trace 写入失败: %s", exc)
        if self.to_logger:
            for event in trace.events:
                logger.info("%s | %s | t%s | %s", event.run_id, event.type,
                            event.turn, json.dumps(event.payload, ensure_ascii=False, default=str)[:400])
        if self.echo:
            print(trace.to_text())
