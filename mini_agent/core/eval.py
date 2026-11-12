"""Harness 评测闭环：case 定义 → 批量执行 → 结构化断言 → 录制 / 确定性重放。

为什么放在 ``core/`` 而不只写在测试里：批量评测是 **Agent Harness 的一等契约**
（harness 与「一个 agent demo」的分水岭），调用方在 CI、回归、接入真实模型后
都应能直接复用，不必依赖 unittest。但它**不新增 CLI 入口**，由
``tests/test_harness_eval.py`` 驱动，保持仓库仍是 2 个运行入口。

重放原理：``RecordingLLM`` 透明录下每次 ``LLMClient.generate()`` 的原始响应，
之后用 ``ScriptedLLM`` 按原顺序喂回。Runtime 只通过 ``generate()`` 认识模型，
因此工具调用序列、context 演化、最终答案与会话状态都能离线复现 ——
与当时接的是真实模型还是离线替身无关，也不需要 API Key。

限制：录制的是模型输出序列，不录制工具副作用（如外部 API 的实时返回）。
内置 case 全部使用本地 mock 工具，因此可完全复现；接入有副作用的工具时，
重放得到的是「同一决策轨迹」，不保证工具返回内容相同。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .llm import ScriptedLLM
from .types import LLMResponse, ToolCall

__all__ = [
    "EvalCase", "CaseResult", "SuiteResult", "RecordingLLM",
    "load_cases", "run_case", "run_suite", "record_and_replay",
]


@dataclass
class EvalCase:
    """一个可复现的 Harness 验收场景。

    ``interference`` 用于多 session 隔离：先在其他 session 制造内容，
    再跑主 ``turns``，最后用 ``forbidden_context`` 断言主会话没有串台。
    """

    id: str
    session_id: str
    turns: list[str]
    expect_tools: list[str] = field(default_factory=list)          # 跨轮累计的工具名序列
    expect_answer_contains: list[str] = field(default_factory=list)
    expect_todos: list[str] | None = None                           # None = 不检查
    forbidden_context: list[str] = field(default_factory=list)
    expect_status: str = "ok"
    interference: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(data["id"]),
            session_id=str(data.get("session_id", "eval/default")),
            turns=[str(x) for x in data.get("turns", [])],
            expect_tools=list(data.get("expect_tools", [])),
            expect_answer_contains=list(data.get("expect_answer_contains", [])),
            expect_todos=(list(data["expect_todos"]) if "expect_todos" in data else None),
            forbidden_context=list(data.get("forbidden_context", [])),
            expect_status=str(data.get("expect_status", "ok")),
            interference=[dict(x) for x in data.get("interference", [])],
        )


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    answer_text: str = ""
    todos: list[str] = field(default_factory=list)
    trace_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteResult:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed_count == self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed_count,
            "all_passed": self.all_passed,
            "results": [r.to_dict() for r in self.results],
        }


def load_cases(path: str | Path) -> list[EvalCase]:
    """从 JSON 加载 case 列表（也接受 ``{"cases": [...]}`` 包装）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    return [EvalCase.from_dict(item) for item in data]


def run_case(runtime: Any, case: EvalCase) -> CaseResult:
    """跑一个 case，返回机器可读的结构化结果；断言失败不抛异常，只记进 failures。"""
    failures: list[str] = []
    tools: list[str] = []
    statuses: list[str] = []
    answers: list[str] = []
    summaries: list[dict[str, Any]] = []

    # 先让其他 session 产生内容，制造隔离干扰
    for item in case.interference:
        sid = str(item.get("session_id", "eval/interference"))
        for text in item.get("turns", []):
            runtime.run(sid, str(text))

    for text in case.turns:
        result = runtime.run(case.session_id, text)
        statuses.append(result.status)
        answers.append(result.answer)
        tools.extend(str(e.get("name")) for e in result.trace.tool_calls())
        summaries.append(result.trace.summary())
        if result.status != case.expect_status:
            failures.append(
                f"第 {len(statuses)} 轮 status={result.status}，期望 {case.expect_status}"
            )

    if tools != case.expect_tools:
        failures.append(f"工具序列 {tools} != 期望 {case.expect_tools}")

    joined = "\n".join(answers)
    for fragment in case.expect_answer_contains:
        if fragment not in joined:
            failures.append(f"答案缺少片段 {fragment!r}")

    session = runtime.sessions.get(case.session_id)
    todos = [str(t.get("item")) for t in (session.state.get("todos", []) if session else [])]
    if case.expect_todos is not None and todos != case.expect_todos:
        failures.append(f"todos {todos} != 期望 {case.expect_todos}")

    if case.forbidden_context and session is not None:
        ctx_text = "\n".join(m.content for m in session.context.messages)
        for fragment in case.forbidden_context:
            if fragment in ctx_text:
                failures.append(f"context 串入了其他会话内容 {fragment!r}")

    return CaseResult(
        case_id=case.id,
        passed=not failures,
        failures=failures,
        tool_sequence=tools,
        statuses=statuses,
        answer_text=joined,
        todos=todos,
        trace_summaries=summaries,
    )


def run_suite(cases: list[EvalCase], runtime_factory: Callable[[], Any]) -> SuiteResult:
    """批量跑 case；每个 case 用独立 runtime，避免 session 状态互相污染。"""
    return SuiteResult(results=[run_case(runtime_factory(), case) for case in cases])


class RecordingLLM:
    """透明包装任意 LLMClient，录下每次 generate 的原始响应。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.protocol_name = getattr(inner, "protocol_name", "json")
        self.recorded: list[dict[str, Any]] = []

    def generate(self, messages, tool_schemas, force_final: bool = False) -> LLMResponse:
        response = self.inner.generate(messages, tool_schemas, force_final=force_final)
        self.recorded.append({
            "raw": response.raw,
            "native_tool_calls": [asdict(tc) for tc in response.native_tool_calls],
            "force_final": force_final,
        })
        return response


def record_and_replay(case: EvalCase, runtime_factory: Callable[[], Any]) -> dict[str, Any]:
    """录制一次真实 run 的模型输出，再用 ScriptedLLM 离线重放并比对轨迹。

    比对的是 harness 可观测契约：工具调用序列、每轮 status、最终答案、会话 todos。
    trace 里的 run_id / 时间戳天然不同，不参与比对。
    """
    first_runtime = runtime_factory()
    recorder = RecordingLLM(first_runtime.llm)
    first_runtime.llm = recorder
    first_runtime.protocol = recorder.protocol_name
    original = run_case(first_runtime, case)

    scripted = [
        LLMResponse(
            raw=item["raw"],
            native_tool_calls=[ToolCall(**d) for d in item.get("native_tool_calls", [])],
        )
        for item in recorder.recorded
    ]
    replay_runtime = runtime_factory()
    replay_runtime.llm = ScriptedLLM(scripted)
    replay_runtime.protocol = recorder.protocol_name
    replayed = run_case(replay_runtime, case)

    differences: list[str] = []
    for label, before, after in (
        ("tool_sequence", original.tool_sequence, replayed.tool_sequence),
        ("statuses", original.statuses, replayed.statuses),
        ("answer_text", original.answer_text, replayed.answer_text),
        ("todos", original.todos, replayed.todos),
    ):
        if before != after:
            differences.append(f"{label}: {before!r} != {after!r}")

    return {
        "case_id": case.id,
        "recorded_responses": len(recorder.recorded),
        "match": not differences,
        "differences": differences,
        "original": {
            "tool_sequence": original.tool_sequence,
            "statuses": original.statuses,
            "todos": original.todos,
        },
        "replayed": {
            "tool_sequence": replayed.tool_sequence,
            "statuses": replayed.statuses,
            "todos": replayed.todos,
        },
    }
