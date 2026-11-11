"""测试公共辅助。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

logging.getLogger("mini_agent").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_agent.core.context import ContextPolicy  # noqa: E402
from mini_agent.core.llm import HeuristicLLM, ScriptedLLM  # noqa: E402
from mini_agent.core.runtime import AgentRuntime, RuntimeConfig  # noqa: E402
from mini_agent.core.session import SessionManager  # noqa: E402
from mini_agent.core.trace import TraceSink  # noqa: E402
from mini_agent.tools.base import build_default_registry  # noqa: E402


def tool_json(thought: str = "需要调用工具", calls: list[dict] | None = None) -> str:
    return json.dumps({
        "thought": thought,
        "action": {"type": "tool", "tool_calls": calls or []},
    }, ensure_ascii=False)


def final_json(content: str, thought: str = "信息已经足够") -> str:
    return json.dumps({
        "thought": thought,
        "action": {"type": "final", "content": content},
    }, ensure_ascii=False)


def call(name: str, **kwargs) -> dict:
    return {"name": name, "arguments": kwargs}


def make_runtime(
    responses: list,
    *,
    max_tool_rounds: int = 5,
    registry=None,
    policy: ContextPolicy | None = None,
    trace_sink: TraceSink | None = None,
    persist_dir: str | Path | None = None,
    max_llm_retries: int = 1,
    max_parse_repairs: int = 1,
):
    llm = ScriptedLLM(responses)
    rt = AgentRuntime(
        llm=llm,
        registry=registry or build_default_registry(),
        sessions=SessionManager(persist_dir=persist_dir, policy=policy),
        config=RuntimeConfig(max_tool_rounds=max_tool_rounds,
                             max_llm_retries=max_llm_retries,
                             max_parse_repairs=max_parse_repairs),
        context_policy=policy,
        trace_sink=trace_sink or TraceSink(to_logger=False),
    )
    return rt, llm


def make_heuristic_runtime(**kwargs):
    llm = HeuristicLLM()
    rt = AgentRuntime(
        llm=llm,
        registry=kwargs.pop("registry", None) or build_default_registry(),
        sessions=SessionManager(persist_dir=kwargs.pop("persist_dir", None),
                                policy=kwargs.pop("policy", None)),
        config=RuntimeConfig(**kwargs),
        trace_sink=TraceSink(to_logger=False),
    )
    return rt, llm


def all_text(messages) -> str:
    return "\n".join(m.content for m in messages)
