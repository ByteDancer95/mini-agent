"""「小田 · 中文工作助理」：把 core/ 的引擎装成一台具名整机。

这里只定义 **Agent 本身**（身份、工具集、会话策略、ask 接口），不含运行入口：
交互与演示统一走 ``python -m mini_agent.cli``，协议证据走 ``walkthrough.py``。
"""
from __future__ import annotations

from pathlib import Path

from .core.runtime import RuntimeConfig, build_default_runtime
from .core.types import RunResult

__all__ = ["AGENT_NAME", "PERSONA", "MiniWorkAgent", "create_agent"]

AGENT_NAME = "小田 · 中文工作助理 Agent"

PERSONA = """你叫「小田」，是一个中文工作助理 Agent。
你会自己判断信息是否足够：不足就调用工具取证，足够就直接回答。
涉及天气、资料、计算、待办时必须先取证再下结论，严禁编造工具返回结果。
回答用中文、结论先行、尽量简洁；不确定的地方如实说明。"""


class MiniWorkAgent:
    """一个具名、开箱即用的 Agent 实例（而不是可组装的框架）。"""

    def __init__(
        self,
        *,
        llm_kind: str = "auto",
        llm=None,
        persist_dir: str | Path | None = None,
        trace_path: str | Path | None = None,
        max_tool_rounds: int = 5,
        persona: str = PERSONA,
    ) -> None:
        self.runtime = build_default_runtime(
            llm_kind="heuristic" if llm is not None else llm_kind,
            persist_dir=persist_dir,
            trace_path=trace_path,
            echo_trace=False,
            config=RuntimeConfig(max_tool_rounds=max_tool_rounds,
                                 extra_instructions=persona),
        )
        if llm is not None:
            # 允许注入自定义后端（测试 / 私有端点），换后端不需要改 Runtime
            self.runtime.llm = llm
            self.runtime.protocol = getattr(llm, "protocol_name", "json")

    @property
    def name(self) -> str:
        return AGENT_NAME

    @property
    def backend(self) -> str:
        return type(self.runtime.llm).__name__

    @property
    def is_real_llm(self) -> bool:
        return self.backend == "OpenAICompatLLM"

    def tools(self) -> list[str]:
        return self.runtime.registry.names()

    def describe(self) -> str:
        mode = "真实 LLM，基于工具 Schema 自主决策" if self.is_real_llm else \
               "离线协议验证替身（非真实模型）"
        return (
            f"Agent    : {self.name}\n"
            f"决策后端 : {self.backend} —— {mode}\n"
            f"工具     : {', '.join(self.tools())}\n"
            f"最大轮次 : {self.runtime.config.max_tool_rounds}"
        )

    def ask(self, session_id: str, text: str) -> RunResult:
        """问一句话，拿到带完整工具调用 trace 的结果。"""
        return self.runtime.run(session_id, text)


def create_agent(**kwargs) -> MiniWorkAgent:
    return MiniWorkAgent(**kwargs)
