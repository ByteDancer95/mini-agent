"""「某一个具体的 Agent」：小田 · 中文工作助理。

分工说明（回应「题目是不是要一个具体的 agent」这个疑问）：

* ``mini_agent/core/`` 是**引擎**：Runtime loop、工具注册、解析、会话、context、trace。
  它自己不是交付物，也不打算被别的项目当框架引用。
* 本文件是把引擎装成**一台整机**的地方：固定的名字、身份（persona）、工具集、
  会话策略与运行入口。``python -m mini_agent.agent`` 起来就是一个可以直接对话、
  能自己决定调不调工具、能跨轮追问的具体 Agent。

因此「一个具体可实现的 Agent」= 本文件这个实体 + 它内核里自己实现的 Runtime。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.llm import check_llm_config, load_env_file
from .core.runtime import RuntimeConfig, build_default_runtime
from .core.types import RunResult

__all__ = ["AGENT_NAME", "PERSONA", "MiniWorkAgent", "create_agent"]

AGENT_NAME = "小田 · 中文工作助理 Agent"

PERSONA = """你叫「小田」，是一个中文工作助理 Agent。
你会自己判断信息是否足够：不足就调用工具取证，足够就直接回答。
涉及天气、资料、计算、待办时必须先取证再下结论，严禁编造工具返回结果。
回答用中文、结论先行、尽量简洁；不确定的地方如实说明。"""

SCRIPTED_TURNS = [
    "查一下广州今天的天气，顺便记个待办：明天给大棚番茄喷药",
    "再帮我算一下 18*24 是多少",
    "继续，把我的待办列出来",
    "不用工具，用一句话说说你刚才都做了什么",
]


class MiniWorkAgent:
    """一个具名的、开箱即用的 Agent 实例（而不是一个可组装的库）。"""

    def __init__(
        self,
        *,
        llm_kind: str = "auto",
        persist_dir: str | Path | None = None,
        trace_path: str | Path | None = None,
        max_tool_rounds: int = 5,
        persona: str = PERSONA,
    ) -> None:
        self.runtime = build_default_runtime(
            llm_kind=llm_kind,
            persist_dir=persist_dir,
            trace_path=trace_path,
            echo_trace=False,
            config=RuntimeConfig(max_tool_rounds=max_tool_rounds,
                                 extra_instructions=persona),
        )

    # ---- 身份 -------------------------------------------------------------
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
               "离线协议验证替身（非真实模型，仅用于验证 loop / 解析 / 隔离）"
        return (
            f"Agent    : {self.name}\n"
            f"决策后端 : {self.backend} —— {mode}\n"
            f"工具     : {', '.join(self.tools())}\n"
            f"最大轮次 : {self.runtime.config.max_tool_rounds}\n"
            f"活跃会话 : {len(self.runtime.sessions.ids())}"
        )

    # ---- 唯一对外动作 -----------------------------------------------------
    def ask(self, session_id: str, text: str) -> RunResult:
        """问这个 Agent 一句话，拿到带完整工具调用 trace 的结果。"""
        return self.runtime.run(session_id, text)

    def scripted_conversation(self, session_id: str = "demo"):
        """一段固定的多轮对话，用来证明这是一台能连续工作的整机。"""
        for text in SCRIPTED_TURNS:
            yield text, self.ask(session_id, text)


def create_agent(**kwargs) -> MiniWorkAgent:
    return MiniWorkAgent(**kwargs)


def _repl(agent: MiniWorkAgent, session_id: str) -> None:
    print(agent.describe())
    print("直接输入问题；/use <id> 切换会话，/quit 退出。")
    while True:
        try:
            line = input(f"[{session_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not line:
            continue
        if line.lower() in {"/quit", "/exit", "/q"}:
            print("bye.")
            return
        if line.lower().startswith("/use "):
            session_id = line.split(maxsplit=1)[1].strip()
            agent.runtime.sessions.get_or_create(session_id)
            print(f"已切换到会话 {session_id}")
            continue
        result = agent.ask(session_id, line)
        print(f"\n{result.answer}\n[status={result.status} turns={result.turns}]")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="小田 · 中文工作助理 Agent（一个具体的 Agent 实例）")
    p.add_argument("--llm", default="auto", choices=["auto", "openai", "heuristic"])
    p.add_argument("--session", default="demo")
    p.add_argument("--persist", default=None, help="会话持久化目录")
    p.add_argument("--trace-file", default=None, help="trace JSONL 输出路径")
    p.add_argument("--repl", action="store_true", help="进入交互对话（默认跑固定多轮演示）")
    p.add_argument("--trace", action="store_true", help="同时打印每轮工具调用 trace")
    p.add_argument("--check-llm", action="store_true", help="只读自检真实 LLM 配置后退出")
    args = p.parse_args(argv)

    load_env_file(Path(__file__).resolve().parents[1] / ".env")
    if args.check_llm:
        status = check_llm_config()
        print("真实 LLM 配置自检（只读，未发起请求）")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return 0 if status["ok"] else 1

    agent = create_agent(llm_kind=args.llm, persist_dir=args.persist,
                         trace_path=args.trace_file)
    if args.repl:
        _repl(agent, args.session)
        return 0

    print("=" * 70)
    print(agent.describe())
    print("=" * 70)
    for i, (text, result) in enumerate(agent.scripted_conversation(args.session), 1):
        print(f"\n--- 第 {i} 轮")
        print(f"用户: {text}")
        print(f"小田: {result.answer}")
        print(f"[status={result.status} turns={result.turns} "
              f"工具调用={len(result.trace.tool_calls())}]")
        if args.trace:
            print(result.trace.to_text())
    todos = agent.runtime.sessions.get_or_create(args.session).state.get("todos", [])
    print(f"\n会话 {args.session} 最终待办: {[t['item'] for t in todos]}")
    print("（以上为同一个 Agent 跨 4 轮连续对话，历史与待办一直保留）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
