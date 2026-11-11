"""交互式 CLI：多 session 切换、随时继续对话、trace 回显。

    python -m mini_agent.cli                      # 零密钥演示模式（未配置 LLM 时自动降级）
    python -m mini_agent.cli --llm openai        # 使用环境变量配置的真实 LLM
    python -m mini_agent.cli --demo              # 跑题目指定的双窗口隔离演示
    python -m mini_agent.cli --trace --trace-file logs/agent_trace.jsonl

REPL 内可用命令：
    /list              列出所有 session
    /use <id>          切换 session（切回即可继续对话）
    /new <id>          新建并切换
    /todos             查看当前 session 的待办
    /history           查看当前 session 的历史消息
    /trace on|off      切换 trace 打印
    /tools             列出已注册工具与 Schema
    /reset             清空当前 session 上下文
    /quit              退出
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .core.context import ContextPolicy
from .core.llm import load_env_file
from .core.runtime import RuntimeConfig, build_default_runtime
from .core.trace import TraceSink

BANNER = """\
============================================================
 Mini-Agent  |  从零实现的 Agent Runtime（无 agent 框架依赖）
 输入 /help 查看命令，/quit 退出
============================================================"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini-Agent CLI")
    p.add_argument("--llm", default="auto", choices=["auto", "openai", "heuristic"],
                   help="LLM 实现；auto=有配置用真实模型，否则零密钥演示模式")
    p.add_argument("--session", default="default", help="初始 session id")
    p.add_argument("--persist", default=None,
                   help="session 持久化目录（默认内存；指定后可跨进程继续对话）")
    p.add_argument("--trace-file", default=None, help="trace JSONL 输出路径")
    p.add_argument("--trace", action="store_true", help="每轮打印工具调用 trace")
    p.add_argument("--verbose", action="store_true", help="在 stderr 输出 INFO 级执行日志")
    p.add_argument("--max-rounds", type=int, default=5, help="单次输入最大工具轮次")
    p.add_argument("--include-thought", action="store_true",
                   help="把思考过程也放进 LLM context（默认只进审计/日志）")
    p.add_argument("--no-compress", action="store_true", help="关闭基础压缩")
    p.add_argument("--demo", action="store_true", help="运行双窗口隔离演示后退出")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_env_file(Path(__file__).resolve().parents[1] / ".env")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    policy = ContextPolicy(
        include_thought_in_llm_context=args.include_thought,
        compress=not args.no_compress,
    )
    runtime = build_default_runtime(
        llm_kind=args.llm,
        persist_dir=args.persist,
        trace_path=args.trace_file,
        echo_trace=False,
        config=RuntimeConfig(max_tool_rounds=args.max_rounds),
        context_policy=policy,
    )
    sink: TraceSink = runtime.trace_sink

    if args.demo:
        from .demo import run_demo
        return run_demo(runtime)

    print(BANNER)
    print(f"LLM      : {type(runtime.llm).__name__} (protocol={runtime.protocol})")
    print(f"工具     : {', '.join(runtime.registry.names())}")
    print(f"Session  : {args.session}   最大工具轮次: {args.max_rounds}")
    print(f"持久化   : {args.persist or '关闭（仅内存）'}")
    current = args.session
    show_trace = args.trace

    while True:
        try:
            line = input(f"\n[{current}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in {"/quit", "/exit", "/q"}:
                print("bye.")
                break
            if cmd in {"/help", "/?"}:
                print(__doc__)
            elif cmd == "/list":
                print("sessions:", runtime.sessions.ids() or "(空)")
            elif cmd in {"/use", "/new"}:
                if not arg:
                    print("用法: /use <session_id>")
                else:
                    if cmd == "/new" and runtime.sessions.exists(arg):
                        print(f"session '{arg}' 已存在，已直接切换")
                    current = arg
                    runtime.sessions.get_or_create(current)
                    print(f"已切换到 session '{current}'")
            elif cmd == "/todos":
                session = runtime.sessions.get_or_create(current)
                print(session.state_snapshot() or "(当前 session 待办为空)")
            elif cmd == "/history":
                session = runtime.sessions.get_or_create(current)
                for m in session.context.messages:
                    print(f"  [{m.role}] {m.content[:200].replace(chr(10), ' ')}")
            elif cmd == "/tools":
                for s in runtime.registry.schemas():
                    print(f"  - {s['name']}: {s['description'][:60]}")
            elif cmd == "/reset":
                runtime.sessions.delete(current)
                runtime.sessions.get_or_create(current)
                print(f"session '{current}' 已清空")
            elif cmd == "/trace":
                if arg in {"on", "off"}:
                    show_trace = arg == "on"
                    sink.echo = False  # 由下面手动打印
                    print(f"trace 打印: {'on' if show_trace else 'off'}")
                else:
                    print("用法: /trace on|off")
            else:
                print(f"未知命令 {cmd}，输入 /help 查看")
            continue

        result = runtime.run(current, line)
        print(f"\n{result.answer}")
        print(f"\n[status={result.status} turns={result.turns} run={result.run_id}]")
        if show_trace:
            print(result.trace.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
