"""唯一交互入口：REPL / 双窗口隔离演示 / 真实 LLM 配置自检 / trace。

    python -m mini_agent.cli                      # 交互对话
    python -m mini_agent.cli --demo              # 题目指定的多 session 隔离演示
    python -m mini_agent.cli --check-llm         # 只读自检（不请求、不打印密钥）
    python -m mini_agent.cli --trace --trace-file logs/agent_trace.jsonl
    python -m mini_agent.cli --llm openai        # 使用 .env 配置的真实 LLM

REPL 命令：/list /use <id> /new <id> /todos /history /tools /trace on|off /reset /quit
证据导出（另一个入口）：python -m mini_agent.walkthrough
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .core.context import ContextPolicy
from .core.llm import check_llm_config, load_env_file
from .core.runtime import AgentRuntime, RuntimeConfig, build_default_runtime

BANNER = """\
============================================================
 小田 · 工作助理 Agent  |  Runtime 从零实现，无 agent 框架依赖
 输入 /help 查看命令，/quit 退出
============================================================"""

WINDOW_1 = "userA/window1"
WINDOW_2 = "userA/window2"

# 题目场景：窗口1 查天气记待办 / 窗口2 写周报记待办；之后各自继续对话互不干扰
DEMO_STEPS: list[tuple[str, str, str]] = [
    (WINDOW_1, "查一下广州今天的天气，顺便记个待办：明天给大棚番茄喷药",
     "并行调用 weather + todo.add"),
    (WINDOW_2, "帮我写一份本周周报，另外记个待办：周五前提交课程报告",
     "调用 read_docs + todo.add，与窗口1 隔离"),
    (WINDOW_1, "继续，把我刚才记的待办列出来", "带工具追问 todo.list"),
    (WINDOW_2, "继续，不用工具，用一句话复述刚才周报的要点", "纯对话追问，不调工具"),
    (WINDOW_1, "再帮我算一下 18*24 是多少", "换话题继续，历史与待办仍保留"),
]


def run_demo(runtime: AgentRuntime | None = None) -> int:
    """双窗口隔离演示：同一 Agent 的两个 session 分别随时继续对话。"""
    rt = runtime or build_default_runtime(llm_kind="heuristic")
    print("=" * 66)
    print(f"多 Session 隔离演示  (后端 = {type(rt.llm).__name__})")
    print("=" * 66)
    results = []
    for i, (sid, text, note) in enumerate(DEMO_STEPS, 1):
        result = rt.run(sid, text)
        results.append(result)
        names = [e["name"] for e in result.trace.tool_calls()]
        print(f"\nstep {i} [{sid}] {note}")
        print(f"  > {text}")
        print(f"  工具: {', '.join(names) or '(无)'}  status={result.status} turns={result.turns}")
        print("  " + result.answer.replace("\n", "\n  ")[:300])

    s1 = rt.sessions.get_or_create(WINDOW_1)
    s2 = rt.sessions.get_or_create(WINDOW_2)
    t1 = [t["item"] for t in s1.state.get("todos", [])]
    t2 = [t["item"] for t in s2.state.get("todos", [])]
    u1 = [m.content for m in s1.context.messages if m.role == "user"]
    u2 = [m.content for m in s2.context.messages if m.role == "user"]
    text1 = "\n".join(m.content for m in s1.context.messages)
    text2 = "\n".join(m.content for m in s2.context.messages)

    print("\n" + "=" * 66)
    print("隔离验证结果")
    print("=" * 66)
    print(f"  window1 todos : {t1}")
    print(f"  window2 todos : {t2}")
    print(f"  window1 用户消息数: {len(u1)}  window2 用户消息数: {len(u2)}")
    checks = {
        "两个 session 的待办互不相同": t1 != t2,
        "窗口1 待办包含喷药": any("喷药" in x for x in t1),
        "窗口2 待办包含课程报告": any("课程报告" in x for x in t2),
        "窗口1 历史不含窗口2 用户输入": not any(x in text1 for x in u2),
        "窗口2 历史不含窗口1 用户输入": not any(x in text2 for x in u1),
        "窗口1 累计 3 轮 / 窗口2 累计 2 轮": len(u1) == 3 and len(u2) == 2,
        "所有 run 都成功": all(r.status == "ok" for r in results),
        "至少发生了工具调用": sum(len(r.trace.tool_calls()) for r in results) >= 5,
    }
    for label, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed = [k for k, v in checks.items() if not v]
    print(f"\n结论: {'全部通过' if not failed else '失败项: ' + ', '.join(failed)}")
    return 0 if not failed else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini-Agent CLI")
    p.add_argument("--llm", default="auto", choices=["auto", "openai", "heuristic"])
    p.add_argument("--session", default="default")
    p.add_argument("--persist", default=None, help="session 持久化目录")
    p.add_argument("--trace-file", default=None, help="trace JSONL 输出路径")
    p.add_argument("--trace", action="store_true", help="每轮打印工具调用 trace")
    p.add_argument("--verbose", action="store_true", help="stderr 输出 INFO 级日志")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--include-thought", action="store_true",
                   help="把思考过程也放进 LLM context（默认只进审计/日志）")
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--demo", action="store_true", help="运行双窗口隔离演示后退出")
    p.add_argument("--check-llm", action="store_true",
                   help="只读自检真实 LLM 配置（不发起请求、不打印密钥）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_env_file(Path(__file__).resolve().parents[1] / ".env")

    if args.check_llm:
        status = check_llm_config()
        print("真实 LLM 配置自检（只读，未发起请求）")
        for key, value in status.items():
            print(f"  {key}: {value}")
        print("结论:", "已配置，可用 --llm openai 让模型基于 Schema 自主决策"
              if status["ok"] else "未配置，当前只能使用离线协议验证后端")
        return 0 if status["ok"] else 1

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    policy = ContextPolicy(include_thought_in_llm_context=args.include_thought,
                           compress=not args.no_compress)
    runtime = build_default_runtime(
        llm_kind=args.llm, persist_dir=args.persist, trace_path=args.trace_file,
        echo_trace=False, config=RuntimeConfig(max_tool_rounds=args.max_rounds),
        context_policy=policy,
    )
    if args.demo:
        return run_demo(runtime)

    backend = type(runtime.llm).__name__
    real_llm = backend == "OpenAICompatLLM"
    print(BANNER)
    print(f"决策后端 : {backend}（{'真实 LLM，基于工具 Schema 自主决策' if real_llm else '离线协议验证替身，非真实模型'}）")
    if not real_llm:
        print("           配置 .env 后加 --llm openai 即为真实模型；--check-llm 可自检")
    print(f"工具     : {', '.join(runtime.registry.names())}")
    print(f"Session  : {args.session}   最大工具轮次: {args.max_rounds}")
    print(f"持久化   : {args.persist or '关闭（仅内存）'}")
    current, show_trace = args.session, args.trace

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
            cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
            if cmd in {"/quit", "/exit", "/q"}:
                print("bye.")
                break
            elif cmd in {"/help", "/?"}:
                print(__doc__)
            elif cmd == "/list":
                print("sessions:", runtime.sessions.ids() or "(空)")
            elif cmd in {"/use", "/new"}:
                if not arg:
                    print("用法: /use <session_id>")
                else:
                    current = arg
                    runtime.sessions.get_or_create(current)
                    print(f"已切换到 session '{current}'")
            elif cmd == "/todos":
                print(runtime.sessions.get_or_create(current).state_snapshot() or "(待办为空)")
            elif cmd == "/history":
                for m in runtime.sessions.get_or_create(current).context.messages:
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
