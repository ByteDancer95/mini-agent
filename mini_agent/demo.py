"""题目指定的多 session 隔离演示（零密钥即可运行）。

用户 A 同时开两个窗口：
  window1: 查天气 + 记待办
  window2: 写周报 + 记待办
之后两个窗口分别「随时继续对话」：一个带工具追问、一个纯对话追问、再换话题追问，
验证历史、工具状态、trace 三者都互不干扰。
"""
from __future__ import annotations

from .core.runtime import AgentRuntime, build_default_runtime

WINDOW_1 = "userA/window1"
WINDOW_2 = "userA/window2"

STEPS: list[tuple[str, str, str]] = [
    (WINDOW_1, "查一下广州今天的天气，顺便记个待办：明天给大棚番茄喷药",
     "窗口1：并行调用 weather + todo.add"),
    (WINDOW_2, "帮我写一份本周周报，另外记个待办：周五前提交课程报告",
     "窗口2：调用 read_docs + todo.add，与窗口1 完全隔离"),
    (WINDOW_1, "继续，把我刚才记的待办列出来",
     "窗口1 继续对话（带工具追问 todo.list），验证只看到自己的待办"),
    (WINDOW_2, "继续，不用工具，用一句话复述刚才周报的要点",
     "窗口2 继续对话（纯对话追问，依赖历史上下文，不调工具）"),
    (WINDOW_1, "再帮我算一下 18*24 是多少",
     "窗口1 再次继续并换话题，历史与待办仍保留"),
]


def run_demo(runtime: AgentRuntime | None = None) -> int:
    rt = runtime or build_default_runtime(llm_kind="heuristic")
    print("=" * 72)
    print("Mini-Agent 多 Session 隔离演示  (LLM =", type(rt.llm).__name__, ")")
    print("=" * 72)

    results = []
    for i, (sid, text, note) in enumerate(STEPS, 1):
        print(f"\n--- step {i} | {note}")
        print(f"[{sid}] > {text}")
        result = rt.run(sid, text)
        results.append(result)
        print(f"\n{result.answer}")
        print(result.trace.to_text())
        print(f"[status={result.status} turns={result.turns}]")

    # ---- 隔离断言 ---------------------------------------------------------
    s1 = rt.sessions.get_or_create(WINDOW_1)
    s2 = rt.sessions.get_or_create(WINDOW_2)
    t1 = [t["item"] for t in s1.state.get("todos", [])]
    t2 = [t["item"] for t in s2.state.get("todos", [])]
    u1 = [m.content for m in s1.context.messages if m.role == "user"]
    u2 = [m.content for m in s2.context.messages if m.role == "user"]

    print("\n" + "=" * 72)
    print("隔离验证结果")
    print("=" * 72)
    print(f"  window1 todos : {t1}")
    print(f"  window2 todos : {t2}")
    print(f"  window1 用户消息数: {len(u1)}  window2 用户消息数: {len(u2)}")
    print(f"  window1 context 中是否出现窗口2 的输入: "
          f"{any(x in '\n'.join(m.content for m in s1.context.messages) for x in u2)}")
    print(f"  window2 context 中是否出现窗口1 的输入: "
          f"{any(x in '\n'.join(m.content for m in s2.context.messages) for x in u1)}")

    checks = {
        "两个 session 的待办互不相同": t1 != t2,
        "窗口1 待办包含喷药": any("喷药" in x for x in t1),
        "窗口2 待办包含课程报告": any("课程报告" in x for x in t2),
        "窗口1 历史不含窗口2 用户输入": not any(x in u1 for x in u2),
        "窗口2 历史不含窗口1 用户输入": not any(x in u2 for x in u1),
        "窗口1 累计 3 轮用户输入": len(u1) == 3,
        "窗口2 累计 2 轮用户输入": len(u2) == 2,
        "所有 run 都成功": all(r.status == "ok" for r in results),
        "至少发生了工具调用": sum(len(r.trace.tool_calls()) for r in results) >= 5,
    }
    for label, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed = [k for k, v in checks.items() if not v]
    print(f"\n结论: {'全部通过' if not failed else '失败项: ' + ', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(run_demo())
