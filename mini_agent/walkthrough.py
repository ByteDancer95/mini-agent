"""一轮完整协议交互实录：把「LLM 基于 Schema 自主决策」逐帧摊开看。

    python -m mini_agent.walkthrough                        # 离线协议验证后端
    python -m mini_agent.walkthrough --llm openai           # 配好 .env 后，逐帧展示真实模型决策
    python -m mini_agent.walkthrough --save docs/RUN_TRANSCRIPT.md

它复用与 AgentRuntime **完全相同**的组件（build_system_prompt / ContextManager /
OutputParser / ToolRegistry / LLMClient），只是把中间结果打印出来，
用来回答「这到底是不是一个真的在按 Schema 决策的 Agent」。

注意：默认后端是离线替身，输出用于展示协议数据形态，**不声称是真实模型生成**；
配好环境变量后用 --llm openai 跑出来的才是真实 LLM 的 thought 与 tool_calls。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import PERSONA
from .core.context import ContextManager
from .core.errors import LLMConfigError
from .core.llm import build_llm, load_env_file
from .core.parser import OutputParser
from .core.prompt import build_system_prompt
from .core.runtime import AgentRuntime, RuntimeConfig
from .core.session import SessionManager
from .core.trace import TraceSink
from .core.types import Role, ToolContext
from .tools.base import build_default_registry

DEFAULT_QUESTION = "查一下广州今天的天气，顺便记个待办：明天给大棚番茄喷药"


def _block(title: str, body: str) -> str:
    return f"### {title}\n\n```text\n{body.rstrip()}\n```\n"


def build_transcript(question: str = DEFAULT_QUESTION, llm_kind: str = "heuristic") -> str:
    registry = build_default_registry()
    schemas = registry.schemas()
    parser = OutputParser()
    llm = build_llm(llm_kind)
    backend = type(llm).__name__
    is_real = backend == "OpenAICompatLLM"

    system = build_system_prompt(schemas, extra_instructions=PERSONA,
                                 protocol=getattr(llm, "protocol_name", "json"))
    ctx = ContextManager()
    ctx.add_user(question)

    out: list[str] = [
        "# 一轮 Agent 决策的完整实录",
        "",
        f"- 问题：`{question}`",
        f"- 决策后端：`{backend}` —— "
        + ("**真实 LLM**，基于工具 Schema 自主决策" if is_real else
           "离线协议验证替身（只用于验证 loop / 解析 / 工具 / 隔离的数据形态，**不是真实模型输出**）"),
        "- 组件：与 `AgentRuntime.run()` 完全同构（同一套 prompt / context / parser / registry）",
        "",
        "---",
        "",
        "## 帧 1｜交给模型的工具 Schema 目录",
        "",
        "模型只能看到这张表，据此自己判断「直接回复」还是「调用工具」。",
        "",
        _block("tool_schemas", json.dumps(schemas, ensure_ascii=False, indent=2)),
        "## 帧 2｜发给模型的 messages（system prompt 摘要）",
        "",
        _block("system prompt（前 900 字）", system[:900] + f"\n…（共 {len(system)} 字）"),
        _block("messages 角色序列",
               "\n".join(f"{i}. [{m.role}] {m.content[:80].replace(chr(10), ' ')}"
                         for i, m in enumerate(ctx.build_messages(system, None)))),
    ]

    # ---- 第一次 LLM 调用 --------------------------------------------------
    resp1 = llm.generate(ctx.build_messages(system, None), schemas, False)
    o1 = parser.parse(resp1)
    out += [
        "## 帧 3｜模型第 1 次原始输出（未做任何加工）",
        "",
        _block("LLM 原文", resp1.raw),
        "## 帧 4｜解析结果：思考过程 / 工具调用 / 决策",
        "",
        _block("parse",
               f"source      = {o1.source}\n"
               f"thought     = {o1.thought}\n"
               f"decision    = {o1.decision()}\n"
               f"tool_calls  = {json.dumps([c.to_dict() for c in o1.tool_calls], ensure_ascii=False)}\n"
               f"warning     = {o1.parse_warning}"),
    ]

    # ---- 执行工具 ---------------------------------------------------------
    ctx.add_decision(o1)  # 注意：thought 被剥离，只把 action 写回 context
    tool_ctx = ToolContext("walkthrough", {})
    lines = []
    for call in o1.tool_calls:
        result = registry.execute(call, tool_ctx)
        ctx.add_tool_result(call, result)
        lines.append(f"{call.name}({json.dumps(call.arguments, ensure_ascii=False)})"
                     f"\n  -> ok={result.ok} elapsed={result.elapsed_ms}ms"
                     f"\n  -> {json.dumps(result.to_payload(), ensure_ascii=False, default=str)}")
    out += [
        "## 帧 5｜Runtime 执行工具（Schema 校验 + 异常隔离）",
        "",
        _block("tool results", "\n\n".join(lines) or "(本次没有工具调用)"),
        "## 帧 6｜工具结果回灌后的 context",
        "",
        "思考过程 **不** 进入 LLM context；工具结果以 `role=tool` 紧跟决策之后。",
        "",
        _block("messages 角色序列",
               "\n".join(f"{i}. [{m.role}] {m.content[:140].replace(chr(10), ' ')}"
                         for i, m in enumerate(ctx.build_messages(system, None)))),
    ]

    # ---- 第二次 LLM 调用 --------------------------------------------------
    resp2 = llm.generate(ctx.build_messages(system, None), schemas, False)
    o2 = parser.parse(resp2)
    out += [
        "## 帧 7｜模型看到工具结果后的第 2 次输出",
        "",
        _block("LLM 原文", resp2.raw),
        "## 帧 8｜最终答案",
        "",
        f"> {o2.final_answer or '(未给出 final)'}",
        "",
    ]

    # ---- 与真实 Runtime 对照 ----------------------------------------------
    rt = AgentRuntime(
        llm=build_llm(llm_kind),
        registry=build_default_registry(),
        sessions=SessionManager(),
        config=RuntimeConfig(extra_instructions=PERSONA),
        trace_sink=TraceSink(to_logger=False),
    )
    run = rt.run("walkthrough-session", question)
    out += [
        "---",
        "",
        "## 附录｜同一个问题交给真正的 `AgentRuntime.run()` 后的 trace",
        "",
        _block("trace", run.trace.to_text()),
        f"status = `{run.status}`  turns = `{run.turns}`  "
        f"工具调用数 = `{len(run.trace.tool_calls())}`",
        "",
        "> 复现：`python -m mini_agent.walkthrough"
        + (" --llm openai" if llm_kind == "openai" else "")
        + " --save <path>`",
        "",
    ]
    if is_real:
        out.append("> 本文件含真实模型输出，提交前请确认其中没有敏感信息。")
    else:
        out.append("> 本文件由离线替身生成，仅用于展示协议形态；配置 `.env` 后加 `--llm openai` "
                   "即可生成真实模型版本。")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="逐帧展示一轮 Agent 决策")
    p.add_argument("--llm", default="heuristic", choices=["heuristic", "openai"])
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--save", default=None, help="把实录写成 Markdown 文件")
    args = p.parse_args(argv)

    load_env_file(Path(__file__).resolve().parents[1] / ".env")
    try:
        text = build_transcript(args.question, args.llm)
    except LLMConfigError as exc:
        print(f"真实 LLM 未配置：{exc}\n请先填 .env，或用默认 --llm heuristic 查看协议实录。")
        return 1
    print(text)
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"\n已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
