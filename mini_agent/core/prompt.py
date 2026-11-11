"""系统提示词构造：把工具 Schema 目录 + 输出协议注入 prompt，让 LLM 自主决策调用。

设计要点
--------
* 工具目录由 ``ToolRegistry.schemas()`` 动态生成，不写死在代码里 → 自定义扩展自动生效。
* 输出协议是自研的严格 JSON 结构（不依赖厂商的 function calling），
  因此任何 OpenAI 兼容端点都能跑；同时 llm.py 也支持 native 模式。
* ``force_final=True`` 用于达到最大轮次后的强制收尾。
"""
from __future__ import annotations

import json
from typing import Any

__all__ = ["PROTOCOL_SPEC", "build_system_prompt", "catalog_json", "REPAIR_HINT", "FORCE_FINAL_HINT"]

PROTOCOL_SPEC = """# 输出协议（必须严格遵守）
你的每一次回复只能是 **一个** JSON 对象，不要输出 Markdown 代码块，不要输出任何额外文字。
两种合法形状，二选一：

1) 需要调用工具时：
{"thought": "简要说明你为什么需要调用这些工具（1-2 句）",
 "action": {"type": "tool",
            "tool_calls": [{"name": "工具名", "arguments": {"参数名": 值}}]}}

2) 可以直接回答时：
{"thought": "简要说明你已经掌握足够信息（1-2 句）",
 "action": {"type": "final", "content": "给用户的最终回答"}}

规则：
- 由你自己判断「直接回复」还是「调用工具」，判断依据只有上面的工具 Schema 与当前对话。
- arguments 必须符合对应工具 parameters JSON Schema 的类型与 required 约束。
- 一次可以同时调用多个互不依赖的工具，放进同一个 tool_calls 数组。
- 工具结果会以 role=tool 的消息返回给你；看到结果后，要么继续调用工具，要么给出最终答案。
- 严禁编造工具返回结果；严禁输出协议之外的字段或解释性文字。
- 最终答案使用与用户相同的语言。
"""

FORCE_FINAL_HINT = (
    "\n# 强制收尾\n已达到最大工具轮次，本轮 **禁止** 再调用任何工具，"
    "必须立刻返回 {\"thought\": ..., \"action\": {\"type\": \"final\", \"content\": ...}}，"
    "基于已获得的工具结果给出尽可能完整的回答，并说明哪些信息未能取得。\n"
)

REPAIR_HINT = (
    "你上一次的输出不是合法的协议 JSON，无法解析。请重新输出 **仅一个** JSON 对象，"
    "严格遵循上面的两种合法形状，不要 Markdown 代码块、不要任何前后缀文字。"
)


def catalog_json(tool_schemas: list[dict[str, Any]]) -> str:
    return json.dumps(tool_schemas, ensure_ascii=False, indent=2)


def build_system_prompt(
    tool_schemas: list[dict[str, Any]],
    force_final: bool = False,
    extra_instructions: str = "",
    protocol: str = "json",
) -> str:
    """构造发给 LLM 的 system 消息（每次 loop 迭代重建，不存进历史）。

    protocol="json"   -> 把工具目录与自研 JSON 输出协议写进 prompt（默认，通用）。
    protocol="native" -> 工具目录改由 Chat Completions 的 tools 参数传递。
    """
    if protocol == "native":
        parts = [
            "你是一个在工具循环（tool loop）中工作的助手 Mini-Agent。",
            "判断是否需要调用已提供的 tools；需要就调用，不需要就直接回答。",
            "看到工具结果后，要么继续调用工具，要么用与用户相同的语言给出最终答案。",
            "严禁编造工具返回结果。",
        ]
        if extra_instructions.strip():
            parts.append("# 补充约定\n" + extra_instructions.strip())
        if force_final:
            parts.append("\n# 强制收尾\n已达到最大工具轮次，禁止再调用工具，请立刻给出最终答案。")
        return "\n".join(parts)

    parts = [
        "你是一个在工具循环（tool loop）中工作的助手 Mini-Agent。",
        "你需要先判断是否已有足够信息回答，还是必须调用工具获取信息。\n",
        "# 可用工具（JSON Schema 目录）",
        "下面是全部已注册工具，字段名/类型/必填项以 parameters 为准：",
        catalog_json(tool_schemas),
        "\n" + PROTOCOL_SPEC,
    ]
    if extra_instructions.strip():
        parts.append("\n# 补充约定\n" + extra_instructions.strip())
    if force_final:
        parts.append(FORCE_FINAL_HINT)
    return "\n".join(parts)
