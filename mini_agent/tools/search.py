"""search：可 mock 的检索工具（零外部依赖、零密钥、结果确定可复现）。

用本地静态语料 + 字符/标签重合度打分，模拟一次向量检索。
真实项目中只需替换 handler 内部实现，Schema 和 Runtime 都不用改。
"""
from __future__ import annotations

import re
from typing import Any

from ..core.types import ToolContext, ToolResult, ToolSpec

__all__ = ["SEARCH_SPEC", "search_handler", "CORPUS"]

CORPUS: list[dict[str, Any]] = [
    {"title": "番茄早疫病识别与防治", "tags": ["番茄", "早疫病", "病害", "防治", "大棚"],
     "snippet": "叶片出现同心轮纹状褐斑，湿度大时病部生黑色霉层；建议通风降湿并轮换用药。"},
    {"title": "水稻稻飞虱绿色防控要点", "tags": ["水稻", "稻飞虱", "虫害", "农药"],
     "snippet": "田间保持浅水层，优先选用生物农药，注意安全间隔期。"},
    {"title": "大棚温湿度管理手册", "tags": ["大棚", "天气", "温度", "湿度", "种植"],
     "snippet": "晴天上午适时通风，夜间保温；连续阴雨需重点预防灰霉病。"},
    {"title": "农药成本估算参考表", "tags": ["成本", "农药", "估算", "预算"],
     "snippet": "按亩用量 × 单价 × 施药次数估算，另计入人工与器械折旧。"},
    {"title": "周报写作规范", "tags": ["周报", "文档", "写作", "模板"],
     "snippet": "本周进展 / 风险与阻塞 / 下周计划 三段式，每条一句话、带可核对的结论。"},
    {"title": "AI 智能体工具调用协议说明", "tags": ["agent", "工具", "协议", "schema"],
     "snippet": "模型依据工具 JSON Schema 自主决策，输出结构化工具调用，runtime 负责执行与回灌结果。"},
]

SEARCH_SPEC = ToolSpec(
    name="search",
    description=(
        "在本地知识语料中检索与 query 相关的资料，返回标题与摘要（mock 实现，结果可复现）。"
        "需要背景知识、资料引用或不确定事实时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词或自然语言问题", "minLength": 1},
            "top_k": {"type": "integer", "description": "返回条数", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


def _score(query: str, doc: dict[str, Any]) -> int:
    tokens = [t for t in re.split(r"[\s,，、。;；:：/]+", query) if t]
    score = 0
    for tag in doc["tags"]:
        if tag in query:
            score += 3
        if any(t and t in tag for t in tokens):
            score += 1
    for token in tokens:
        if len(token) >= 2 and token in doc["title"] + doc["snippet"]:
            score += 2
    # 字符级兜底，保证任意 query 也有稳定排序
    score += sum(1 for ch in set(query) if ch in doc["title"])
    return score


def search_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(args["query"]).strip()
    top_k = int(args.get("top_k", 3))
    if not query:
        return ToolResult(ok=False, error="query 不能为空")
    ranked = sorted(CORPUS, key=lambda d: (-_score(query, d), d["title"]))
    picked = ranked[:top_k]
    results = [
        {"title": d["title"], "snippet": d["snippet"], "score": _score(query, d), "source": "mock"}
        for d in picked
    ]
    return ToolResult(ok=True, data={
        "query": query, "results": results, "total": len(CORPUS),
        "returned": len(results), "mock": True,
    })
