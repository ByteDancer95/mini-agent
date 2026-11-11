"""工具装配工厂。

* build_minimal_registry() -> 题目要求的 3 个必需工具：calculator / search / todo
* build_default_registry()  -> 在上面基础上挂载扩展工具：weather / read_docs

自定义扩展推荐写法（Runtime 无需任何改动）::

    from mini_agent.core.types import ToolSpec
    registry.tool("echo", "回显文本", {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    })(lambda args, ctx: {"text": args["text"]})
"""
from __future__ import annotations

from typing import Iterable

from ..core.registry import ToolRegistry
from .calculator import CALCULATOR_SPEC, calculator_handler
from .read_docs import READ_DOCS_SPEC, read_docs_handler
from .search import SEARCH_SPEC, search_handler
from .todo import TODO_SPEC, todo_handler
from .weather import WEATHER_SPEC, weather_handler

__all__ = ["build_minimal_registry", "build_default_registry", "EXTENSIONS"]

EXTENSIONS = {
    "weather": (WEATHER_SPEC, weather_handler),
    "read_docs": (READ_DOCS_SPEC, read_docs_handler),
}


def build_minimal_registry() -> ToolRegistry:
    """只注册 3 个必需工具。"""
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator_handler)
    registry.register(SEARCH_SPEC, search_handler)
    registry.register(TODO_SPEC, todo_handler)
    return registry


def build_default_registry(extensions: Iterable[str] | None = None) -> ToolRegistry:
    """3 个必需工具 + 可选扩展。

    extensions=None 表示挂载全部扩展；传 [] 等价于 build_minimal_registry()。
    """
    registry = build_minimal_registry()
    wanted = list(EXTENSIONS.keys()) if extensions is None else list(extensions)
    for name in wanted:
        if name not in EXTENSIONS:
            raise ValueError(f"未知扩展工具 '{name}'，可选: {list(EXTENSIONS)}")
        spec, handler = EXTENSIONS[name]
        registry.register(spec, handler)
    return registry
