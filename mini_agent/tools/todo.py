"""todo：会话级待办（必需工具）。

关键设计：待办数据只存放在 ``ctx.state['todos']`` —— 这个 dict 由 Session 私有持有，
不同 session 传入不同的 ctx.state，因此天然隔离，不存在任何模块级全局列表。
"""
from __future__ import annotations

import copy
from typing import Any

from ..core.types import ToolContext, ToolResult, ToolSpec

__all__ = ["TODO_SPEC", "todo_handler"]

TODO_SPEC = ToolSpec(
    name="todo",
    description=(
        "管理当前会话的待办清单。action 支持 add（新增）、list（列出）、"
        "complete（标记完成）、remove（删除）、clear（清空）。"
        "待办按会话隔离，不同窗口互不影响。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "complete", "remove", "clear"],
                "description": "要执行的操作",
            },
            "item": {"type": "string", "description": "add 时的待办内容", "minLength": 1},
            "priority": {
                "type": "string", "enum": ["low", "normal", "high"],
                "description": "add 时的优先级，默认 normal",
            },
            "id": {"type": "integer", "description": "complete/remove 时指定待办编号", "minimum": 1},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
)


def _snapshot(state: dict[str, Any]) -> list[dict[str, Any]]:
    return copy.deepcopy(state.get("todos", []))


def _find(todos: list[dict[str, Any]], args: dict[str, Any]) -> dict[str, Any] | None:
    if args.get("id") is not None:
        return next((t for t in todos if t["id"] == args["id"]), None)
    item = str(args.get("item") or "").strip()
    if item:
        return next((t for t in todos if item in t["item"]), None)
    return None


def todo_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    action = args["action"]
    state = ctx.state
    todos = state.setdefault("todos", [])
    seq = int(state.get("_todo_seq", 0))
    message = ""

    if action == "add":
        item = str(args.get("item") or "").strip()
        if not item:
            return ToolResult(ok=False, error="add 需要提供非空 item")
        seq += 1
        state["_todo_seq"] = seq
        todos.append({
            "id": seq, "item": item,
            "priority": args.get("priority", "normal"), "done": False,
        })
        message = f"已新增待办 #{seq}: {item}"
    elif action == "list":
        message = f"当前共 {len(todos)} 项待办"
    elif action == "complete":
        target = _find(todos, args)
        if target is None:
            return ToolResult(ok=False, error="找不到要完成的待办（请提供正确的 id 或 item）",
                              data={"todos": _snapshot(state)})
        target["done"] = True
        message = f"已完成待办 #{target['id']}: {target['item']}"
    elif action == "remove":
        target = _find(todos, args)
        if target is None:
            return ToolResult(ok=False, error="找不到要删除的待办（请提供正确的 id 或 item）",
                              data={"todos": _snapshot(state)})
        todos.remove(target)
        message = f"已删除待办 #{target['id']}"
    elif action == "clear":
        todos.clear()
        message = "待办已清空"
    else:  # Schema 已挡住，这里兜底
        return ToolResult(ok=False, error=f"不支持的 action: {action}")

    return ToolResult(ok=True, data={
        "action": action, "message": message,
        "session_id": ctx.session_id, "todos": _snapshot(state),
    })
