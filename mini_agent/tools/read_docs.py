"""read_docs：扩展工具示例。读取仓库 docs/ 下的本地文档，带路径穿越防护。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.types import ToolContext, ToolResult, ToolSpec

__all__ = ["READ_DOCS_SPEC", "read_docs_handler", "default_docs_root"]

_ALLOWED_SUFFIXES = {".md", ".txt", ".markdown", ".rst"}


def default_docs_root() -> Path:
    """仓库根目录下的 docs/（mini_agent/tools/read_docs.py -> 上溯 3 层）。"""
    return Path(__file__).resolve().parents[2] / "docs"


READ_DOCS_SPEC = ToolSpec(
    name="read_docs",
    description="读取本地 docs/ 目录中的文档内容（如周报模板、写作规范）。需要模板或内部规范时使用。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 docs/ 的文件名，如 'weekly_report_template.md'",
                     "minLength": 1},
            "max_chars": {"type": "integer", "description": "最多读取字符数",
                          "minimum": 100, "maximum": 20000},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)


def read_docs_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    rel = str(args["path"]).strip()
    max_chars = int(args.get("max_chars", 4000))
    root = Path(ctx.docs_root).resolve() if ctx.docs_root else default_docs_root().resolve()

    if not root.is_dir():
        return ToolResult(ok=False, error=f"文档根目录不存在: {root}")
    if Path(rel).is_absolute() or rel.startswith("\\\\"):
        return ToolResult(ok=False, error="只允许相对路径")
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        return ToolResult(ok=False, error="检测到路径穿越，已拒绝访问")
    if target.suffix.lower() not in _ALLOWED_SUFFIXES:
        return ToolResult(ok=False, error=f"只允许读取 {sorted(_ALLOWED_SUFFIXES)} 文件")
    if not target.is_file():
        available = sorted(p.name for p in root.iterdir() if p.is_file()) if root.is_dir() else []
        return ToolResult(ok=False, error=f"文档不存在: {rel}；可用文件: {available}")

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult(ok=False, error=f"读取失败: {exc}")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n…（已截断）"
    return ToolResult(ok=True, data={
        "path": rel, "content": text, "chars": len(text), "truncated": truncated,
    })
