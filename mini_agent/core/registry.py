"""工具注册机制 + 最小 JSON Schema 校验 + 安全执行边界。

* 每个工具 = ToolSpec(name, description, parameters JSON Schema) + handler。
* ``schemas()`` 的产物直接喂给 LLM（prompt 目录或 native tools 参数），
  因此新增工具无需改动 Runtime 一行代码。
* ``execute()`` 是唯一的工具执行入口：未知工具 / 参数非法 / handler 抛异常
  全部转成 ``ToolResult(ok=False, error=...)`` 回喂给 LLM，让 agent 有机会自我纠正，
  而不是把整个 loop 打断。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from .errors import ToolValidationError
from .types import ToolCall, ToolContext, ToolResult, ToolSpec

__all__ = ["RegisteredTool", "ToolRegistry", "validate_against_schema"]

Handler = Callable[[dict[str, Any], ToolContext], Any]


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: Handler


# ---------------------------------------------------------------------------
# 最小 JSON Schema 校验器（只实现本项目需要的子集，零第三方依赖）
# ---------------------------------------------------------------------------
def _type_ok(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate_against_schema(schema: dict[str, Any], value: Any, path: str = "$") -> None:
    """校验 value 是否满足 schema；不满足时抛 ToolValidationError。"""
    if not isinstance(schema, dict):
        return

    expected = schema.get("type")
    if isinstance(expected, str) and not _type_ok(value, expected):
        raise ToolValidationError(
            f"{path} 类型错误: 期望 {expected}，实际 {type(value).__name__}"
        )

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(
            f"{path} 取值 {value!r} 不在允许集合 {schema['enum']!r} 中"
        )

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolValidationError(f"{path} 长度不足 {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolValidationError(f"{path} 长度超过 {schema['maxLength']}")
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], value):
                    raise ToolValidationError(f"{path} 不匹配 pattern {schema['pattern']!r}")
            except re.error:
                pass  # 非法正则不阻塞执行

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolValidationError(f"{path} 小于 minimum={schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolValidationError(f"{path} 大于 maximum={schema['maximum']}")

    if isinstance(value, dict):
        for key in schema.get("required", []) or []:
            if key not in value:
                raise ToolValidationError(f"{path} 缺少必填参数 '{key}'")
        props = schema.get("properties") or {}
        for key, sub in value.items():
            if key in props:
                validate_against_schema(props[key], sub, f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ToolValidationError(f"{path} 出现未声明参数 '{key}'")
            elif isinstance(schema.get("additionalProperties"), dict):
                validate_against_schema(schema["additionalProperties"], sub, f"{path}.{key}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ToolValidationError(f"{path} 元素少于 {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ToolValidationError(f"{path} 元素多于 {schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                validate_against_schema(item_schema, item, f"{path}[{i}]")


class ToolRegistry:
    """工具表。同名重复注册默认拒绝，可显式 replace=True。"""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    # ---- 注册 -------------------------------------------------------------
    def register(self, spec: ToolSpec, handler: Handler, *, replace: bool = False) -> ToolSpec:
        if not isinstance(spec, ToolSpec):
            raise TypeError("spec 必须是 ToolSpec")
        if not isinstance(spec.name, str) or not spec.name.strip():
            raise ValueError("工具名不能为空")
        if not callable(handler):
            raise TypeError("handler 必须可调用")
        if spec.name in self._tools and not replace:
            raise ValueError(f"工具 '{spec.name}' 已注册；如需覆盖请传 replace=True")
        if not isinstance(spec.parameters, dict):
            raise ValueError(f"工具 '{spec.name}' 的 parameters 必须是 dict")
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)
        return spec

    def tool(self, name: str, description: str, parameters: dict[str, Any]) -> Callable[[Handler], Handler]:
        """装饰器写法::

            @registry.tool("echo", "回显文本", {...})
            def echo(args, ctx):
                return {"text": args["text"]}
        """
        def decorator(fn: Handler) -> Handler:
            self.register(ToolSpec(name, description, parameters), fn)
            return fn
        return decorator

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # ---- 查询 -------------------------------------------------------------
    def get(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry.spec if entry else None

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[n].spec.schema_dict() for n in self._tools]

    def native_schemas(self) -> list[dict[str, Any]]:
        return [self._tools[n].spec.native_schema() for n in self._tools]

    # ---- 执行 -------------------------------------------------------------
    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        entry = self._tools.get(call.name)
        if entry is None:
            return self._finish(
                ToolResult(ok=False, error=(
                    f"unknown_tool: '{call.name}' 未注册。可用工具: {', '.join(self.names()) or '(无)'}"
                )), started,
            )

        try:
            validate_against_schema(entry.spec.parameters, call.arguments)
        except ToolValidationError as exc:
            return self._finish(ToolResult(ok=False, error=f"参数校验失败: {exc}"), started)

        try:
            output = entry.handler(dict(call.arguments), ctx)
        except Exception as exc:  # noqa: BLE001 - 工具异常必须被隔离，不能打断 loop
            return self._finish(
                ToolResult(ok=False, error=f"工具执行异常 {type(exc).__name__}: {exc}"), started,
            )

        if isinstance(output, ToolResult):
            return self._finish(output, started)
        if isinstance(output, dict):
            # 允许 handler 直接返回 {"ok": False, "error": ...}
            if output.get("ok") is False:
                return self._finish(
                    ToolResult(ok=False, data=output.get("result"),
                               error=str(output.get("error") or "tool failed")), started,
                )
            return self._finish(ToolResult(ok=True, data=output), started)
        return self._finish(ToolResult(ok=True, data=output), started)

    @staticmethod
    def _finish(result: ToolResult, started: float) -> ToolResult:
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        return result
