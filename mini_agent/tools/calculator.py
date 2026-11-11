"""calculator：基于 AST 白名单的安全数学求值，**不使用 eval / exec**。

只允许：数字字面量、+ - * / // % ** 、一元正负、白名单函数与常量、list/tuple 字面量。
任何 Name 如果不是白名单常量/函数、任何其它 AST 节点都会直接抛错。
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from ..core.types import ToolContext, ToolResult, ToolSpec

__all__ = ["CALCULATOR_SPEC", "calculator_handler", "safe_eval"]

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil, "exp": math.exp,
}
_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}

CALCULATOR_SPEC = ToolSpec(
    name="calculator",
    description=(
        "安全计算一个数学表达式并返回数值。支持 + - * / // % ** 、括号、"
        "sqrt/abs/round/min/max/pow/log/sin/cos/floor/ceil 等函数与 pi/e 常量。"
        "需要算数、单位换算、成本估算时使用；不支持变量赋值或任意代码。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的表达式，例如 '12*(3+4)' 或 'sqrt(144)/2'",
                "minLength": 1,
                "maxLength": 200,
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"只允许数字字面量，收到 {value!r}")
        return value  # type: ignore[return-value]

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符 {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if op is operator.pow and abs(right) > 100:
            raise ValueError("幂运算指数过大")
        try:
            result = op(left, right)
        except ZeroDivisionError as exc:
            raise ValueError("除数为 0") from exc
        return _check_number(result)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符 {type(node.op).__name__}")
        return _check_number(op(_eval_node(node.operand)))

    if isinstance(node, ast.Name):
        if node.id in _NAMES:
            return _NAMES[node.id]
        raise ValueError(f"不允许的名称 '{node.id}'")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("只允许调用白名单函数")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        args = [_eval_node(a) for a in node.args]
        try:
            return _check_number(_FUNCS[node.func.id](*args))
        except (ValueError, OverflowError, TypeError) as exc:
            raise ValueError(f"函数调用失败: {exc}") from exc

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e) for e in node.elts]  # type: ignore[return-value]

    raise ValueError(f"不允许的语法节点 {type(node).__name__}")


def _check_number(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return value  # type: ignore[return-value]
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"结果不是数字: {value!r}") from exc
    if math.isnan(number) or math.isinf(number):
        raise ValueError("结果不是有限数")
    if abs(number) > 1e15:
        raise ValueError("结果绝对值过大")
    return number


def safe_eval(expression: str) -> float:
    """求值并返回有限数字；任何非法输入都会抛 ValueError。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式不能为空")
    expr = expression.strip()
    if len(expr) > 200:
        raise ValueError("表达式过长（上限 200 字符）")
    expr = (expr.replace("×", "*").replace("÷", "/").replace("^", "**")
            .replace("乘以", "*").replace("除以", "/")
            .replace("加上", "+").replace("减去", "-"))
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误: {exc.msg}") from exc
    return _eval_node(tree.body)


def calculator_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    expression = str(args["expression"])
    try:
        value = safe_eval(expression)
    except ValueError as exc:
        return ToolResult(ok=False, error=f"计算失败: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"计算失败: {type(exc).__name__}: {exc}")
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return ToolResult(ok=True, data={"expression": expression, "result": value})
