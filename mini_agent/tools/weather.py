"""weather：扩展工具示例（mock）。不调用任何外部天气 API，零密钥。"""
from __future__ import annotations

from typing import Any

from ..core.types import ToolContext, ToolResult, ToolSpec

__all__ = ["WEATHER_SPEC", "weather_handler"]

_PRESETS: dict[str, dict[str, Any]] = {
    "北京": {"condition": "晴", "temperature": 18, "wind": "西北风 3 级", "humidity": 42},
    "上海": {"condition": "多云", "temperature": 22, "wind": "东风 2 级", "humidity": 66},
    "广州": {"condition": "多云转阵雨", "temperature": 26, "wind": "东南风 2 级", "humidity": 78},
    "深圳": {"condition": "阵雨", "temperature": 27, "wind": "南风 3 级", "humidity": 80},
    "杭州": {"condition": "阴", "temperature": 20, "wind": "北风 2 级", "humidity": 70},
    "成都": {"condition": "小雨", "temperature": 17, "wind": "静风", "humidity": 85},
    "武汉": {"condition": "晴", "temperature": 21, "wind": "东北风 2 级", "humidity": 55},
}

WEATHER_SPEC = ToolSpec(
    name="weather",
    description="查询指定城市的当日天气（mock 数据，用于演示工具扩展）。涉及天气/温度/是否适合施药时使用。",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，例如 '广州'", "minLength": 1},
            "date": {"type": "string", "description": "日期，留空表示今天"},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)


def weather_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    city = str(args["city"]).strip()
    if not city:
        return ToolResult(ok=False, error="city 不能为空")
    base = _PRESETS.get(city) or {
        "condition": "晴间多云", "temperature": 23, "wind": "微风", "humidity": 60,
    }
    temp = int(base["temperature"])
    advice = ("湿度偏高，建议延后叶面施药并加强通风" if int(base["humidity"]) > 75
              else "天气条件适合田间作业")
    return ToolResult(ok=True, data={
        "city": city, "date": args.get("date") or "today",
        "condition": base["condition"], "temperature": temp,
        "wind": base["wind"], "humidity": base["humidity"],
        "advice": advice, "mock": True,
    })
