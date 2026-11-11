"""异常类型。Runtime 边界会捕获所有异常并转成 RunResult(status='error')，不向外抛。"""
from __future__ import annotations

__all__ = [
    "MiniAgentError", "LLMError", "LLMConfigError", "ToolNotFoundError",
    "ToolValidationError", "ParseError", "LoopLimitExceeded",
]


class MiniAgentError(Exception):
    """base"""


class LLMError(MiniAgentError):
    """LLM 调用失败（网络 / HTTP / 响应格式）。"""


class LLMConfigError(LLMError):
    """缺少 LLM 配置（例如未设置环境变量）。"""


class ToolNotFoundError(MiniAgentError):
    """工具未注册。"""


class ToolValidationError(MiniAgentError):
    """工具参数不满足 Schema。"""


class ParseError(MiniAgentError):
    """LLM 输出无法解析成有效决策。"""


class LoopLimitExceeded(MiniAgentError):
    """达到最大轮次。"""
