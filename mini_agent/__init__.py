"""mini-agent: 一个不依赖任何 agent 框架的最小可用 Agent Runtime。

核心设计: LLM 只是一个 ``generate(messages, tool_schemas) -> LLMResponse`` 的可替换组件，
loop / 解析 / 工具执行 / 会话隔离 / context 管理全部由本包自己实现。
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["AgentRuntime", "RuntimeConfig"]

# 延迟导入，避免 import mini_agent 时强制依赖子模块
def __getattr__(name: str):  # pragma: no cover
    if name in __all__:
        from .core.runtime import AgentRuntime, RuntimeConfig
        return {"AgentRuntime": AgentRuntime, "RuntimeConfig": RuntimeConfig}[name]
    raise AttributeError(name)
