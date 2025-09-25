"""
llm_studio: A modular, provider-agnostic agent framework.

Key concepts:
- Normalized message and tool-call schema across providers
- Pluggable provider adapters (OpenAI, Anthropic/Claude, Google/Gemini)
- Tool system with JSON-schema validation and execution
- Agent runtime that loops over model-tool interactions

Note: Provider adapters are implemented with no external dependencies or
network calls here, providing interfaces and request/response shaping only.
"""

from .agent.runtime import Agent
from .models.registry import get_provider, register_provider
from .tools.registry import get_tool, register_tool
from .schemas.messages import (
    Message,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ToolMessage,
)
from .schemas.tooling import ToolSpec
from .logging import (
    setup_logging,
    get_logger,
    LogLevel,
    LogFormat,
    AgentLogger,
    AgentLoggerContext,
)
from .batch import (
    OpenAIBatchProcessor,
    BatchTask,
    BatchResult,
    BatchJob,
    BatchStatus,
    CompletionWindow,
)

__all__ = [
    "Agent",
    "get_provider",
    "register_provider",
    "get_tool",
    "register_tool",
    "Message",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "ToolSpec",
    "setup_logging",
    "get_logger",
    "LogLevel",
    "LogFormat",
    "AgentLogger",
    "AgentLoggerContext",
    "OpenAIBatchProcessor",
    "BatchTask",
    "BatchResult",
    "BatchJob",
    "BatchStatus",
    "CompletionWindow",
]
