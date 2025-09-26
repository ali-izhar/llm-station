"""
llm_station: A modular, provider-agnostic agent framework.

Key concepts:
- Normalized message and tool-call schema across providers
- Pluggable provider adapters (OpenAI, Anthropic/Claude, Google/Gemini)
- Smart tools system with automatic provider routing
- Agent runtime with clean tool integration

Note: Provider adapters are implemented with no external dependencies or
network calls here, providing interfaces and request/response shaping only.
"""

from .agent.runtime import Agent
from .models.registry import get_provider, register_provider
from .tools.registry import (
    get_tool_recommendations,
    get_available_tools,
    get_tool_info,
    recommend_tools,
)
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
    GoogleBatchProcessor,
    GoogleBatchTask,
    GoogleBatchResult,
    GoogleBatchJob,
    GoogleBatchStatus,
    AnthropicBatchProcessor,
    AnthropicBatchRequest,
    AnthropicBatchResult,
    AnthropicBatchJob,
    AnthropicBatchStatus,
)

__all__ = [
    "Agent",
    "get_provider",
    "register_provider",
    "get_tool_recommendations",
    "get_available_tools",
    "get_tool_info",
    "recommend_tools",
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
    "GoogleBatchProcessor",
    "GoogleBatchTask",
    "GoogleBatchResult",
    "GoogleBatchJob",
    "GoogleBatchStatus",
    "AnthropicBatchProcessor",
    "AnthropicBatchRequest",
    "AnthropicBatchResult",
    "AnthropicBatchJob",
    "AnthropicBatchStatus",
]
