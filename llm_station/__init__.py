"""llm_station: A modular, provider-agnostic agent framework."""

from .agent import Agent
from .provider import (
    Provider,
    get as get_provider,
    register as register_provider,
    list_providers,
)
from .tools import (
    Tool,
    get as get_tool,
    get_spec,
    register as register_tool,
    register_provider_tool,
    list_tools,
)
from .types import (
    Message,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
    ToolSpec,
    ToolResult,
    ModelResponse,
    ModelConfig,
)

# Import providers to auto-register them
from . import providers  # noqa: F401

# Import tools to auto-register them
from . import tools  # noqa: F401

__all__ = [
    # Main API
    "Agent",
    # Provider API
    "Provider",
    "get_provider",
    "register_provider",
    "list_providers",
    # Tool API
    "Tool",
    "get_tool",
    "get_spec",
    "register_tool",
    "register_provider_tool",
    "list_tools",
    # Types
    "Message",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "ToolCall",
    "ToolSpec",
    "ToolResult",
    "ModelResponse",
    "ModelConfig",
]
