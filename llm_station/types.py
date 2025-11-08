"""Core types and data structures for llm_station."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

# Message types
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """Base message type."""

    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


@dataclass
class SystemMessage(Message):
    """System message."""

    def __init__(self, content: str):
        super().__init__(role="system", content=content)


@dataclass
class UserMessage(Message):
    """User message."""

    def __init__(self, content: str, name: Optional[str] = None):
        super().__init__(role="user", content=content, name=name)


@dataclass
class AssistantMessage(Message):
    """Assistant message with optional tool calls."""

    tool_calls: Optional[List[ToolCall]] = None
    grounding_metadata: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        content: str,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        grounding_metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(role="assistant", content=content, name=name)
        self.tool_calls = tool_calls
        self.grounding_metadata = grounding_metadata


@dataclass
class ToolMessage(Message):
    """Tool result message."""

    def __init__(self, content: str, tool_call_id: str, name: Optional[str] = None):
        super().__init__(
            role="tool", content=content, name=name, tool_call_id=tool_call_id
        )


@dataclass
class ToolCall:
    """Tool call requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelResponse:
    """Normalized model response."""

    content: str
    tool_calls: List[ToolCall]
    raw: Optional[Dict[str, Any]] = None
    grounding_metadata: Optional[Dict[str, Any]] = None


# Tool types
@dataclass
class ToolSpec:
    """Tool specification for providers."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    requires_network: bool = False
    requires_filesystem: bool = False
    provider: Optional[str] = None
    provider_type: Optional[str] = None
    provider_config: Optional[Dict[str, Any]] = None


@dataclass
class ToolResult:
    """Result from tool execution."""

    name: str
    content: str
    tool_call_id: str
    is_error: bool = False
    meta: Optional[Dict[str, Any]] = None


# Configuration types
@dataclass
class ModelConfig:
    """Provider-agnostic model configuration."""

    provider: str
    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    response_json_schema: Optional[Dict[str, Any]] = None
    stream: bool = False
    provider_kwargs: Optional[Dict[str, Any]] = None
