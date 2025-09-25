#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """Normalized representation of a tool/function call requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


@dataclass
class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__(role="system", content=content)


@dataclass
class UserMessage(Message):
    def __init__(self, content: str, name: Optional[str] = None):
        super().__init__(role="user", content=content, name=name)


@dataclass
class AssistantMessage(Message):
    tool_calls: Optional[List[ToolCall]] = None
    grounding_metadata: Optional[Dict[str, Any]] = (
        None  # Preserve metadata from ModelResponse
    )

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
    def __init__(self, content: str, tool_call_id: str, name: Optional[str] = None):
        super().__init__(
            role="tool", content=content, name=name, tool_call_id=tool_call_id
        )


@dataclass
class ModelResponse:
    """Normalized model response returned by provider adapters."""

    content: str
    tool_calls: List[ToolCall]
    # Raw provider-specific payload for debugging/analysis if needed
    raw: Optional[Dict[str, Any]] = None
    # Grounding metadata for search results and citations
    grounding_metadata: Optional[Dict[str, Any]] = None
