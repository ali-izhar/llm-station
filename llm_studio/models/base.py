from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..schemas.messages import Message, ModelResponse
from ..schemas.tooling import ToolSpec


@dataclass
class ModelConfig:
    provider: str
    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Structured output support (JSON Schema). Providers may ignore if unsupported.
    response_json_schema: Optional[Dict[str, Any]] = None
    # Whether to stream (not implemented in this skeleton)
    stream: bool = False
    # Optional: prefer a specific API flavor if applicable (e.g., "responses" vs "chat")
    api: Optional[str] = None
    # OpenAI Responses API-specific optional fields
    tool_choice: Optional[str] = None
    include: Optional[List[str]] = None  # e.g., ["web_search_call.action.sources"]
    reasoning: Optional[Dict[str, Any]] = (
        None  # e.g., {"effort": "low"|"medium"|"high"}
    )
    # Additional OpenAI-specific parameters for advanced usage
    instructions: Optional[str] = None  # Custom instructions for Responses API


class ProviderAdapter(abc.ABC):
    """Abstract provider adapter to normalize requests and responses."""

    name: str  # unique identifier, e.g., "openai", "anthropic", "google"

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.extra = kwargs

    @abc.abstractmethod
    def supports_tools(self) -> bool:
        return True

    @abc.abstractmethod
    def prepare_tools(self, tools: Iterable[ToolSpec]) -> Any:
        """Convert normalized ToolSpec list into provider-specific schema."""

    @abc.abstractmethod
    def generate(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> ModelResponse:
        """Perform a single model generate call and return normalized response.

        Implementations should:
          - Map messages to provider format
          - Include tools if provided
          - Apply structured output hints if possible
          - Parse tool-call responses to normalized ToolCall list
        """
        raise NotImplementedError
