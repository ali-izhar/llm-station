#!/usr/bin/env python3
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

from ..schemas.messages import Message, ModelResponse
from ..schemas.tooling import ToolSpec


@dataclass
class ModelConfig:
    """Provider-agnostic model configuration."""

    provider: str
    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Common parameters across providers
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    # Structured output support (JSON Schema). Providers may ignore if unsupported.
    response_json_schema: Optional[Dict[str, Any]] = None
    # Streaming support
    stream: bool = False
    # Provider-specific parameters (each provider handles its own)
    provider_kwargs: Optional[Dict[str, Any]] = None


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
