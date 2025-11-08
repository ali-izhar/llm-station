"""Provider base class and registry."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Type

from .types import Message, ModelConfig, ModelResponse, ToolSpec


class Provider(abc.ABC):
    """Base provider class for LLM APIs."""

    name: str  # Set by subclasses

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.kwargs = kwargs

    @abc.abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider supports tools."""
        return True

    @abc.abstractmethod
    def prepare_tools(self, tools: List[ToolSpec]) -> Any:
        """Convert ToolSpec list to provider-specific format."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> ModelResponse:
        """Generate response from messages."""
        raise NotImplementedError


# Simple registry
_registry: Dict[str, Type[Provider]] = {}


def register(name: str, provider_class: Type[Provider]) -> None:
    """Register a provider."""
    name = name.lower().strip()
    provider_class.name = name
    _registry[name] = provider_class


def get(name: str, **kwargs: Any) -> Provider:
    """Get a provider instance."""
    name = name.lower().strip()
    if name not in _registry:
        raise KeyError(f"Unknown provider: {name}. Available: {list(_registry.keys())}")
    return _registry[name](**kwargs)


def list_providers() -> Dict[str, Type[Provider]]:
    """List all registered providers."""
    return dict(_registry)
