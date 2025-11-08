"""Provider implementations."""

from .mock import MockProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .google import GoogleProvider
from .huggingface import HuggingFaceProvider

# Auto-register providers
from ..provider import register

register("mock", MockProvider)
register("anthropic", AnthropicProvider)
register("openai", OpenAIProvider)
register("google", GoogleProvider)
register("huggingface", HuggingFaceProvider)

__all__ = [
    "MockProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "HuggingFaceProvider",
]
