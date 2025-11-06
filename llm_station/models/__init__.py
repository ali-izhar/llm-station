from .registry import register_provider

# Import adapters and register them
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider
from .mock import MockProvider
from .huggingface import HuggingFaceProvider

# Register on import for convenience
register_provider(OpenAIProvider.name, OpenAIProvider)
register_provider(AnthropicProvider.name, AnthropicProvider)
register_provider(GoogleProvider.name, GoogleProvider)
register_provider(MockProvider.name, MockProvider)
register_provider(HuggingFaceProvider.name, HuggingFaceProvider)

__all__ = [
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MockProvider",
    "HuggingFaceProvider",
]
