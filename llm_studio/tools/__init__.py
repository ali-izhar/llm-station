#!/usr/bin/env python3

from .registry import register_tool, register_provider_tool
from .fetch_url import FetchUrlTool
from .json_format import JsonFormatTool

# Import provider tool factories
from .web_search.openai import OpenAIWebSearch
from .web_search.anthropic import AnthropicWebSearch
from .web_search.google import GoogleWebSearch, GoogleSearchRetrieval
from .web_fetch.anthropic import AnthropicWebFetch
from .code_execution.anthropic import AnthropicCodeExecution
from .code_execution.openai import OpenAICodeInterpreter
from .code_execution.google import GoogleCodeExecution
from .image_generation.openai import OpenAIImageGeneration
from .image_generation.google import GoogleImageGeneration
from .url_context.google import GoogleUrlContext

# Register built-in local tools on import
register_tool("fetch_url", FetchUrlTool)
register_tool("json_format", JsonFormatTool)

# Register provider-native tools with default configurations

# OpenAI tools
register_provider_tool("openai_web_search", lambda: OpenAIWebSearch().spec())
register_provider_tool(
    "openai_web_search_preview", lambda: OpenAIWebSearch(preview=True).spec()
)
register_provider_tool(
    "openai_code_interpreter", lambda: OpenAICodeInterpreter().spec()
)
register_provider_tool(
    "openai_image_generation", lambda: OpenAIImageGeneration().spec()
)

# Anthropic tools
register_provider_tool("anthropic_web_search", lambda: AnthropicWebSearch().spec())
# register_provider_tool("anthropic_web_fetch", lambda: AnthropicWebFetch().spec())  # Not supported in current API
# register_provider_tool("anthropic_code_execution", lambda: AnthropicCodeExecution().spec())  # Requires beta access

# Google tools
register_provider_tool("google_search", lambda: GoogleWebSearch().spec())
register_provider_tool(
    "google_search_retrieval",
    lambda: GoogleSearchRetrieval(mode="MODE_DYNAMIC", dynamic_threshold=0.7).spec(),
)
register_provider_tool("google_code_execution", lambda: GoogleCodeExecution().spec())
register_provider_tool("google_url_context", lambda: GoogleUrlContext().spec())
register_provider_tool(
    "google_image_generation", lambda: GoogleImageGeneration().spec()
)

# Register alternative tool names with best defaults for user experience
register_provider_tool(
    "web_search",
    lambda: GoogleWebSearch().spec(),
)  # Default to Google Gemini 2.0+ search (most advanced)
register_provider_tool(
    "code_execution", lambda: GoogleCodeExecution().spec()
)  # Default to Google (most reliable)
register_provider_tool(
    "code_interpreter", lambda: OpenAICodeInterpreter().spec()
)  # Default to OpenAI (most advanced)
register_provider_tool(
    "image_generation", lambda: OpenAIImageGeneration().spec()
)  # Default to OpenAI Responses API
register_provider_tool(
    "url_context", lambda: GoogleUrlContext().spec()
)  # Default to Google

__all__ = [
    "FetchUrlTool",
    "JsonFormatTool",
    "OpenAIWebSearch",
    "OpenAICodeInterpreter",
    "OpenAIImageGeneration",
    "AnthropicWebSearch",
    "AnthropicWebFetch",
    "AnthropicCodeExecution",
    "GoogleWebSearch",
    "GoogleSearchRetrieval",
    "GoogleCodeExecution",
    "GoogleUrlContext",
    "GoogleImageGeneration",
]
