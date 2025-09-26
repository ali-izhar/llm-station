#!/usr/bin/env python3
"""
LLM Station Tools - Smart Tool Interface

This module provides a clean, provider-agnostic tool interface using simple,
memorable names that automatically route to the best available provider.

Available Smart Tools:
- search: Web search and research
- code: Code execution and data analysis
- image: Image generation and creation
- json: JSON formatting and parsing
- fetch: URL fetching and downloading
- url: URL content processing and extraction

Usage:
    agent.generate("Your prompt", tools=["search", "code", "json"])
"""

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

# Register local tools
register_tool("fetch_url", FetchUrlTool)
register_tool("json_format", JsonFormatTool)

# Register provider tools for smart routing
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

register_provider_tool("anthropic_web_search", lambda: AnthropicWebSearch().spec())
register_provider_tool("anthropic_web_fetch", lambda: AnthropicWebFetch().spec())
register_provider_tool(
    "anthropic_code_execution", lambda: AnthropicCodeExecution().spec()
)

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

__all__ = [
    # Local tools
    "FetchUrlTool",
    "JsonFormatTool",
    # Provider tool factories
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
