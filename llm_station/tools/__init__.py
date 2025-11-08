"""Tool base class, registry, and implementations."""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Type

from ..types import ToolResult, ToolSpec

# Smart tool routing
_SMART_TOOLS: Dict[str, list] = {
    "search": [
        {"tool": "google_search", "provider": "google", "score": 9},
        {"tool": "anthropic_web_search", "provider": "anthropic", "score": 8},
        {"tool": "openai_web_search", "provider": "openai", "score": 7},
        {"tool": "huggingface_web_search", "provider": "huggingface", "score": 6},
    ],
    "code": [
        {"tool": "google_code_execution", "provider": "google", "score": 9},
        {"tool": "openai_code_interpreter", "provider": "openai", "score": 8},
        {"tool": "anthropic_code_execution", "provider": "anthropic", "score": 7},
    ],
    "image": [
        {"tool": "openai_image_generation", "provider": "openai", "score": 9},
        {"tool": "google_image_generation", "provider": "google", "score": 8},
    ],
    "fetch": [{"tool": "fetch_url", "provider": "local", "score": 6}],
    "url": [
        {"tool": "google_url_context", "provider": "google", "score": 9},
        {"tool": "fetch_url", "provider": "local", "score": 6},
    ],
    "json": [{"tool": "json_format", "provider": "local", "score": 7}],
}

_TOOL_ALIASES: Dict[str, str] = {
    "websearch": "search",
    "web_search": "search",
    "python": "code",
    "execute": "code",
    "compute": "code",
    "run": "code",
    "draw": "image",
    "create_image": "image",
    "generate_image": "image",
    "download": "fetch",
    "format_json": "json",
    "json_format": "json",
}


class Tool(abc.ABC):
    """Base tool class."""

    @abc.abstractmethod
    def spec(self) -> ToolSpec:
        """Return tool specification."""
        raise NotImplementedError

    def validate_args(self, args: Dict[str, Any]) -> None:
        """Validate tool arguments."""
        schema = self.spec().input_schema or {}
        required = schema.get("required", [])
        for r in required:
            if r not in args:
                raise ValueError(f"Missing required argument: {r}")

    @abc.abstractmethod
    def run(self, *, tool_call_id: str, **kwargs: Any) -> ToolResult:
        """Execute tool."""
        raise NotImplementedError


# Simple registry
_registry: Dict[str, Type[Tool]] = {}
_provider_tools: Dict[str, ToolSpec] = {}  # Provider-native tools


def register(name: str, tool_class: Type[Tool]) -> None:
    """Register a local tool."""
    _registry[name] = tool_class


def register_provider_tool(name: str, spec: ToolSpec) -> None:
    """Register a provider-native tool."""
    _provider_tools[name] = spec


def get(name: str) -> Tool:
    """Get a local tool instance."""
    if name not in _registry:
        raise KeyError(f"Unknown tool: {name}. Available: {list(_registry.keys())}")
    return _registry[name]()


def get_spec(name: str, provider_preference: Optional[str] = None) -> ToolSpec:
    """Get a tool spec with smart routing support."""
    # Check local tools first
    if name in _registry:
        return _registry[name]().spec()

    # Check provider tools (exact match)
    if name in _provider_tools:
        return _provider_tools[name]

    # Smart routing for generic names
    resolved_name = _TOOL_ALIASES.get(name, name)

    if resolved_name not in _SMART_TOOLS:
        available_tools = ", ".join(sorted(_SMART_TOOLS.keys()))
        raise KeyError(f"Unknown tool: {name}. Available: {available_tools}")

    tool_options = _SMART_TOOLS[resolved_name]

    # Try provider preference first
    if provider_preference:
        for option in tool_options:
            if option["provider"] == provider_preference:
                tool_name = option["tool"]
                if tool_name in _provider_tools:
                    return _provider_tools[tool_name]
                if tool_name in _registry:
                    return _registry[tool_name]().spec()

    # Use highest scored option
    best_option = max(tool_options, key=lambda x: x["score"])
    tool_name = best_option["tool"]
    if tool_name in _provider_tools:
        return _provider_tools[tool_name]
    if tool_name in _registry:
        return _registry[tool_name]().spec()

    raise KeyError(f"Tool not found: {tool_name}")


def list_tools() -> Dict[str, str]:
    """List all available tools."""
    result = {}
    for name in _registry:
        result[name] = "local"
    for name in _provider_tools:
        result[name] = "provider"
    return result


# Import tool implementations
from .search import (
    OpenAISearch,
    AnthropicSearch,
    GoogleSearch,
    HuggingFaceSearch,
)
from .code import OpenAICode, AnthropicCode, GoogleCode
from .image import OpenAIImage, GoogleImage
from .local import FetchUrlTool, JsonFormatTool

# Register local tools
register("fetch_url", FetchUrlTool)
register("json_format", JsonFormatTool)

# Register provider tools - data-driven approach
_PROVIDER_TOOL_REGISTRATIONS = [
    # OpenAI tools
    ("openai_web_search", OpenAISearch, {}),
    ("openai_web_search_preview", OpenAISearch, {"preview": True}),
    ("openai_code_interpreter", OpenAICode, {}),
    ("openai_image_generation", OpenAIImage, {}),
    # Anthropic tools
    ("anthropic_web_search", AnthropicSearch, {}),
    ("anthropic_code_execution", AnthropicCode, {}),
    # Google tools
    ("google_search", GoogleSearch, {}),
    ("google_code_execution", GoogleCode, {}),
    ("google_image_generation", GoogleImage, {}),
    # HuggingFace tools
    ("huggingface_web_search", HuggingFaceSearch, {}),
]

# Register Google URL context as a server-side tool spec
from ..types import ToolSpec

register_provider_tool(
    "google_url_context",
    ToolSpec(
        name="google_url_context",
        description="Google Gemini URL context tool for processing URLs",
        input_schema={},
        provider="google",
        provider_type="url_context",
    ),
)

for tool_name, tool_class, init_kwargs in _PROVIDER_TOOL_REGISTRATIONS:
    register_provider_tool(tool_name, tool_class(**init_kwargs).spec())

__all__ = [
    # Base class and registry
    "Tool",
    "register",
    "register_provider_tool",
    "get",
    "get_spec",
    "list_tools",
    # Search tools
    "OpenAISearch",
    "AnthropicSearch",
    "GoogleSearch",
    "HuggingFaceSearch",
    # Code tools
    "OpenAICode",
    "AnthropicCode",
    "GoogleCode",
    # Image tools
    "OpenAIImage",
    "GoogleImage",
    # Local tools
    "FetchUrlTool",
    "JsonFormatTool",
]
