#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Type, Callable, Optional, List, Any

from .base import Tool
from ..schemas.tooling import ToolSpec


# Registries for different tool types
_LOCAL_TOOLS: Dict[str, Type[Tool]] = {}
_PROVIDER_TOOLS: Dict[str, Callable[[], ToolSpec]] = {}

# Core tool definitions (no duplication)
_SEARCH_TOOLS = [
    {
        "tool": "google_search",
        "provider": "google",
        "score": 9,
        "desc": "Google Gemini 2.0+ search with grounding",
    },
    {
        "tool": "anthropic_web_search",
        "provider": "anthropic",
        "score": 8,
        "desc": "Anthropic web search with citations",
    },
    {
        "tool": "openai_web_search",
        "provider": "openai",
        "score": 7,
        "desc": "OpenAI web search via Responses API",
    },
]

_CODE_TOOLS = [
    {
        "tool": "google_code_execution",
        "provider": "google",
        "score": 9,
        "desc": "Google Gemini code execution with data analysis",
    },
    {
        "tool": "openai_code_interpreter",
        "provider": "openai",
        "score": 8,
        "desc": "OpenAI Code Interpreter with containers",
    },
    {
        "tool": "anthropic_code_execution",
        "provider": "anthropic",
        "score": 7,
        "desc": "Anthropic code execution (beta)",
    },
]

_IMAGE_TOOLS = [
    {
        "tool": "openai_image_generation",
        "provider": "openai",
        "score": 9,
        "desc": "OpenAI image generation via Responses API",
    },
    {
        "tool": "google_image_generation",
        "provider": "google",
        "score": 8,
        "desc": "Google Gemini 2.5 native image generation",
    },
]

# Smart tool mappings - no duplication, aliases handled in _TOOL_ALIASES
_SMART_TOOLS: Dict[str, List[Dict[str, Any]]] = {
    "search": _SEARCH_TOOLS,
    "code": _CODE_TOOLS,
    "image": _IMAGE_TOOLS,
    "fetch": [
        {
            "tool": "fetch_url",
            "provider": "local",
            "score": 6,
            "desc": "Basic URL fetching",
        }
    ],
    "url": [
        {
            "tool": "google_url_context",
            "provider": "google",
            "score": 8,
            "desc": "Google URL context processing",
        },
        {
            "tool": "fetch_url",
            "provider": "local",
            "score": 6,
            "desc": "Basic URL fetching",
        },
    ],
    "json": [
        {
            "tool": "json_format",
            "provider": "local",
            "score": 7,
            "desc": "Local JSON formatting",
        }
    ],
}

# Tool aliases - map alternative names to primary names
_TOOL_ALIASES = {
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
    "webpage": "url",
    "url_context": "url",
    "format_json": "json",
    "json_format": "json",
}


def register_tool(name: str, cls: Type[Tool]) -> None:
    """Register a local tool that executes in the agent."""
    _LOCAL_TOOLS[name] = cls


def register_provider_tool(name: str, factory: Callable[[], ToolSpec]) -> None:
    """Register a provider-native tool factory."""
    _PROVIDER_TOOLS[name] = factory


def get_tool(name: str) -> Tool:
    """Get a local tool instance."""
    if name not in _LOCAL_TOOLS:
        raise KeyError(f"Unknown local tool: {name}. Registered: {list(_LOCAL_TOOLS)}")
    return _LOCAL_TOOLS[name]()


def get_tool_spec(name: str, **kwargs: Any) -> ToolSpec:
    """Get a tool spec with smart routing support.

    Args:
        name: Tool name (generic, provider-specific, or local)
        **kwargs: Smart routing options:
            - provider_preference: Preferred provider
            - exclude_providers: List of providers to exclude

    Returns:
        ToolSpec for the requested tool
    """
    # Check local tools first
    if name in _LOCAL_TOOLS:
        return _LOCAL_TOOLS[name]().spec()

    # Check provider tools (exact match)
    if name in _PROVIDER_TOOLS:
        return _PROVIDER_TOOLS[name]()

    # Smart routing for generic names
    return _get_smart_tool_spec(name, **kwargs)


def _get_smart_tool_spec(name: str, **kwargs: Any) -> ToolSpec:
    """Internal smart routing logic."""
    provider_preference = kwargs.get("provider_preference")
    exclude_providers = kwargs.get("exclude_providers", [])

    # Resolve aliases
    resolved_name = _TOOL_ALIASES.get(name, name)

    # Get tool options
    if resolved_name not in _SMART_TOOLS:
        raise KeyError(
            f"Unknown tool: {name}. Available: {list(get_available_tools().keys())}"
        )

    tool_options = _SMART_TOOLS[resolved_name]

    # Filter by exclusions
    if exclude_providers:
        tool_options = [
            opt for opt in tool_options if opt["provider"] not in exclude_providers
        ]

    if not tool_options:
        raise KeyError(f"No available providers for {name} after filtering")

    # Try provider preference first
    if provider_preference:
        for option in tool_options:
            if option["provider"] == provider_preference:
                return _get_provider_tool_spec(option["tool"])

    # Use highest scored option
    best_option = max(tool_options, key=lambda x: x["score"])
    return _get_provider_tool_spec(best_option["tool"])


def _get_provider_tool_spec(tool_name: str) -> ToolSpec:
    """Get spec for a provider or local tool by name."""
    # Check provider tools first
    if tool_name in _PROVIDER_TOOLS:
        return _PROVIDER_TOOLS[tool_name]()

    # Check local tools
    if tool_name in _LOCAL_TOOLS:
        return _LOCAL_TOOLS[tool_name]().spec()

    raise KeyError(f"Tool not found: {tool_name}")


def get_available_tools() -> Dict[str, str]:
    """Get all available tools with their types."""
    result = {}

    # Add local tools
    for name in _LOCAL_TOOLS:
        result[name] = "local"

    # Add provider tools
    for name in _PROVIDER_TOOLS:
        result[name] = "provider"

    # Add smart tools and aliases
    for name in _SMART_TOOLS:
        result[name] = "smart"
    for alias in _TOOL_ALIASES:
        result[alias] = "smart"

    return result


def get_tool_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a tool."""
    # Resolve aliases
    resolved_name = _TOOL_ALIASES.get(name, name)

    if resolved_name in _SMART_TOOLS:
        options = _SMART_TOOLS[resolved_name]
        return {
            "name": name,
            "type": "smart",
            "providers": [
                {
                    "provider": opt["provider"],
                    "description": opt["desc"],
                    "score": opt["score"],
                }
                for opt in options
            ],
            "aliases": [
                alias
                for alias, target in _TOOL_ALIASES.items()
                if target == resolved_name
            ],
        }
    elif name in _LOCAL_TOOLS:
        return {"name": name, "type": "local", "description": "Local tool"}
    elif name in _PROVIDER_TOOLS:
        return {
            "name": name,
            "type": "provider",
            "description": "Provider-specific tool",
        }
    else:
        raise KeyError(f"Unknown tool: {name}")


def recommend_tools(task_description: str) -> List[str]:
    """Get tool recommendations based on keywords."""
    task = task_description.lower()
    tools = []

    if any(w in task for w in ["search", "find", "research", "news"]):
        tools.append("search")
    if any(w in task for w in ["code", "calculate", "analyze", "python"]):
        tools.append("code")
    if any(w in task for w in ["image", "draw", "visual", "generate"]):
        tools.append("image")
    if any(w in task for w in ["json", "format", "structure"]):
        tools.append("json")
    if any(w in task for w in ["url", "fetch", "download"]):
        tools.append("url" if "content" in task else "fetch")

    return tools


def get_tool_recommendations(
    task_description: str, provider_preference: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get tool recommendations with descriptions."""
    recommendations = []

    for tool_name in recommend_tools(task_description):
        try:
            info = get_tool_info(tool_name)
            if info["type"] == "smart" and info["providers"]:
                best = info["providers"][0]
                recommendations.append(
                    {
                        "tool": tool_name,
                        "provider": best["provider"],
                        "reason": best["description"],
                    }
                )
            else:
                recommendations.append(
                    {
                        "tool": tool_name,
                        "provider": "local",
                        "reason": f"Local {tool_name} tool",
                    }
                )
        except KeyError:
            continue

    return recommendations
