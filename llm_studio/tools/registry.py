#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Type, Callable

from .base import Tool
from ..schemas.tooling import ToolSpec


# Registries for different tool types
_LOCAL_TOOLS: Dict[str, Type[Tool]] = {}
_PROVIDER_TOOLS: Dict[str, Callable[[], ToolSpec]] = {}


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


def get_tool_spec(name: str) -> ToolSpec:
    """Get a tool spec (local or provider-native)."""
    # Check local tools first
    if name in _LOCAL_TOOLS:
        return _LOCAL_TOOLS[name]().spec()

    # Check provider tools
    if name in _PROVIDER_TOOLS:
        return _PROVIDER_TOOLS[name]()

    raise KeyError(f"Unknown tool: {name}. Available: {list_all_tools()}")


def list_tools() -> Dict[str, Type[Tool]]:
    """List local tools only (for backward compatibility)."""
    return dict(_LOCAL_TOOLS)


def list_provider_tools() -> Dict[str, Callable[[], ToolSpec]]:
    """List provider-native tools."""
    return dict(_PROVIDER_TOOLS)


def list_all_tools() -> Dict[str, str]:
    """List all available tools with their types."""
    result = {}

    # Add local tools
    for name in _LOCAL_TOOLS:
        result[name] = "local"

    # Add provider tools
    for name in _PROVIDER_TOOLS:
        result[name] = "provider"

    return result
