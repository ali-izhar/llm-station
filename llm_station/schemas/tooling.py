#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolSpec:
    """Normalized tool specification used across providers.

    Providers will convert this into their specific tool/function schemas.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON Schema for inputs (for local/function tools)
    # Optional: whether tool may access network/files, etc. for policy or display
    requires_network: bool = False
    requires_filesystem: bool = False
    # Provider-native tools can set these so adapters
    # pass them through without expecting local execution.
    provider: Optional[str] = None  # e.g., "openai", "google", "anthropic"
    provider_type: Optional[str] = None  # e.g., "web_search", "code_interpreter"
    provider_config: Optional[Dict[str, Any]] = None  # e.g., filters, user_location


@dataclass
class ToolResult:
    name: str
    content: str
    tool_call_id: str
    is_error: bool = False
    meta: Optional[Dict[str, Any]] = None
