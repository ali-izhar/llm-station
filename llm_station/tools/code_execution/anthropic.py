#!/usr/bin/env python3
"""Anthropic Code Execution Tool (Messages API with Beta Features)"""

from __future__ import annotations
from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class AnthropicCodeExecution:
    """Factory for Anthropic code execution tool.

    Provides secure Python and bash execution in sandboxed containers
    with file manipulation and data science libraries.
    """

    def __init__(
        self,
        *,
        container_id: Optional[str] = None,
        max_execution_time: Optional[int] = None,
    ) -> None:
        """Initialize Anthropic Code Execution tool.

        Args:
            container_id: Optional existing container ID for reuse
            max_execution_time: Optional timeout for execution (server-controlled)
        """
        self.container_id = container_id
        self.max_execution_time = max_execution_time

    def spec(self) -> ToolSpec:
        """Generate ToolSpec for Anthropic Code Execution tool."""
        provider_config: Dict[str, Any] = {}

        # Add container configuration if specified
        if self.container_id:
            provider_config["container_id"] = self.container_id
        if self.max_execution_time:
            provider_config["max_execution_time"] = self.max_execution_time

        return ToolSpec(
            name="code_execution",
            description="Anthropic code execution tool - Bash commands and file manipulation in secure sandbox",
            input_schema={},
            requires_network=False,  # No internet access in sandbox
            requires_filesystem=True,  # Can create and manipulate files
            provider="anthropic",
            provider_type="code_execution_20250825",
            provider_config=provider_config or None,
        )
