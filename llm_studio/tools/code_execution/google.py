#!/usr/bin/env python3
from __future__ import annotations

from ...schemas.tooling import ToolSpec


class GoogleCodeExecution:
    """Factory for Google Gemini code execution tool.

    Provides Python code generation and execution with data analysis,
    visualization, and file processing capabilities.
    """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_execution",
            description="Google Gemini code execution tool - generates and runs Python code with data analysis and visualization capabilities",
            input_schema={},
            requires_network=False,  # Runs in secure sandboxed environment
            requires_filesystem=True,  # Can work with uploaded files
            provider="google",
            provider_type="code_execution",
            provider_config=None,
        )
