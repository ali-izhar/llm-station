#!/usr/bin/env python3
"""Anthropic Code Execution Tool (Messages API with Beta Features)"""

from __future__ import annotations
from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class AnthropicCodeExecution:
    """Factory for Anthropic server code execution tool (Messages API).

    Claude's code execution tool provides secure, sandboxed code execution with:
    - Bash command execution for system operations
    - File manipulation (create, view, edit files)
    - Python environment with data science libraries
    - Container persistence across multi-turn conversations

    Key Features:
    - **Bash Commands**: Execute shell commands for system operations and package management
    - **File Operations**: Create, view, and edit files directly using text_editor_code_execution
    - **Python Environment**: Pre-loaded with pandas, numpy, matplotlib, scikit-learn, etc.
    - **Secure Sandbox**: Isolated environment with no internet access
    - **Container Reuse**: Maintain files and state across conversation turns
    - **File Upload Support**: Process uploaded CSV, Excel, images, etc.

    Supported Models:
    - claude-opus-4-1-20250805 (latest)
    - claude-opus-4-20250514
    - claude-sonnet-4-20250514
    - claude-3-7-sonnet-20250219
    - claude-3-5-haiku-latest

    Environment Specifications:
    - **Python**: 3.11.12
    - **OS**: Linux x86_64 container
    - **Memory**: 1GiB RAM
    - **Storage**: 5GiB workspace
    - **CPU**: 1 CPU core
    - **Networking**: Disabled for security
    - **Expiration**: 30 days after creation

    Pre-installed Libraries:
    - **Data Science**: pandas, numpy, scipy, scikit-learn, statsmodels
    - **Visualization**: matplotlib, seaborn
    - **File Processing**: pyarrow, openpyxl, pillow, pypdf, pdfplumber
    - **Math**: sympy, mpmath
    - **Utilities**: tqdm, python-dateutil, sqlite

    Usage Examples:
        # Basic code execution
        tool = AnthropicCodeExecution()
        response = agent.generate(
            "Calculate the mean and standard deviation of [1,2,3,4,5,6,7,8,9,10]",
            tools=[tool.spec()]
        )

        # Data analysis workflow
        response = agent.generate(
            "Check Python version, create a CSV analysis script, and run it",
            tools=[tool.spec()]
        )

        # File manipulation
        response = agent.generate(
            "Create a config.yaml file, then update the database port from 5432 to 3306",
            tools=[tool.spec()]
        )

    Response Structure:
    The tool returns different response types:
    - bash_code_execution_tool_result: For shell commands
    - text_editor_code_execution_tool_result: For file operations
    - Results include stdout, stderr, return_code, and operation details

    Technical Details:
    - Uses `code_execution_20250825` tool version (latest)
    - Requires beta header: "code-execution-2025-08-25"
    - Server-side execution with full isolation
    - Container ID available for reuse across requests
    - Supports Files API integration for data upload

    Pricing:
    - $0.05 per session-hour (minimum 5 minutes)
    - Execution time billed even if files uploaded but tool not used
    - Container reuse recommended for cost efficiency

    Error Handling:
    - unavailable: Tool temporarily unavailable
    - execution_time_exceeded: Exceeded time limit
    - container_expired: Container no longer available
    - invalid_tool_input: Invalid parameters
    - too_many_requests: Rate limit exceeded
    - file_not_found: File doesn't exist (file operations)
    - string_not_found: String not found (str_replace operations)

    Requirements:
    - Beta feature: Add "code-execution-2025-08-25" to anthropic-beta header
    - Compatible models listed above
    - Files API integration requires additional beta header
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


class AnthropicCodeExecutionLegacy:
    """Legacy Anthropic code execution tool (Python-only).

    Use AnthropicCodeExecution for new implementations with full capabilities.
    This class is maintained for backward compatibility.
    """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_execution_legacy",
            description="Anthropic legacy code execution tool - Python only",
            input_schema={},
            requires_network=False,
            requires_filesystem=False,
            provider="anthropic",
            provider_type="code_execution_20250522",
            provider_config=None,
        )


# Convenience factory functions
def create_basic_code_execution() -> ToolSpec:
    """Create basic code execution tool with default settings."""
    return AnthropicCodeExecution().spec()


def create_container_code_execution(container_id: str) -> ToolSpec:
    """Create code execution tool with existing container for reuse."""
    return AnthropicCodeExecution(container_id=container_id).spec()


# Constants for reference
SUPPORTED_MODELS = {
    "claude-opus-4-1-20250805": "Latest model with full code execution",
    "claude-opus-4-20250514": "Opus model with code execution",
    "claude-sonnet-4-20250514": "Sonnet model with code execution",
    "claude-3-7-sonnet-20250219": "Sonnet 3.7 with code execution",
    "claude-3-5-haiku-latest": "Haiku model with code execution",
}

EXECUTION_CAPABILITIES = {
    "bash_commands": "Execute shell commands for system operations",
    "file_operations": "Create, view, edit files with text_editor",
    "python_execution": "Run Python code with data science libraries",
    "data_analysis": "Process CSV, Excel, JSON files",
    "visualization": "Create charts with matplotlib, seaborn",
    "container_persistence": "Maintain files across conversation turns",
}

PRE_INSTALLED_LIBRARIES = {
    "data_science": ["pandas", "numpy", "scipy", "scikit-learn", "statsmodels"],
    "visualization": ["matplotlib", "seaborn"],
    "file_processing": ["pyarrow", "openpyxl", "pillow", "pypdf", "pdfplumber"],
    "math": ["sympy", "mpmath"],
    "utilities": ["tqdm", "python-dateutil", "pytz", "sqlite"],
}
