"""Code execution tools for all providers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ..types import ToolSpec


class OpenAICode:
    """OpenAI Code Interpreter tool for Python execution in sandboxed containers."""

    def __init__(
        self,
        *,
        container_type: Union[str, Dict[str, Any]] = "auto",
        file_ids: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.container_type = container_type
        self.file_ids = file_ids or []
        self.name = name
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate container configuration parameters."""
        if isinstance(self.container_type, str):
            if self.container_type != "auto" and not self.container_type.startswith(
                "cntr_"
            ):
                raise ValueError(
                    f"container_type must be 'auto' or a container ID starting with 'cntr_', "
                    f"got: {self.container_type}"
                )
        elif isinstance(self.container_type, dict):
            if "type" not in self.container_type:
                raise ValueError("container_type dict must have 'type' field")
        else:
            raise ValueError(
                f"container_type must be string or dict, got: {type(self.container_type)}"
            )

        if self.file_ids:
            for file_id in self.file_ids:
                if not isinstance(file_id, str) or not file_id.strip():
                    raise ValueError(
                        f"file_ids must contain non-empty strings, got: {file_id}"
                    )

    def _build_container_config(self) -> Union[str, Dict[str, Any]]:
        """Build container configuration for the tool spec."""
        if isinstance(self.container_type, dict):
            return self.container_type
        elif self.container_type == "auto":
            config = {"type": "auto"}
            if self.file_ids:
                config["file_ids"] = self.file_ids
            return config
        else:
            return self.container_type

    def spec(self) -> ToolSpec:
        """Generate ToolSpec for OpenAI Code Interpreter tool."""
        container_config = self._build_container_config()
        provider_config = {"container": container_config}

        if self.name:
            provider_config["name"] = self.name

        return ToolSpec(
            name="code_interpreter",
            description="OpenAI Code Interpreter tool for running Python code in sandboxed containers (Responses API)",
            input_schema={},
            requires_network=False,
            requires_filesystem=True,
            provider="openai",
            provider_type="code_interpreter",
            provider_config=provider_config,
        )


class AnthropicCode:
    """Anthropic code execution tool for Python and bash in sandboxed containers."""

    def __init__(
        self,
        *,
        container_id: Optional[str] = None,
        max_execution_time: Optional[int] = None,
    ) -> None:
        self.container_id = container_id
        self.max_execution_time = max_execution_time

    def spec(self) -> ToolSpec:
        """Generate ToolSpec for Anthropic Code Execution tool."""
        provider_config: Dict[str, Any] = {}

        if self.container_id:
            provider_config["container_id"] = self.container_id
        if self.max_execution_time:
            provider_config["max_execution_time"] = self.max_execution_time

        return ToolSpec(
            name="code_execution",
            description="Anthropic code execution tool - Bash commands and file manipulation in secure sandbox",
            input_schema={},
            requires_network=False,
            requires_filesystem=True,
            provider="anthropic",
            provider_type="code_execution_20250825",
            provider_config=provider_config if provider_config else None,
        )


class GoogleCode:
    """Google Gemini code execution tool for Python with data analysis."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_execution",
            description="Google Gemini code execution tool - generates and runs Python code with data analysis and visualization capabilities",
            input_schema={},
            requires_network=False,
            requires_filesystem=True,
            provider="google",
            provider_type="code_execution",
            provider_config=None,
        )
