#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

from ...schemas.tooling import ToolSpec


class OpenAICodeInterpreter:
    """Factory for OpenAI Code Interpreter tool.

    Provides Python code execution in sandboxed containers with file processing
    and data analysis capabilities via Responses API.
    """

    def __init__(
        self,
        *,
        container_type: Union[str, Dict[str, Any]] = "auto",
        file_ids: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize OpenAI Code Interpreter tool.

        Args:
            container_type: Either "auto" for automatic container creation,
                          or a container ID string like "cntr_abc123"
            file_ids: List of file IDs to include in auto containers
            name: Optional name for containers (used for explicit creation)
        """
        self.container_type = container_type
        self.file_ids = file_ids or []
        self.name = name

        # Validate inputs
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate container configuration parameters."""
        if isinstance(self.container_type, str):
            if self.container_type not in {
                "auto"
            } and not self.container_type.startswith("cntr_"):
                raise ValueError(
                    f"container_type must be 'auto' or a container ID starting with 'cntr_', "
                    f"got: {self.container_type}"
                )
        elif isinstance(self.container_type, dict):
            # Allow dict format for advanced configurations
            if "type" not in self.container_type:
                raise ValueError("container_type dict must have 'type' field")
        else:
            raise ValueError(
                f"container_type must be string or dict, got: {type(self.container_type)}"
            )

        # Validate file IDs format
        if self.file_ids:
            for file_id in self.file_ids:
                if not isinstance(file_id, str) or not file_id.strip():
                    raise ValueError(
                        f"file_ids must contain non-empty strings, got: {file_id}"
                    )

    def _build_container_config(self) -> Union[str, Dict[str, Any]]:
        """Build container configuration for the tool spec."""
        if isinstance(self.container_type, dict):
            # Use provided dict directly
            return self.container_type
        elif self.container_type == "auto":
            # Auto mode with optional file IDs
            config = {"type": "auto"}
            if self.file_ids:
                config["file_ids"] = self.file_ids
            return config
        else:
            # Explicit container ID
            return self.container_type

    def spec(self) -> ToolSpec:
        """Generate ToolSpec for OpenAI Code Interpreter tool."""
        container_config = self._build_container_config()

        provider_config = {"container": container_config}

        # Add optional name for container creation
        if self.name:
            provider_config["name"] = self.name

        return ToolSpec(
            name="code_interpreter",
            description="OpenAI Code Interpreter tool for running Python code in sandboxed containers (Responses API)",
            input_schema={},  # Not used for provider-native tools
            requires_network=False,  # Sandboxed environment
            requires_filesystem=True,  # Can create/access files in container
            provider="openai",
            provider_type="code_interpreter",
            provider_config=provider_config,
        )
