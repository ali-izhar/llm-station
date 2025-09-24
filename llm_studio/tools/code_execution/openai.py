from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ...schemas.tooling import ToolSpec


class OpenAICodeInterpreter:
    """Factory for OpenAI Code Interpreter tool (Responses API).

    OpenAI's Code Interpreter tool allows models to write and run Python code in a
    sandboxed environment to solve complex problems. The tool operates in a fully
    sandboxed virtual machine container and can:

    - Process files with diverse data and formatting
    - Generate files with data and images of graphs
    - Write and run code iteratively until problems are solved
    - Boost visual intelligence for image processing (crop, zoom, rotate, transform)

    The model knows this tool as the "python tool". For explicit invocation,
    ask for "the python tool" in prompts.

    Container Management:
        The tool requires a container (sandboxed VM) which can be created in two ways:
        1. Auto mode: Automatically creates/reuses containers
        2. Explicit mode: Use pre-created container IDs

    File Support:
        - Input files: Automatically uploaded to container from model input
        - Output files: Generated files cited in message annotations
        - File operations: Upload, download, list container files
        - Supported formats: 30+ file types (CSV, JSON, images, code files, etc.)

    Container Expiration:
        - Containers expire after 20 minutes of inactivity
        - Treat containers as ephemeral - download needed files while active
        - Expired containers cannot be reactivated, create new ones instead

    Args:
        container_type: Container creation mode - "auto" or explicit container ID
        file_ids: List of file IDs to include in auto-created containers
        name: Optional name for explicitly created containers

    Examples:
        Auto container (recommended):
            tool = OpenAICodeInterpreter()

        Auto container with files:
            tool = OpenAICodeInterpreter(
                container_type="auto",
                file_ids=["file-abc123", "file-def456"]
            )

        Explicit container:
            tool = OpenAICodeInterpreter(container_type="cntr_abc123")

    Usage with Agent:
        agent = Agent(provider="openai", model="gpt-4.1", api="responses")
        response = agent.generate(
            "Calculate 4 * 3.82, then find its square root twice",
            tools=["openai_code_interpreter"]
        )

    Limitations:
        - Available only in Responses API (not Chat Completions)
        - 100 RPM rate limit per organization
        - Containers expire after 20 minutes of inactivity
        - All container data is ephemeral and should be downloaded if needed
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


class OpenAICodeInterpreterExplicit:
    """Helper factory for explicit container creation workflow.

    This class provides a more explicit interface for creating containers
    separately from tool usage, following the two-step process in the docs.

    Usage:
        # Step 1: Create container
        container_factory = OpenAICodeInterpreterExplicit()
        container_spec = container_factory.container_spec(name="my-analysis")

        # Step 2: Use container with tool
        tool = OpenAICodeInterpreter(container_type=container_id)
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize explicit container factory.

        Args:
            name: Optional name for the container
        """
        self.name = name

    def container_spec(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Generate container creation specification.

        This can be used with OpenAI's /v1/containers endpoint to create
        a container explicitly before using it with the Code Interpreter tool.

        Args:
            name: Container name (overrides instance name if provided)

        Returns:
            Dict suitable for container creation API call
        """
        container_name = name or self.name or "llm-studio-container"

        return {
            "name": container_name,
            # Additional container parameters can be added here as the API evolves
        }

    def tool_spec(self, container_id: str) -> ToolSpec:
        """Generate tool spec for pre-created container.

        Args:
            container_id: ID of the pre-created container (e.g., "cntr_abc123")

        Returns:
            ToolSpec configured for the specific container
        """
        if not container_id.startswith("cntr_"):
            raise ValueError(
                f"container_id must start with 'cntr_', got: {container_id}"
            )

        return OpenAICodeInterpreter(container_type=container_id).spec()


# Convenience factory functions for common use cases
def create_auto_code_interpreter(file_ids: Optional[List[str]] = None) -> ToolSpec:
    """Create auto-mode Code Interpreter tool (recommended).

    Args:
        file_ids: Optional list of file IDs to include in container

    Returns:
        ToolSpec for auto-mode Code Interpreter
    """
    return OpenAICodeInterpreter(container_type="auto", file_ids=file_ids).spec()


def create_explicit_code_interpreter(container_id: str) -> ToolSpec:
    """Create Code Interpreter tool with explicit container ID.

    Args:
        container_id: Pre-created container ID (e.g., "cntr_abc123")

    Returns:
        ToolSpec for explicit container Code Interpreter
    """
    return OpenAICodeInterpreter(container_type=container_id).spec()


# File format constants for reference
SUPPORTED_FILE_FORMATS = {
    # Code files
    ".c": "text/x-c",
    ".cs": "text/x-csharp",
    ".cpp": "text/x-c++",
    ".java": "text/x-java",
    ".php": "text/x-php",
    ".py": "text/x-python",
    ".rb": "text/x-ruby",
    ".js": "text/javascript",
    ".ts": "application/typescript",
    ".css": "text/css",
    ".sh": "application/x-sh",
    # Data files
    ".csv": "text/csv",
    ".json": "application/json",
    ".xml": "application/xml",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".tex": "text/x-tex",
    # Office documents
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pdf": "application/pdf",
    # Images
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".gif": "image/gif",
    ".png": "image/png",
    # Archives and data
    ".pkl": "application/octet-stream",
    ".tar": "application/x-tar",
    ".zip": "application/zip",
}
