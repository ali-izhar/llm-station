#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class OpenAIImageGeneration:
    """Factory for OpenAI Image Generation tool (Responses API).

    Uses the Responses API image_generation tool which integrates image generation
    into conversations and leverages the model's world knowledge.

    The Responses API approach:
    - Uses mainline models (gpt-4.1-mini, gpt-4.1, gpt-5) that call the image_generation tool
    - Supports conversational context and multi-turn editing
    - Leverages world knowledge for better image generation
    - Should work without special organization verification

    Args:
        size: Image dimensions ("1024x1024", "1024x1536", "1536x1024", "auto")
        quality: Rendering quality ("low", "medium", "high", "auto")
        format: Output format ("png", "jpeg", "webp")
        compression: Compression level 0-100 for JPEG/WebP
        background: Background type ("transparent", "opaque", "auto")
        partial_images: Number of streaming partial images (0-3)
        input_fidelity: Input preservation ("low", "high")

    Examples:
        Basic generation:
            tool = OpenAIImageGeneration()

        High quality with transparency:
            tool = OpenAIImageGeneration(
                size="1024x1024",
                quality="high",
                background="transparent",
                format="png"
            )

        Streaming generation:
            tool = OpenAIImageGeneration(
                partial_images=2,
                quality="medium"
            )

    Usage:
        agent = Agent(provider="openai", model="gpt-4.1-mini", api_key=key)
        response = agent.generate(
            "Generate an image of a red circle",
            tools=["openai_image_generation"]
        )
    """

    def __init__(
        self,
        *,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        format: Optional[str] = None,
        compression: Optional[int] = None,
        background: Optional[str] = None,
        partial_images: Optional[int] = None,
        input_fidelity: Optional[str] = None,
    ):
        self.size = size
        self.quality = quality
        self.format = format
        self.compression = compression
        self.background = background
        self.partial_images = partial_images
        self.input_fidelity = input_fidelity

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate image generation configuration parameters."""
        # Validate size
        if self.size is not None:
            valid_sizes = {
                "1024x1024",
                "1024x1536",
                "1536x1024",
                "1792x1024",
                "1024x1792",
                "auto",
            }
            if self.size not in valid_sizes:
                raise ValueError(f"size must be one of {valid_sizes}, got: {self.size}")

        # Validate quality (Responses API values)
        if self.quality is not None:
            valid_qualities = {"low", "medium", "high", "auto"}
            if self.quality not in valid_qualities:
                raise ValueError(
                    f"quality must be one of {valid_qualities}, got: {self.quality}"
                )

        # Validate format
        if self.format is not None:
            valid_formats = {"png", "jpeg", "webp"}
            if self.format not in valid_formats:
                raise ValueError(
                    f"format must be one of {valid_formats}, got: {self.format}"
                )

        # Validate compression
        if self.compression is not None:
            if not isinstance(self.compression, int) or not (
                0 <= self.compression <= 100
            ):
                raise ValueError(f"compression must be 0-100, got: {self.compression}")
            if self.format and self.format not in {"jpeg", "webp"}:
                raise ValueError(
                    f"compression only valid for jpeg/webp, got format: {self.format}"
                )

        # Validate background
        if self.background is not None:
            valid_backgrounds = {"transparent", "opaque", "auto"}
            if self.background not in valid_backgrounds:
                raise ValueError(
                    f"background must be one of {valid_backgrounds}, got: {self.background}"
                )

        # Validate partial_images
        if self.partial_images is not None:
            if not isinstance(self.partial_images, int) or not (
                0 <= self.partial_images <= 3
            ):
                raise ValueError(
                    f"partial_images must be 0-3, got: {self.partial_images}"
                )

        # Validate input_fidelity
        if self.input_fidelity is not None:
            valid_fidelity = {"low", "high"}
            if self.input_fidelity not in valid_fidelity:
                raise ValueError(
                    f"input_fidelity must be one of {valid_fidelity}, got: {self.input_fidelity}"
                )

    def spec(self) -> ToolSpec:
        """Generate ToolSpec for OpenAI Image Generation tool (Responses API)."""
        provider_config: Dict[str, Any] = {}

        # Add all configuration options for the Responses API image_generation tool
        if self.size is not None:
            provider_config["size"] = self.size
        if self.quality is not None:
            provider_config["quality"] = self.quality
        if self.format is not None:
            provider_config["format"] = self.format
        if self.compression is not None:
            provider_config["compression"] = self.compression
        if self.background is not None:
            provider_config["background"] = self.background
        if self.partial_images is not None:
            provider_config["partial_images"] = self.partial_images
        if self.input_fidelity is not None:
            provider_config["input_fidelity"] = self.input_fidelity

        return ToolSpec(
            name="image_generation",
            description="OpenAI Image Generation tool for creating and editing images (Responses API)",
            input_schema={},  # Not used for provider-native tools
            requires_network=True,
            requires_filesystem=False,
            provider="openai",
            provider_type="image_generation",
            provider_config=provider_config or None,
        )


class OpenAIImageGenerationAdvanced:
    """Advanced factory for multi-turn image editing with Responses API.

    Supports:
    - Previous response ID references for context
    - Specific image generation call ID references
    - Multi-turn conversational editing
    """

    def __init__(
        self,
        *,
        previous_response_id: Optional[str] = None,
        image_generation_call_id: Optional[str] = None,
        **kwargs: Any,
    ):
        self.previous_response_id = previous_response_id
        self.image_generation_call_id = image_generation_call_id
        self.base_tool = OpenAIImageGeneration(**kwargs)

    def spec(self) -> ToolSpec:
        """Generate ToolSpec with multi-turn references."""
        base_spec = self.base_tool.spec()

        # Add multi-turn configuration
        if self.previous_response_id or self.image_generation_call_id:
            config = base_spec.provider_config or {}

            if self.previous_response_id:
                config["previous_response_id"] = self.previous_response_id
            if self.image_generation_call_id:
                config["image_generation_call_id"] = self.image_generation_call_id

            base_spec.provider_config = config

        return base_spec


# Convenience factory functions
def create_basic_image_generation() -> ToolSpec:
    """Create basic image generation tool with default settings."""
    return OpenAIImageGeneration().spec()


def create_hd_image_generation(size: str = "1024x1024") -> ToolSpec:
    """Create HD quality image generation tool."""
    return OpenAIImageGeneration(quality="high", size=size).spec()


def create_transparent_image_generation() -> ToolSpec:
    """Create image generation tool with transparent background."""
    return OpenAIImageGeneration(
        background="transparent", format="png", quality="high"
    ).spec()


def create_streaming_image_generation(partial_images: int = 2) -> ToolSpec:
    """Create streaming image generation with partial images."""
    return OpenAIImageGeneration(partial_images=partial_images, quality="medium").spec()


# Constants for reference
SUPPORTED_MODELS = {
    "gpt-4.1-mini": "Cost-effective model with image generation",
    "gpt-4.1": "Advanced model with image generation",
    "gpt-5": "Latest model with image generation",
    "gpt-4o": "Multimodal model with image generation",
}

SUPPORTED_SIZES = {
    "1024x1024": "Square format (1:1)",
    "1024x1536": "Portrait format (2:3)",
    "1536x1024": "Landscape format (3:2)",
    "1792x1024": "Wide landscape (7:4)",
    "1024x1792": "Tall portrait (4:7)",
    "auto": "Model automatically selects best size",
}

QUALITY_OPTIONS = {
    "low": "Fast generation, lower detail (272 tokens)",
    "medium": "Balanced speed and quality (1056 tokens)",
    "high": "Maximum detail, slower (4160 tokens)",
    "auto": "Model automatically selects quality",
}

BACKGROUND_OPTIONS = {
    "transparent": "Transparent background (PNG/WebP only)",
    "opaque": "Solid background color",
    "auto": "Model automatically selects background type",
}

FORMAT_OPTIONS = {
    "png": "Lossless format, supports transparency",
    "jpeg": "Lossy format, smaller files, faster",
    "webp": "Modern format, efficient compression",
}
