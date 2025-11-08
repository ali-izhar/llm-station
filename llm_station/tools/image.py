"""Image generation tools for all providers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import ToolSpec

# Validation constants
VALID_SIZES = {
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "1792x1024",
    "1024x1792",
    "auto",
}

VALID_QUALITIES = {"low", "medium", "high", "auto"}
VALID_FORMATS = {"png", "jpeg", "webp"}
VALID_BACKGROUNDS = {"transparent", "opaque", "auto"}
VALID_FIDELITY = {"low", "high"}


class OpenAIImage:
    """OpenAI Image Generation tool for creating and editing images."""

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
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate image generation configuration parameters."""
        if self.size is not None and self.size not in VALID_SIZES:
            raise ValueError(f"size must be one of {VALID_SIZES}, got: {self.size}")

        if self.quality is not None and self.quality not in VALID_QUALITIES:
            raise ValueError(
                f"quality must be one of {VALID_QUALITIES}, got: {self.quality}"
            )

        if self.format is not None and self.format not in VALID_FORMATS:
            raise ValueError(
                f"format must be one of {VALID_FORMATS}, got: {self.format}"
            )

        if self.compression is not None:
            if not isinstance(self.compression, int) or not (
                0 <= self.compression <= 100
            ):
                raise ValueError(f"compression must be 0-100, got: {self.compression}")
            if self.format and self.format not in {"jpeg", "webp"}:
                raise ValueError(
                    f"compression only valid for jpeg/webp, got format: {self.format}"
                )

        if self.background is not None and self.background not in VALID_BACKGROUNDS:
            raise ValueError(
                f"background must be one of {VALID_BACKGROUNDS}, got: {self.background}"
            )

        if self.partial_images is not None:
            if not isinstance(self.partial_images, int) or not (
                0 <= self.partial_images <= 3
            ):
                raise ValueError(
                    f"partial_images must be 0-3, got: {self.partial_images}"
                )

        if (
            self.input_fidelity is not None
            and self.input_fidelity not in VALID_FIDELITY
        ):
            raise ValueError(
                f"input_fidelity must be one of {VALID_FIDELITY}, got: {self.input_fidelity}"
            )

    def spec(self) -> ToolSpec:
        """Generate ToolSpec for OpenAI Image Generation tool."""
        provider_config: Dict[str, Any] = {}

        for param in [
            "size",
            "quality",
            "format",
            "compression",
            "background",
            "partial_images",
            "input_fidelity",
        ]:
            value = getattr(self, param, None)
            if value is not None:
                provider_config[param] = value

        return ToolSpec(
            name="image_generation",
            description="OpenAI Image Generation tool for creating and editing images (Responses API)",
            input_schema={},
            requires_network=True,
            requires_filesystem=False,
            provider="openai",
            provider_type="image_generation",
            provider_config=provider_config or None,
        )


class GoogleImage:
    """Google Gemini image generation tool with multimodal output."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="image_generation",
            description="Google Gemini 2.5 native image generation - creates and edits images with multimodal reasoning",
            input_schema={},
            requires_network=False,
            requires_filesystem=False,
            provider="google",
            provider_type="image_generation",
            provider_config=None,
        )
