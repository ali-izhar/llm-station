#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class OpenAIImageGeneration:
    """Factory for OpenAI Image Generation tool.

    Provides image generation and editing capabilities via Responses API.
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
