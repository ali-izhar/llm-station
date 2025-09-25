#!/usr/bin/env python3
from __future__ import annotations

from ...schemas.tooling import ToolSpec


class GoogleImageGeneration:
    """Factory for Google Gemini image generation tool.

    Provides native image generation capabilities with multimodal output
    for Gemini 2.5+ models.
    """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="image_generation",
            description="Google Gemini 2.5 native image generation - creates and edits images with multimodal reasoning",
            input_schema={},
            requires_network=False,  # Built into model
            requires_filesystem=False,  # Image data in response
            provider="google",
            provider_type="image_generation",
            provider_config=None,
        )
