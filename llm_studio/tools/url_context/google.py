#!/usr/bin/env python3
from __future__ import annotations

from ...schemas.tooling import ToolSpec


class GoogleUrlContext:
    """Factory for Google URL context tool.

    Provides direct content processing from web pages, PDFs, and images
    for Gemini models with automatic grounding.
    """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="url_context",
            description="Google Gemini URL context tool - directly processes content from web pages, PDFs, and images",
            input_schema={},
            requires_network=True,
            requires_filesystem=False,  # Processes remote content, not local files
            provider="google",
            provider_type="url_context",
            provider_config=None,
        )
