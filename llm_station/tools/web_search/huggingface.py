#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class HuggingFaceWebSearch:
    """Factory for HuggingFace web search tool with SerpAPI integration.

    Provides web search functionality using SerpAPI and integrates results
    into the HuggingFace model's context via prompt augmentation.
    """

    def __init__(
        self,
        *,
        use_web_search: bool = True,
        extract_full_text: bool = True,
        max_urls: int = 3,
        num_search_results: int = 5,
    ) -> None:
        """
        Args:
            use_web_search: Enable/disable web search functionality
            extract_full_text: Whether to extract full text from URLs (default: True)
            max_urls: Maximum number of URLs to extract full text from (default: 3)
            num_search_results: Number of search results to return (default: 5)
        """
        # Validate parameters
        if max_urls < 0:
            raise ValueError("max_urls must be non-negative")
        if num_search_results <= 0:
            raise ValueError("num_search_results must be greater than 0")

        self.use_web_search = use_web_search
        self.extract_full_text = extract_full_text
        self.max_urls = max_urls
        self.num_search_results = num_search_results

    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        cfg: Dict[str, Any] = {
            "use_web_search": self.use_web_search,
            "extract_full_text": self.extract_full_text,
            "max_urls": self.max_urls,
            "num_search_results": self.num_search_results,
        }

        return ToolSpec(
            name="web_search",
            description="HuggingFace web search with SerpAPI integration. Performs web searches and integrates results into the model's context.",
            input_schema={},
            requires_network=True,
            provider="huggingface",
            provider_type="web_search",
            provider_config=cfg,
        )

