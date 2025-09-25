#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class GoogleWebSearch:
    """Factory for Google Gemini 2.0+ search tool with automatic grounding.

    The new search tool in Gemini 2.0 models automatically retrieves accurate and
    grounded artifacts from the web. Unlike the legacy search grounding in Gemini 1.5,
    this tool doesn't require dynamic retrieval threshold configuration.

    Key features:
    - Automatic search result retrieval and grounding
    - Citations and sources included in response metadata
    - Search entry point with rendered content for Google Search Suggestions
    - Enhanced grounding chunks with web information
    - Works with Gemini 2.0+ models (gemini-2.5-flash, gemini-2.5-pro, etc.)

    Usage:
        agent = Agent(provider="google", model="gemini-2.5-flash", api_key=api_key)
        response = agent.generate(
            "Research the latest AI developments",
            tools=["google_search"]
        )

        # Access grounding metadata
        if response.grounding_metadata:
            sources = response.grounding_metadata.get("sources", [])
            citations = response.grounding_metadata.get("citations", [])
            search_entry_point = response.grounding_metadata.get("search_entry_point")

    Note: You must enable Google Search Suggestions to display search queries
    that are included in the grounded response's metadata.
    """

    def __init__(self) -> None:
        pass

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="google_search",
            description="Google Gemini 2.0+ search tool with automatic grounding and citations",
            input_schema={},
            requires_network=True,
            provider="google",
            provider_type="google_search",
            provider_config=None,
        )


class GoogleSearchRetrieval:
    """Factory for legacy Google Search Retrieval tool (Gemini 1.5).

    Allows dynamic retrieval configuration when using legacy models. For new
    models prefer GoogleWebSearch.

    Args:
        mode: Retrieval mode. Valid values: "MODE_DYNAMIC"
        dynamic_threshold: Confidence threshold for search (0.0-1.0). Only search if confidence > threshold
    """

    def __init__(
        self,
        *,
        mode: Optional[str] = None,
        dynamic_threshold: Optional[float] = None,
    ) -> None:
        # Validate mode
        if mode is not None and mode not in {"MODE_DYNAMIC"}:
            raise ValueError(f"Invalid mode: {mode}. Valid values: MODE_DYNAMIC")

        # Validate dynamic_threshold
        if dynamic_threshold is not None and not (0.0 <= dynamic_threshold <= 1.0):
            raise ValueError("dynamic_threshold must be between 0.0 and 1.0")

        self.mode = mode
        self.dynamic_threshold = dynamic_threshold

    def spec(self) -> ToolSpec:
        cfg: Dict[str, Any] = {}
        # Mirror SDK naming: dynamic_retrieval_config { mode, dynamic_threshold }
        drc: Dict[str, Any] = {}
        if self.mode is not None:
            drc["mode"] = self.mode
        if self.dynamic_threshold is not None:
            drc["dynamic_threshold"] = self.dynamic_threshold
        if drc:
            cfg["dynamic_retrieval_config"] = drc
        return ToolSpec(
            name="google_search_retrieval",
            description="Legacy Google Search Retrieval tool (Gemini 1.5)",
            input_schema={},
            requires_network=True,
            provider="google",
            provider_type="google_search_retrieval",
            provider_config=cfg or None,
        )
