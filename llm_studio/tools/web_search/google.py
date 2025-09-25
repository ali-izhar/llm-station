#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class GoogleWebSearch:
    """Factory for Google Gemini search tool.

    Provides automatic search grounding and citations for Gemini 2.0+ models.
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
    """Factory for Google Search Retrieval tool with dynamic configuration."""

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
