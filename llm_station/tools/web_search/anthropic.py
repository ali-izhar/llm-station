#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class AnthropicWebSearch:
    """Factory for Anthropic web search tool.

    Provides real-time web search with automatic citations and domain filtering.
    """

    def __init__(
        self,
        *,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
        user_location: Optional[Dict[str, Any]] = None,
        max_uses: Optional[int] = None,
        cache_control: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validate mutually exclusive domains
        if allowed_domains is not None and blocked_domains is not None:
            raise ValueError(
                "allowed_domains and blocked_domains cannot be used together"
            )

        # Validate max_uses
        if max_uses is not None and max_uses <= 0:
            raise ValueError("max_uses must be greater than 0")

        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.user_location = user_location
        self.max_uses = max_uses
        self.cache_control = cache_control

    def spec(self) -> ToolSpec:
        cfg: Dict[str, Any] = {}

        if self.allowed_domains is not None:
            cfg["allowed_domains"] = self.allowed_domains
        if self.blocked_domains is not None:
            cfg["blocked_domains"] = self.blocked_domains

        # User location with proper structure validation
        if self.user_location is not None:
            location = dict(self.user_location)
            # Ensure type is set to "approximate" as required by API
            location["type"] = "approximate"
            # Validate field constraints from documentation
            if "country" in location and len(location["country"]) != 2:
                raise ValueError("country must be a 2-character ISO country code")
            cfg["user_location"] = location

        if self.max_uses is not None:
            cfg["max_uses"] = self.max_uses
        if self.cache_control is not None:
            cfg["cache_control"] = self.cache_control

        return ToolSpec(
            name="web_search",
            description="Anthropic built-in web search tool (Messages API)",
            input_schema={},
            requires_network=True,
            provider="anthropic",
            provider_type="web_search_20250305",
            provider_config=cfg or None,
        )
