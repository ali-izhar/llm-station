#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

from ...schemas.tooling import ToolSpec


class OpenAIWebSearch:
    """Factory for OpenAI web search tool.

    Provides real-time web search with citations via Responses API.
    Supports domain filtering and geographic refinement.
    """

    def __init__(
        self,
        *,
        allowed_domains: Optional[List[str]] = None,
        user_location: Optional[Dict[str, Any]] = None,
        preview: bool = False,
    ) -> None:
        self.allowed_domains = (
            self._validate_domains(allowed_domains) if allowed_domains else None
        )
        self.user_location = (
            self._validate_user_location(user_location) if user_location else None
        )
        self.preview = preview

    def _validate_domains(self, domains: List[str]) -> List[str]:
        """Validate and normalize domain list according to OpenAI requirements."""
        if len(domains) > 20:
            raise ValueError(
                f"allowed_domains can contain at most 20 domains, got {len(domains)}"
            )

        normalized_domains = []
        url_pattern = re.compile(r"^https?://")

        for domain in domains:
            if not domain or not isinstance(domain, str):
                raise ValueError(f"Domain must be a non-empty string, got: {domain}")

            # Remove http/https prefix as per OpenAI docs
            normalized_domain = url_pattern.sub("", domain.strip())
            if not normalized_domain:
                raise ValueError(
                    f"Domain cannot be empty after removing protocol: {domain}"
                )

            normalized_domains.append(normalized_domain)

        return normalized_domains

    def _validate_user_location(self, location: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user location parameters according to OpenAI requirements."""
        if not isinstance(location, dict):
            raise ValueError("user_location must be a dictionary")

        validated = {"type": "approximate"}  # Required by OpenAI

        # Validate country (two-letter ISO code)
        if "country" in location:
            country = location["country"]
            if not isinstance(country, str) or len(country) != 2:
                raise ValueError(
                    f"country must be a two-letter ISO country code (e.g., 'US'), got: {country}"
                )
            validated["country"] = country.upper()

        # Validate city (free text)
        if "city" in location:
            city = location["city"]
            if not isinstance(city, str) or not city.strip():
                raise ValueError(f"city must be a non-empty string, got: {city}")
            validated["city"] = city.strip()

        # Validate region (free text)
        if "region" in location:
            region = location["region"]
            if not isinstance(region, str) or not region.strip():
                raise ValueError(f"region must be a non-empty string, got: {region}")
            validated["region"] = region.strip()

        # Validate timezone (IANA format)
        if "timezone" in location:
            timezone = location["timezone"]
            if not isinstance(timezone, str) or not timezone.strip():
                raise ValueError(
                    f"timezone must be a non-empty string, got: {timezone}"
                )
            # Basic IANA timezone format validation (should contain '/')
            if "/" not in timezone:
                raise ValueError(
                    f"timezone should be in IANA format (e.g., 'America/Chicago'), got: {timezone}"
                )
            validated["timezone"] = timezone.strip()

        return validated

    def spec(self) -> ToolSpec:
        """Generate ToolSpec for OpenAI web search tool."""
        provider_config: Dict[str, Any] = {}

        # Domain filtering for allowed domains
        if self.allowed_domains:
            provider_config["filters"] = {"allowed_domains": self.allowed_domains}

        # User location for geographic search refinement
        if self.user_location:
            provider_config["user_location"] = self.user_location

        return ToolSpec(
            name="web_search",
            description="OpenAI web search tool with up-to-date information and citations (Responses API)",
            input_schema={},  # Not used for provider-native tools
            requires_network=True,
            provider="openai",
            provider_type="web_search_preview" if self.preview else "web_search",
            provider_config=provider_config or None,
        )
