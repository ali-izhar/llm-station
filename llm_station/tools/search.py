"""Web search tools for all providers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..types import ToolSpec

# Constants
MAX_ALLOWED_DOMAINS = 20


class OpenAISearch:
    """OpenAI web search tool with citations via Responses API."""

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
        """Validate and normalize domain list."""
        if len(domains) > MAX_ALLOWED_DOMAINS:
            raise ValueError(
                f"allowed_domains can contain at most {MAX_ALLOWED_DOMAINS} domains, got {len(domains)}"
            )

        normalized_domains = []
        url_pattern = re.compile(r"^https?://")

        for domain in domains:
            if not domain or not isinstance(domain, str):
                raise ValueError(f"Domain must be a non-empty string, got: {domain}")

            normalized_domain = url_pattern.sub("", domain.strip())
            if not normalized_domain:
                raise ValueError(
                    f"Domain cannot be empty after removing protocol: {domain}"
                )

            normalized_domains.append(normalized_domain)

        return normalized_domains

    def _validate_user_location(self, location: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user location parameters."""
        if not isinstance(location, dict):
            raise ValueError("user_location must be a dictionary")

        validated = {"type": "approximate"}

        if "country" in location:
            country = location["country"]
            if not isinstance(country, str) or len(country) != 2:
                raise ValueError(
                    f"country must be a two-letter ISO country code, got: {country}"
                )
            validated["country"] = country.upper()

        if "city" in location:
            city = location["city"]
            if not isinstance(city, str) or not city.strip():
                raise ValueError(f"city must be a non-empty string, got: {city}")
            validated["city"] = city.strip()

        if "region" in location:
            region = location["region"]
            if not isinstance(region, str) or not region.strip():
                raise ValueError(f"region must be a non-empty string, got: {region}")
            validated["region"] = region.strip()

        if "timezone" in location:
            timezone = location["timezone"]
            if not isinstance(timezone, str) or not timezone.strip():
                raise ValueError(
                    f"timezone must be a non-empty string, got: {timezone}"
                )
            if "/" not in timezone:
                raise ValueError(
                    f"timezone should be in IANA format (e.g., 'America/Chicago'), got: {timezone}"
                )
            validated["timezone"] = timezone.strip()

        return validated

    def spec(self) -> ToolSpec:
        """Generate ToolSpec for OpenAI web search tool."""
        provider_config: Dict[str, Any] = {}

        if self.allowed_domains:
            provider_config["filters"] = {"allowed_domains": self.allowed_domains}

        if self.user_location:
            provider_config["user_location"] = self.user_location

        return ToolSpec(
            name="web_search",
            description="OpenAI web search tool with up-to-date information and citations (Responses API)",
            input_schema={},
            requires_network=True,
            provider="openai",
            provider_type="web_search_preview" if self.preview else "web_search",
            provider_config=provider_config if provider_config else None,
        )


class AnthropicSearch:
    """Anthropic web search tool with automatic citations."""

    def __init__(
        self,
        *,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        user_location: Optional[Dict[str, Any]] = None,
        max_uses: Optional[int] = None,
        cache_control: Optional[Dict[str, Any]] = None,
    ) -> None:
        if allowed_domains is not None and blocked_domains is not None:
            raise ValueError(
                "allowed_domains and blocked_domains cannot be used together"
            )

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

        if self.user_location is not None:
            location = dict(self.user_location)
            location["type"] = "approximate"
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
            provider_config=cfg if cfg else None,
        )


class GoogleSearch:
    """Google Gemini search tool with automatic grounding."""

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


class HuggingFaceSearch:
    """HuggingFace web search tool with SerpAPI integration."""

    def __init__(
        self,
        *,
        use_web_search: bool = True,
        extract_full_text: bool = True,
        max_urls: int = 3,
        num_search_results: int = 5,
    ) -> None:
        if max_urls < 0:
            raise ValueError("max_urls must be non-negative")
        if num_search_results <= 0:
            raise ValueError("num_search_results must be greater than 0")

        self.use_web_search = use_web_search
        self.extract_full_text = extract_full_text
        self.max_urls = max_urls
        self.num_search_results = num_search_results

    def spec(self) -> ToolSpec:
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
