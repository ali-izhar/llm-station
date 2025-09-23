from __future__ import annotations

from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class OpenAIWebSearch:
    """Factory for OpenAI built-in web_search tool (Responses API).

    Produces a provider-native ToolSpec that instructs the OpenAI adapter to
    use the Responses API `web_search` (or `web_search_preview`) tool. This is
    not a local executable tool; no local execution occurs in the agent.

    Args:
        allowed_domains: List of up to 20 domains to limit search results to
        user_location: Geographic location for search refinement
        preview: Whether to use web_search_preview instead of web_search
    """

    def __init__(
        self,
        *,
        allowed_domains: Optional[list[str]] = None,
        user_location: Optional[Dict[str, Any]] = None,
        preview: bool = False,
    ) -> None:
        self.allowed_domains = allowed_domains
        self.user_location = user_location
        self.preview = preview

    def spec(self) -> ToolSpec:
        provider_config: Dict[str, Any] = {}

        # Domain filtering: allowed_domains array (up to 20)
        if self.allowed_domains:
            if len(self.allowed_domains) > 20:
                raise ValueError("allowed_domains can contain at most 20 domains")
            provider_config["filters"] = {"allowed_domains": self.allowed_domains}

        # User location for geographic search refinement
        if self.user_location:
            # Ensure proper structure with type: "approximate"
            location = {"type": "approximate", **self.user_location}
            provider_config["user_location"] = location

        return ToolSpec(
            name="web_search",
            description="OpenAI built-in web search tool (Responses API)",
            input_schema={},  # not used for provider-native tools
            requires_network=True,
            provider="openai",
            provider_type="web_search_preview" if self.preview else "web_search",
            provider_config=provider_config or None,
        )
