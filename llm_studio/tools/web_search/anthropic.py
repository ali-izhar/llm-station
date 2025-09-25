#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class AnthropicWebSearch:
    """Factory for Anthropic server web search tool (Messages API).

    Claude's built-in web search tool provides real-time web search capabilities with
    automatic result integration and optional domain filtering for security and relevance.

    Key Features:
    - **Real-Time Search**: Access current web information and recent developments
    - **Server-Side Execution**: Claude handles search and result processing automatically
    - **Domain Filtering**: Allow or block specific domains for content control
    - **Geographic Relevance**: Location-based search result refinement
    - **Usage Controls**: Limit tool usage to manage API costs
    - **Cache Optimization**: Configurable caching for improved performance

    How It Works:
    1. Include search queries in your prompt or ask for specific information
    2. Claude automatically performs web searches when beneficial
    3. Search results are integrated directly into the response content
    4. Tool execution metadata available in response.grounding_metadata

    Args:
        allowed_domains: List of domains to allow searching (mutually exclusive with blocked_domains)
        blocked_domains: List of domains to block searching (mutually exclusive with allowed_domains)
        user_location: Geographic location for search relevance (country, city, region)
        max_uses: Maximum number of searches per conversation (cost/usage control)
        cache_control: Cache control configuration for performance optimization

    Supported Models:
    - claude-opus-4-1-20250805 (latest)
    - claude-opus-4-20250514
    - claude-sonnet-4-20250514
    - claude-3-7-sonnet-20250219
    - claude-3-5-sonnet-latest (deprecated)
    - claude-3-5-haiku-latest

    Usage Examples:
        # Basic web search
        tool = AnthropicWebSearch()
        response = agent.generate(
            "How do I update a web app to TypeScript 5.5?",
            tools=[tool.spec()]
        )

        # Domain-filtered search for academic content
        academic_search = AnthropicWebSearch(
            allowed_domains=["arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ieee.org"],
            max_uses=5
        )

        # Location-specific search with timezone
        local_search = AnthropicWebSearch(
            user_location={
                "type": "approximate",
                "country": "US",
                "city": "San Francisco",
                "region": "California",
                "timezone": "America/Los_Angeles"
            }
        )

        # Security-focused search with blocked domains
        secure_search = AnthropicWebSearch(
            blocked_domains=["untrustedsource.com", "spam-site.net"],
            max_uses=3
        )

    Response Integration:
    - Search results appear directly in response.content
    - Citations automatically included with clickable URLs and titles
    - Tool metadata available in response.grounding_metadata["web_search"]
    - Usage statistics in response.grounding_metadata["usage"]

    Technical Details:
    - Uses `web_search_20250305` server tool version
    - Execution handled entirely by Anthropic's servers
    - Results processed and integrated before response delivery
    - No local tool calls generated (server-side execution)
    - Supports streaming with search events
    - Compatible with Messages Batches API
    - Supports prompt caching for multi-turn conversations

    Pricing & Usage:
    - $10 per 1,000 searches plus standard token costs
    - Each search counts as one use regardless of results returned
    - Search results count as input tokens in subsequent turns
    - Citations (url, title, cited_text) don't count toward token usage
    - Failed searches are not billed

    Error Handling:
    - max_uses_exceeded: Too many searches in one request
    - too_many_requests: Rate limit exceeded
    - invalid_input: Invalid search query
    - query_too_long: Query exceeds maximum length
    - unavailable: Internal error occurred

    Requirements:
    - Organization administrator must enable web search in Console
    - Compatible models: Opus 4.1+, Sonnet 4+, Sonnet 3.7+, Haiku 3.5+
    - anthropic-version: 2023-06-01 header required
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
