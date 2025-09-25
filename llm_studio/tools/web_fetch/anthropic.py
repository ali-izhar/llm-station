from __future__ import annotations

from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class AnthropicWebFetch:
    """Factory for Anthropic server web fetch tool (Messages API).

    Claude's built-in web fetch tool enables direct retrieval and processing of web content.
    The tool executes server-side and integrates results directly into the response content.

    Key Features:
    - **Server-Side Execution**: No local processing, Claude handles web requests
    - **Content Integration**: Fetched content seamlessly integrated into responses
    - **Domain Control**: Allow/block specific domains for security
    - **Content Limits**: Control maximum tokens to manage context size
    - **Usage Limits**: Prevent excessive API usage with max_uses parameter
    - **Citation Support**: Optional citation extraction from fetched content

    How It Works:
    1. Provide URLs in your prompt or specify fetch requirements
    2. Claude automatically fetches and processes the content
    3. Content is integrated into the response with optional citations
    4. Tool execution metadata available in response.grounding_metadata

    Args:
        allowed_domains: List of domains to allow fetching from (security control)
        blocked_domains: List of domains to block fetching from (security control)
        citations: Citations configuration for fetched documents (disabled by default)
        max_content_tokens: Maximum tokens for web page content (prevents context overflow)
        max_uses: Maximum tool usage per conversation (cost control)
        cache_control: Cache control configuration for performance

    Usage Examples:
        # Basic web content fetching
        tool = AnthropicWebFetch()
        response = agent.generate(
            "Fetch and analyze the content from https://example.com/article",
            tools=[tool.spec()]
        )

        # Domain-restricted fetching for security
        secure_fetch = AnthropicWebFetch(
            allowed_domains=["wikipedia.org", "arxiv.org", "github.com"],
            max_content_tokens=5000
        )

        # Content analysis with citations
        citation_fetch = AnthropicWebFetch(
            citations={"enabled": True},
            max_uses=3
        )

    Response Integration:
    - Fetched content appears directly in response.content
    - Tool metadata available in response.grounding_metadata["web_fetch"]
    - Usage statistics in response.grounding_metadata["usage"]

    Technical Details:
    - Uses `web_fetch_20250910` server tool version
    - Execution handled entirely by Anthropic's servers
    - Content processed and integrated before response delivery
    - No local tool calls generated (server-side execution)
    """

    def __init__(
        self,
        *,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
        citations: Optional[Dict[str, Any]] = None,
        max_content_tokens: Optional[int] = None,
        max_uses: Optional[int] = None,
        cache_control: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validate max_content_tokens
        if max_content_tokens is not None and max_content_tokens <= 0:
            raise ValueError("max_content_tokens must be greater than 0")

        # Validate max_uses
        if max_uses is not None and max_uses <= 0:
            raise ValueError("max_uses must be greater than 0")

        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.citations = citations
        self.max_content_tokens = max_content_tokens
        self.max_uses = max_uses
        self.cache_control = cache_control

    def spec(self) -> ToolSpec:
        cfg: Dict[str, Any] = {}
        if self.allowed_domains is not None:
            cfg["allowed_domains"] = self.allowed_domains
        if self.blocked_domains is not None:
            cfg["blocked_domains"] = self.blocked_domains
        if self.citations is not None:
            cfg["citations"] = self.citations
        if self.max_content_tokens is not None:
            cfg["max_content_tokens"] = self.max_content_tokens
        if self.max_uses is not None:
            cfg["max_uses"] = self.max_uses
        if self.cache_control is not None:
            cfg["cache_control"] = self.cache_control
        return ToolSpec(
            name="web_fetch",
            description="Anthropic built-in web fetch tool (Messages API)",
            input_schema={},
            requires_network=True,
            provider="anthropic",
            provider_type="web_fetch",
            provider_config=cfg or None,
        )
