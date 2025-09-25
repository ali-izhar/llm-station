from __future__ import annotations

from typing import Any, Dict, Optional

from ...schemas.tooling import ToolSpec


class AnthropicWebFetch:
    """Factory for Anthropic web fetch tool.

    Provides server-side web content retrieval with domain filtering and usage controls.
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
