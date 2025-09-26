#!/usr/bin/env python3
"""
Clean Claude tests using smart tools and mocks.
Tests Claude-specific functionality without API calls.
"""

import pytest
from unittest.mock import patch
from llm_station import Agent
from llm_station.schemas.messages import ModelResponse


class TestClaudeSmartTools:
    """Test Claude with smart tools."""

    @patch("llm_station.models.anthropic.AnthropicProvider.generate")
    def test_claude_search_routing(self, mock_generate):
        """Test that Claude agent routes search to Anthropic."""
        mock_generate.return_value = ModelResponse(
            content="Search results", tool_calls=[]
        )

        agent = Agent(
            provider="anthropic",
            model="claude-sonnet-4",
            api_key="test",
            max_tokens=1024,
        )
        response = agent.generate("Search for information", tools=["search"])

        # Verify Anthropic tool was used
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0].provider == "anthropic"

    def test_claude_token_management(self):
        """Test Claude's token management features."""
        from llm_station.models.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test")

        # Test rate limiting logic
        assert provider._can_make_request(1000) == True

        # Add usage near limit
        provider._add_token_usage(9000, 500)
        assert provider._can_make_request(1000) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
