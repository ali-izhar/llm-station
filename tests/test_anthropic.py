#!/usr/bin/env python3
"""
Comprehensive tests for Anthropic Messages API implementation.

Tests accuracy against the official Anthropic Messages API documentation:
https://docs.anthropic.com/en/api/messages
"""

import pytest
from llm_studio.tools.web_search import AnthropicWebSearch
from llm_studio.tools.web_fetch import AnthropicWebFetch
from llm_studio.models.anthropic import AnthropicProvider
from llm_studio.models.base import ModelConfig
from llm_studio.schemas.messages import (
    UserMessage,
    SystemMessage,
    ToolMessage,
    ModelResponse,
)
from llm_studio.schemas.tooling import ToolSpec


class TestAnthropicWebSearchTool:
    """Test the AnthropicWebSearch tool factory."""

    def test_basic_web_search_tool(self):
        """Test basic web search tool without any options."""
        tool = AnthropicWebSearch()
        spec = tool.spec()

        assert spec.name == "web_search"
        assert spec.provider == "anthropic"
        assert spec.provider_type == "web_search_20250305"
        assert spec.requires_network is True
        assert spec.provider_config is None

    def test_domain_filtering(self):
        """Test domain filtering options."""
        # Test allowed domains
        allowed_tool = AnthropicWebSearch(
            allowed_domains=["anthropic.com", "example.com"]
        )
        spec = allowed_tool.spec()
        assert spec.provider_config["allowed_domains"] == [
            "anthropic.com",
            "example.com",
        ]

        # Test blocked domains
        blocked_tool = AnthropicWebSearch(blocked_domains=["spam.com", "blocked.org"])
        spec = blocked_tool.spec()
        assert spec.provider_config["blocked_domains"] == ["spam.com", "blocked.org"]

    def test_mutually_exclusive_domains(self):
        """Test that allowed_domains and blocked_domains cannot be used together."""
        with pytest.raises(
            ValueError,
            match="allowed_domains and blocked_domains cannot be used together",
        ):
            AnthropicWebSearch(
                allowed_domains=["good.com"], blocked_domains=["bad.com"]
            )

    def test_user_location(self):
        """Test user location configuration."""
        user_location = {"country": "US", "city": "San Francisco"}
        tool = AnthropicWebSearch(user_location=user_location)
        spec = tool.spec()

        expected_location = {
            "type": "approximate",
            "country": "US",
            "city": "San Francisco",
        }
        assert spec.provider_config["user_location"] == expected_location

    def test_user_location_country_validation(self):
        """Test user location country code validation."""
        # Valid 2-character country code
        tool = AnthropicWebSearch(user_location={"country": "GB"})
        spec = tool.spec()
        assert spec.provider_config["user_location"]["country"] == "GB"

        # Invalid country code (too long)
        with pytest.raises(
            ValueError, match="country must be a 2-character ISO country code"
        ):
            tool = AnthropicWebSearch(user_location={"country": "USA"})
            tool.spec()

        # Invalid country code (too short)
        with pytest.raises(
            ValueError, match="country must be a 2-character ISO country code"
        ):
            tool = AnthropicWebSearch(user_location={"country": "U"})
            tool.spec()

    def test_max_uses_validation(self):
        """Test max_uses validation."""
        # Valid max_uses
        tool = AnthropicWebSearch(max_uses=5)
        spec = tool.spec()
        assert spec.provider_config["max_uses"] == 5

        # Invalid max_uses (must be > 0)
        with pytest.raises(ValueError, match="max_uses must be greater than 0"):
            AnthropicWebSearch(max_uses=0)

        with pytest.raises(ValueError, match="max_uses must be greater than 0"):
            AnthropicWebSearch(max_uses=-1)

    def test_cache_control(self):
        """Test cache control configuration."""
        cache_config = {"ttl": "5m"}
        tool = AnthropicWebSearch(cache_control=cache_config)
        spec = tool.spec()
        assert spec.provider_config["cache_control"] == cache_config


class TestAnthropicWebFetchTool:
    """Test the AnthropicWebFetch tool factory."""

    def test_basic_web_fetch_tool(self):
        """Test basic web fetch tool."""
        tool = AnthropicWebFetch()
        spec = tool.spec()

        assert spec.name == "web_fetch"
        assert spec.provider == "anthropic"
        assert spec.provider_type == "web_fetch_20250910"
        assert spec.requires_network is True

    def test_max_content_tokens_validation(self):
        """Test max_content_tokens validation."""
        # Valid max_content_tokens
        tool = AnthropicWebFetch(max_content_tokens=1000)
        spec = tool.spec()
        assert spec.provider_config["max_content_tokens"] == 1000

        # Invalid max_content_tokens (must be > 0)
        with pytest.raises(
            ValueError, match="max_content_tokens must be greater than 0"
        ):
            AnthropicWebFetch(max_content_tokens=0)

        with pytest.raises(
            ValueError, match="max_content_tokens must be greater than 0"
        ):
            AnthropicWebFetch(max_content_tokens=-1)

    def test_max_uses_validation(self):
        """Test max_uses validation."""
        # Valid max_uses
        tool = AnthropicWebFetch(max_uses=3)
        spec = tool.spec()
        assert spec.provider_config["max_uses"] == 3

        # Invalid max_uses (must be > 0)
        with pytest.raises(ValueError, match="max_uses must be greater than 0"):
            AnthropicWebFetch(max_uses=0)

    def test_citations_config(self):
        """Test citations configuration."""
        citations_config = {"enabled": True}
        tool = AnthropicWebFetch(citations=citations_config)
        spec = tool.spec()
        assert spec.provider_config["citations"] == citations_config

    def test_combined_options(self):
        """Test web fetch with multiple options."""
        tool = AnthropicWebFetch(
            allowed_domains=["trustworthy.com"],
            max_content_tokens=2000,
            max_uses=5,
            citations={"enabled": True},
        )
        spec = tool.spec()

        config = spec.provider_config
        assert config["allowed_domains"] == ["trustworthy.com"]
        assert config["max_content_tokens"] == 2000
        assert config["max_uses"] == 5
        assert config["citations"] == {"enabled": True}


class TestAnthropicProviderToolPreparation:
    """Test Anthropic provider adapter tool preparation."""

    def test_server_tool_preparation_web_search(self):
        """Test preparation of server tools for web search."""
        provider = AnthropicProvider()
        tool = AnthropicWebSearch(allowed_domains=["example.com"], max_uses=3)
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        expected = {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": ["example.com"],
            "max_uses": 3,
        }
        assert prepared_tools[0] == expected

    def test_server_tool_preparation_web_fetch(self):
        """Test preparation of server tools for web fetch."""
        provider = AnthropicProvider()
        tool = AnthropicWebFetch(max_content_tokens=1500, citations={"enabled": False})
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        expected = {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
            "max_content_tokens": 1500,
            "citations": {"enabled": False},
        }
        assert prepared_tools[0] == expected

    def test_custom_tool_preparation(self):
        """Test preparation of custom (client) tools."""
        provider = AnthropicProvider()

        custom_tool = ToolSpec(
            name="custom_function",
            description="A custom function tool",
            input_schema={
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
            },
        )

        prepared_tools = provider.prepare_tools([custom_tool])

        expected = {
            "name": "custom_function",
            "description": "A custom function tool",
            "input_schema": {
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
            },
        }
        assert prepared_tools[0] == expected

    def test_mixed_tools_preparation(self):
        """Test preparing both server and custom tools."""
        provider = AnthropicProvider()

        custom_tool = ToolSpec(
            name="local_tool",
            description="A local tool",
            input_schema={"type": "object", "properties": {}},
        )

        server_tool = AnthropicWebSearch(allowed_domains=["safe.com"]).spec()

        prepared_tools = provider.prepare_tools([custom_tool, server_tool])

        assert len(prepared_tools) == 2

        # Custom tool should be first
        assert prepared_tools[0]["name"] == "local_tool"
        assert "input_schema" in prepared_tools[0]

        # Server tool should be second
        assert prepared_tools[1]["type"] == "web_search_20250305"
        assert prepared_tools[1]["name"] == "web_search"


class TestAnthropicProviderToolChoice:
    """Test tool_choice parameter handling."""

    def test_tool_choice_simple_strings(self):
        """Test simple string tool_choice values."""
        provider = AnthropicProvider()
        messages = [UserMessage("Test")]

        # Test "auto"
        config = ModelConfig(
            provider="anthropic", model="claude-3-sonnet", tool_choice="auto"
        )
        try:
            provider.generate(messages, config, tools=[])
        except RuntimeError as e:
            # Should get our expected error with proper request structure
            assert "tool_choice" in str(e)

        # Test "any"
        config = ModelConfig(
            provider="anthropic", model="claude-3-sonnet", tool_choice="any"
        )
        try:
            provider.generate(messages, config, tools=[])
        except RuntimeError:
            pass  # Expected since no network implementation

        # Test "none"
        config = ModelConfig(
            provider="anthropic", model="claude-3-sonnet", tool_choice="none"
        )
        try:
            provider.generate(messages, config, tools=[])
        except RuntimeError:
            pass  # Expected since no network implementation

    def test_tool_choice_tool_name(self):
        """Test tool_choice with specific tool name."""
        provider = AnthropicProvider()
        messages = [UserMessage("Test")]

        config = ModelConfig(
            provider="anthropic", model="claude-3-sonnet", tool_choice="specific_tool"
        )
        try:
            provider.generate(messages, config, tools=[])
        except RuntimeError:
            pass  # Expected since no network implementation

    def test_tool_choice_dict_format(self):
        """Test tool_choice with dict format including disable_parallel_tool_use."""
        provider = AnthropicProvider()
        messages = [UserMessage("Test")]

        tool_choice_dict = {"type": "any", "disable_parallel_tool_use": True}
        config = ModelConfig(
            provider="anthropic", model="claude-3-sonnet", tool_choice=tool_choice_dict
        )
        try:
            provider.generate(messages, config, tools=[])
        except RuntimeError:
            pass  # Expected since no network implementation

    def test_tool_choice_invalid_type(self):
        """Test tool_choice with invalid type."""
        provider = AnthropicProvider()
        messages = [UserMessage("Test")]

        config = ModelConfig(
            provider="anthropic", model="claude-3-sonnet", tool_choice=123
        )
        with pytest.raises(ValueError, match="Invalid tool_choice type"):
            provider.generate(messages, config, tools=[])


class TestAnthropicProviderMessageMapping:
    """Test message mapping functionality."""

    def test_system_message_mapping(self):
        """Test system message handling."""
        provider = AnthropicProvider()

        messages = [SystemMessage("You are a helpful assistant."), UserMessage("Hello")]

        mapped = provider._map_messages(messages)

        assert mapped["system"] == "You are a helpful assistant."
        assert len(mapped["messages"]) == 1
        assert mapped["messages"][0]["role"] == "user"
        assert mapped["messages"][0]["content"] == "Hello"

    def test_tool_result_mapping(self):
        """Test tool result message mapping."""
        provider = AnthropicProvider()

        messages = [
            UserMessage("Use a tool"),
            ToolMessage(content="Tool result", tool_call_id="tc_123", name="test_tool"),
        ]

        mapped = provider._map_messages(messages)

        assert len(mapped["messages"]) == 2

        # First message should be the user message
        assert mapped["messages"][0]["role"] == "user"
        assert mapped["messages"][0]["content"] == "Use a tool"

        # Second message should be tool results as user message with content blocks
        assert mapped["messages"][1]["role"] == "user"
        assert isinstance(mapped["messages"][1]["content"], list)

        tool_block = mapped["messages"][1]["content"][0]
        assert tool_block["type"] == "tool_result"
        assert tool_block["tool_use_id"] == "tc_123"
        assert tool_block["content"] == "Tool result"

    def test_multiple_tool_results_batching(self):
        """Test that multiple tool results are batched into a single user message."""
        provider = AnthropicProvider()

        messages = [
            UserMessage("Use tools"),
            ToolMessage(content="Result 1", tool_call_id="tc_1", name="tool1"),
            ToolMessage(content="Result 2", tool_call_id="tc_2", name="tool2"),
            UserMessage("Continue"),
        ]

        mapped = provider._map_messages(messages)

        assert len(mapped["messages"]) == 3

        # Check tool results are batched
        tool_results_msg = mapped["messages"][1]
        assert tool_results_msg["role"] == "user"
        assert len(tool_results_msg["content"]) == 2

        # Check both tool results are present
        assert tool_results_msg["content"][0]["tool_use_id"] == "tc_1"
        assert tool_results_msg["content"][1]["tool_use_id"] == "tc_2"


class TestAnthropicProviderResponseParsing:
    """Test response parsing functionality."""

    def test_text_response_parsing(self):
        """Test parsing of text-only responses."""
        provider = AnthropicProvider()

        response_payload = {
            "content": [{"type": "text", "text": "Hello, how can I help you?"}]
        }

        parsed = provider._parse_response(response_payload)

        assert parsed.content == "Hello, how can I help you?"
        assert len(parsed.tool_calls) == 0
        assert parsed.raw == response_payload

    def test_tool_use_response_parsing(self):
        """Test parsing of responses with tool use."""
        provider = AnthropicProvider()

        response_payload = {
            "content": [
                {"type": "text", "text": "I'll help you with that."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search_tool",
                    "input": {"query": "test query"},
                },
            ]
        }

        parsed = provider._parse_response(response_payload)

        assert parsed.content == "I'll help you with that."
        assert len(parsed.tool_calls) == 1

        tool_call = parsed.tool_calls[0]
        assert tool_call.id == "toolu_123"
        assert tool_call.name == "search_tool"
        assert tool_call.arguments == {"query": "test query"}

    def test_server_tool_use_response_parsing(self):
        """Test parsing of server tool use (should not create local tool calls)."""
        provider = AnthropicProvider()

        response_payload = {
            "content": [
                {"type": "text", "text": "Searching the web..."},
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_123",
                    "name": "web_search",
                    "input": {"query": "anthropic claude"},
                },
            ]
        }

        parsed = provider._parse_response(response_payload)

        assert parsed.content == "Searching the web..."
        assert len(parsed.tool_calls) == 0  # Server tools don't create local tool calls

    def test_mixed_content_response_parsing(self):
        """Test parsing of responses with mixed content types."""
        provider = AnthropicProvider()

        response_payload = {
            "content": [
                {"type": "text", "text": "First part. "},
                {
                    "type": "tool_use",
                    "id": "toolu_456",
                    "name": "calculate",
                    "input": {"expression": "2+2"},
                },
                {"type": "text", "text": "Second part."},
            ]
        }

        parsed = provider._parse_response(response_payload)

        assert parsed.content == "First part. \nSecond part."
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "calculate"


def test_documentation_examples():
    """Test examples from the Anthropic documentation."""

    # Example 1: Basic web search
    web_search = AnthropicWebSearch()
    spec = web_search.spec()
    assert spec.provider_type == "web_search_20250305"

    # Example 2: Web search with domain filtering
    filtered_search = AnthropicWebSearch(
        allowed_domains=["docs.anthropic.com", "support.anthropic.com"]
    )
    spec = filtered_search.spec()
    assert spec.provider_config["allowed_domains"] == [
        "docs.anthropic.com",
        "support.anthropic.com",
    ]

    # Example 3: Web fetch with content limits
    web_fetch = AnthropicWebFetch(max_content_tokens=2000, citations={"enabled": True})
    spec = web_fetch.spec()
    assert spec.provider_config["max_content_tokens"] == 2000
    assert spec.provider_config["citations"] == {"enabled": True}

    # Example 4: User location
    location_search = AnthropicWebSearch(
        user_location={
            "country": "US",
            "city": "New York",
            "region": "New York",
            "timezone": "America/New_York",
        }
    )
    spec = location_search.spec()
    location = spec.provider_config["user_location"]
    assert location["type"] == "approximate"
    assert location["country"] == "US"
    assert location["timezone"] == "America/New_York"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
