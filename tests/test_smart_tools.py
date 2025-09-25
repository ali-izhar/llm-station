#!/usr/bin/env python3
"""
Unit tests for the smart tools system using mocks.
Tests the new provider-agnostic tool interface without API calls.
"""

import pytest
from unittest.mock import Mock, patch

from llm_studio import Agent, get_available_tools, get_tool_info, recommend_tools
from llm_studio.tools.registry import get_tool_spec, _get_smart_tool_spec
from llm_studio.schemas.messages import AssistantMessage, ModelResponse, ToolCall


class TestSmartToolsRegistry:
    """Test the smart tools registry functionality."""

    def test_get_available_tools(self):
        """Test that all smart tools are available."""
        tools = get_available_tools()

        # Check primary smart tools exist
        primary_tools = ["search", "code", "image", "json", "fetch", "url"]
        for tool in primary_tools:
            assert tool in tools
            assert tools[tool] == "smart"

        # Check aliases work
        aliases = ["websearch", "python", "draw", "execute"]
        for alias in aliases:
            assert alias in tools
            assert tools[alias] == "smart"

    def test_tool_aliases_resolve_correctly(self):
        """Test that tool aliases resolve to the correct primary tools."""
        alias_mappings = {
            "websearch": "search",
            "web_search": "search",
            "python": "code",
            "execute": "code",
            "draw": "image",
            "format_json": "json",
        }

        for alias, primary in alias_mappings.items():
            alias_spec = get_tool_spec(alias)
            primary_spec = get_tool_spec(primary)

            # Should resolve to same provider tool
            assert alias_spec.provider == primary_spec.provider
            assert alias_spec.name == primary_spec.name

    def test_provider_preference_routing(self):
        """Test that provider preferences work correctly."""
        # Test search tool with different preferences
        google_spec = get_tool_spec("search", provider_preference="google")
        assert google_spec.provider == "google"

        anthropic_spec = get_tool_spec("search", provider_preference="anthropic")
        assert anthropic_spec.provider == "anthropic"

        openai_spec = get_tool_spec("search", provider_preference="openai")
        assert openai_spec.provider == "openai"

    def test_provider_exclusion(self):
        """Test that provider exclusions work correctly."""
        # Exclude Google, should get Anthropic (next highest score)
        spec = get_tool_spec("search", exclude_providers=["google"])
        assert spec.provider == "anthropic"

        # Exclude Google and Anthropic, should get OpenAI
        spec = get_tool_spec("search", exclude_providers=["google", "anthropic"])
        assert spec.provider == "openai"

    def test_tool_info_detailed(self):
        """Test that tool info provides comprehensive details."""
        info = get_tool_info("search")

        assert info["name"] == "search"
        assert info["type"] == "smart"
        assert "providers" in info
        assert "aliases" in info

        # Should have 3 providers for search
        assert len(info["providers"]) == 3

        # Should be ordered by score
        scores = [p["score"] for p in info["providers"]]
        assert scores == sorted(scores, reverse=True)

    def test_tool_recommendations(self):
        """Test tool recommendation engine."""
        # Test search recommendation
        recs = recommend_tools("Research AI developments")
        assert "search" in recs

        # Test code recommendation
        recs = recommend_tools("Calculate statistics")
        assert "code" in recs

        # Test image recommendation
        recs = recommend_tools("Create an image")
        assert "image" in recs

        # Test JSON recommendation
        recs = recommend_tools("Format as JSON")
        assert "json" in recs

    def test_unknown_tool_error(self):
        """Test error handling for unknown tools."""
        with pytest.raises(KeyError) as exc_info:
            get_tool_spec("unknown_tool")

        assert "Unknown tool: unknown_tool" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)


class TestSmartToolsWithMockAgent:
    """Test smart tools integration with mock agent."""

    def test_basic_smart_tools(self):
        """Test basic smart tools with mock agent."""
        agent = Agent(provider="mock", model="test")

        # Test each primary smart tool
        tools_to_test = ["search", "code", "image", "json"]

        for tool in tools_to_test:
            response = agent.generate(f"Test {tool} tool", tools=[tool])
            assert isinstance(response, AssistantMessage)
            assert len(response.content) > 0

    def test_tool_aliases_with_agent(self):
        """Test that tool aliases work with agent."""
        agent = Agent(provider="mock", model="test")

        # Test aliases
        alias_tests = [("websearch", "search"), ("python", "code"), ("draw", "image")]

        for alias, primary in alias_tests:
            alias_response = agent.generate("Test", tools=[alias])
            primary_response = agent.generate("Test", tools=[primary])

            # Should produce similar responses (both work)
            assert isinstance(alias_response, AssistantMessage)
            assert isinstance(primary_response, AssistantMessage)

    def test_multiple_tools(self):
        """Test using multiple smart tools together."""
        agent = Agent(provider="mock", model="test")

        response = agent.generate(
            "Test multiple tools", tools=["search", "code", "json"]
        )

        assert isinstance(response, AssistantMessage)
        assert len(response.content) > 0

    def test_tool_configuration_dict(self):
        """Test tool configuration with dict format."""
        agent = Agent(provider="mock", model="test")

        # Test provider preference via dict
        response = agent.generate(
            "Test search", tools=[{"name": "search", "provider_preference": "google"}]
        )

        assert isinstance(response, AssistantMessage)


class TestProviderSpecificRouting:
    """Test that smart tools route correctly for different providers."""

    def test_openai_agent_routing(self):
        """Test smart tools routing for OpenAI agent."""
        with patch("llm_studio.models.openai.OpenAIProvider.generate") as mock_generate:
            mock_generate.return_value = ModelResponse(
                content="Test response", tool_calls=[]
            )

            agent = Agent(provider="openai", model="gpt-4o-mini", api_key="test")
            response = agent.generate("Test search", tools=["search"])

            # Should have been called with OpenAI tools
            args, kwargs = mock_generate.call_args
            tools = kwargs.get("tools", [])
            if tools:
                assert any(t.provider == "openai" for t in tools)

    def test_google_agent_routing(self):
        """Test smart tools routing for Google agent."""
        with patch("llm_studio.models.google.GoogleProvider.generate") as mock_generate:
            mock_generate.return_value = ModelResponse(
                content="Test response", tool_calls=[]
            )

            agent = Agent(provider="google", model="gemini-2.5-flash", api_key="test")
            response = agent.generate("Test search", tools=["search"])

            # Should have been called with Google tools
            args, kwargs = mock_generate.call_args
            tools = kwargs.get("tools", [])
            if tools:
                assert any(t.provider == "google" for t in tools)

    def test_anthropic_agent_routing(self):
        """Test smart tools routing for Anthropic agent."""
        with patch(
            "llm_studio.models.anthropic.AnthropicProvider.generate"
        ) as mock_generate:
            mock_generate.return_value = ModelResponse(
                content="Test response", tool_calls=[]
            )

            agent = Agent(provider="anthropic", model="claude-sonnet-4", api_key="test")
            response = agent.generate("Test search", tools=["search"])

            # Should have been called with Anthropic tools
            args, kwargs = mock_generate.call_args
            tools = kwargs.get("tools", [])
            if tools:
                assert any(t.provider == "anthropic" for t in tools)


class TestLocalToolsIntegration:
    """Test local tools integration with smart system."""

    def test_json_tool_execution(self):
        """Test JSON tool executes locally."""
        agent = Agent(provider="mock", model="test")

        # Mock will make a tool call that should be executed locally
        with patch.object(agent, "_execute_tool") as mock_execute:
            mock_execute.return_value = Mock(content='{"test": "result"}')

            # Simulate model making a tool call
            with patch("llm_studio.models.mock.MockProvider.generate") as mock_generate:
                mock_generate.return_value = ModelResponse(
                    content="Test response",
                    tool_calls=[
                        ToolCall(
                            id="call_1", name="json_format", arguments={"data": "test"}
                        )
                    ],
                )

                response = agent.generate("Format as JSON", tools=["json"])

                # Local tool should have been executed
                mock_execute.assert_called_once()

    def test_fetch_tool_spec(self):
        """Test fetch tool specification."""
        spec = get_tool_spec("fetch")

        assert spec.name == "fetch_url"
        assert spec.provider is None  # Local tool
        assert spec.requires_network == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
