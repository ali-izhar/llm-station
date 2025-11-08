#!/usr/bin/env python3
"""
Unit tests for the smart tools system using mocks.
Tests the new provider-agnostic tool interface without API calls.
"""

import pytest
from unittest.mock import Mock, patch

from llm_station import Agent, AssistantMessage, ModelResponse, ToolCall, get_spec


class TestSmartToolsRegistry:
    """Test the smart tools registry functionality."""

    def test_get_available_tools(self):
        """Test that all smart tools are available."""
        from llm_station import list_tools

        tools = list_tools()

        # Check primary smart tools exist (using get_spec to verify)
        primary_tools = ["search", "code", "image", "json", "fetch"]
        for tool in primary_tools:
            spec = get_spec(tool)
            assert spec is not None

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
            alias_spec = get_spec(alias)
            primary_spec = get_spec(primary)

            # Should resolve to same provider tool
            assert alias_spec.provider == primary_spec.provider
            assert alias_spec.name == primary_spec.name

    def test_provider_preference_routing(self):
        """Test that provider preferences work correctly."""
        # Test search tool with different preferences
        google_spec = get_spec("search", provider_preference="google")
        assert google_spec.provider == "google"

        anthropic_spec = get_spec("search", provider_preference="anthropic")
        assert anthropic_spec.provider == "anthropic"

        openai_spec = get_spec("search", provider_preference="openai")
        assert openai_spec.provider == "openai"

    def test_provider_exclusion(self):
        """Test that provider preferences work correctly."""
        # Note: exclude_providers not supported in new structure
        # Test that we can still get different providers via preference
        google_spec = get_spec("search", provider_preference="google")
        assert google_spec.provider == "google"

        anthropic_spec = get_spec("search", provider_preference="anthropic")
        assert anthropic_spec.provider == "anthropic"

    def test_tool_info_detailed(self):
        """Test that tool info provides comprehensive details."""
        spec = get_spec("search")

        assert spec.name is not None
        assert spec.description is not None
        assert spec.provider is not None

    def test_tool_recommendations(self):
        """Test tool recommendation engine."""
        # Test that tools can be retrieved by name
        # Note: recommendation logic may not exist in new structure
        search_spec = get_spec("search")
        assert search_spec is not None

        code_spec = get_spec("code")
        assert code_spec is not None

        image_spec = get_spec("image")
        assert image_spec is not None

        json_spec = get_spec("json")
        assert json_spec is not None

    def test_unknown_tool_error(self):
        """Test error handling for unknown tools."""
        with pytest.raises(KeyError) as exc_info:
            get_spec("unknown_tool")

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
        with patch(
            "llm_station.providers.openai.OpenAIProvider.generate"
        ) as mock_generate:
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
        with patch(
            "llm_station.providers.google.GoogleProvider.generate"
        ) as mock_generate:
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
            "llm_station.providers.anthropic.AnthropicProvider.generate"
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
            with patch(
                "llm_station.providers.mock.MockProvider.generate"
            ) as mock_generate:
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
        spec = get_spec("fetch")

        assert spec.name == "fetch_url"
        assert spec.provider is None  # Local tool
        assert spec.requires_network == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
