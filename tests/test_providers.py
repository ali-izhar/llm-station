#!/usr/bin/env python3
"""
Unit tests for provider adapters using mocks.
Tests all three providers without making actual API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from llm_station import Agent
from llm_station.models.openai import OpenAIProvider
from llm_station.models.anthropic import AnthropicProvider
from llm_station.models.google import GoogleProvider
from llm_station.models.mock import MockProvider
from llm_station.models.base import ModelConfig
from llm_station.schemas.messages import (
    UserMessage,
    SystemMessage,
    ModelResponse,
    ToolCall,
    AssistantMessage,
)
from llm_station.schemas.tooling import ToolSpec
from llm_station.tools.registry import get_tool_spec


class TestOpenAIProvider:
    """Test OpenAI provider functionality."""

    def test_provider_creation(self):
        """Test OpenAI provider can be created."""
        provider = OpenAIProvider(api_key="test")
        assert provider.name == "openai"
        assert provider.supports_tools() == True

    def test_api_type_detection(self):
        """Test API type detection logic."""
        provider = OpenAIProvider(api_key="test")
        config = ModelConfig(provider="openai", model="gpt-4o-mini")

        # No tools = chat completions
        api_type = provider.get_api_type(config, None)
        assert api_type == "chat_completions"

        # OpenAI tools = responses API
        openai_tool = ToolSpec(
            name="web_search",
            description="Test",
            input_schema={},
            provider="openai",
            provider_type="web_search",
        )
        api_type = provider.get_api_type(config, [openai_tool])
        assert api_type == "responses_api"

    def test_tool_preparation(self):
        """Test tool preparation for OpenAI format."""
        provider = OpenAIProvider(api_key="test")

        # Test OpenAI web search tool
        openai_tool = ToolSpec(
            name="web_search",
            description="Test search",
            input_schema={},
            provider="openai",
            provider_type="web_search",
        )

        prepared = provider.prepare_tools([openai_tool])
        assert len(prepared) == 1
        assert prepared[0]["type"] == "web_search"

        # Test local function tool
        local_tool = ToolSpec(
            name="json_format",
            description="Test JSON",
            input_schema={"type": "object", "properties": {"data": {"type": "string"}}},
        )

        prepared = provider.prepare_tools([local_tool])
        assert len(prepared) == 1
        assert prepared[0]["type"] == "function"
        assert prepared[0]["function"]["name"] == "json_format"


class TestAnthropicProvider:
    """Test Anthropic provider functionality."""

    def test_provider_creation(self):
        """Test Anthropic provider can be created."""
        provider = AnthropicProvider(api_key="test")
        assert provider.name == "anthropic"
        assert provider.supports_tools() == True

    def test_token_management(self):
        """Test token usage tracking."""
        provider = AnthropicProvider(api_key="test")

        # Should be able to make request initially
        assert provider._can_make_request(1000) == True

        # Add usage
        provider._add_token_usage(5000, 1000)

        # Should still be under limit
        assert provider._can_make_request(1000) == True

        # Add more usage to approach limit
        provider._add_token_usage(4000, 1000)

        # Should be near limit
        assert provider._can_make_request(1000) == False

    def test_tool_preparation(self):
        """Test tool preparation for Anthropic format."""
        provider = AnthropicProvider(api_key="test")

        # Test Anthropic web search tool
        anthropic_tool = ToolSpec(
            name="web_search",
            description="Test search",
            input_schema={},
            provider="anthropic",
            provider_type="web_search_20250305",
        )

        prepared = provider.prepare_tools([anthropic_tool])
        assert len(prepared) == 1
        assert prepared[0]["type"] == "web_search_20250305"
        assert prepared[0]["name"] == "web_search"


class TestGoogleProvider:
    """Test Google provider functionality."""

    def test_provider_creation(self):
        """Test Google provider can be created."""
        provider = GoogleProvider(api_key="test")
        assert provider.name == "google"
        assert provider.supports_tools() == True

    def test_tool_preparation(self):
        """Test tool preparation for Google format."""
        provider = GoogleProvider(api_key="test")

        # Test Google search tool
        google_tool = ToolSpec(
            name="google_search",
            description="Test search",
            input_schema={},
            provider="google",
            provider_type="google_search",
        )

        prepared = provider.prepare_tools([google_tool])
        assert len(prepared) == 1
        assert prepared[0] == {"google_search": {}}

        # Test local function tool
        local_tool = ToolSpec(
            name="json_format",
            description="Test JSON",
            input_schema={"type": "object", "properties": {"data": {"type": "string"}}},
        )

        prepared = provider.prepare_tools([local_tool])
        assert len(prepared) == 1
        assert "function_declarations" in prepared[0]


class TestMockProvider:
    """Test mock provider functionality."""

    def test_mock_provider_basic(self):
        """Test mock provider basic functionality."""
        provider = MockProvider(api_key="test")
        config = ModelConfig(provider="mock", model="test")

        messages = [UserMessage("Hello world")]
        response = provider.generate(messages, config)

        assert isinstance(response, ModelResponse)
        assert response.content == "Hello world"
        assert len(response.tool_calls) == 0

    def test_mock_provider_tool_calls(self):
        """Test mock provider tool call simulation."""
        provider = MockProvider(api_key="test")
        config = ModelConfig(provider="mock", model="test")

        # Message with tool call pattern
        messages = [UserMessage('call json_format({"data": "test"})')]
        response = provider.generate(messages, config)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "json_format"
        assert response.tool_calls[0].arguments == {"data": "test"}


class TestAgentWithSmartTools:
    """Test Agent class with smart tools using mocks."""

    def test_agent_creation(self):
        """Test agent creation with all providers."""
        agents = [
            Agent(provider="openai", model="gpt-4o-mini", api_key="test"),
            Agent(provider="anthropic", model="claude-sonnet-4", api_key="test"),
            Agent(provider="google", model="gemini-2.5-flash", api_key="test"),
            Agent(provider="mock", model="test"),
        ]

        for agent in agents:
            assert hasattr(agent, "provider_name")
            assert hasattr(agent, "_provider")

    def test_smart_tools_with_mock_agent(self):
        """Test smart tools work with mock agent."""
        agent = Agent(provider="mock", model="test")

        # Test all primary smart tools
        primary_tools = ["search", "code", "image", "json", "fetch", "url"]

        for tool in primary_tools:
            response = agent.generate(f"Test {tool}", tools=[tool])
            assert isinstance(response, AssistantMessage)

    def test_tool_error_handling(self):
        """Test error handling for invalid tools."""
        agent = Agent(provider="mock", model="test")

        with pytest.raises(KeyError) as exc_info:
            agent.generate("Test", tools=["invalid_tool"])

        assert "invalid_tool" in str(exc_info.value)
        assert "Available smart tools:" in str(exc_info.value)

    @patch("llm_station.models.openai.OpenAIProvider.generate")
    def test_provider_receives_correct_tools(self, mock_generate):
        """Test that providers receive correctly routed tools."""
        mock_generate.return_value = ModelResponse(content="Test", tool_calls=[])

        agent = Agent(provider="openai", model="gpt-4o-mini", api_key="test")
        agent.generate("Test search", tools=["search"])

        # Verify provider was called with tools
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])

        # Should have exactly one tool
        assert len(tools) == 1
        # Should be OpenAI tool for OpenAI agent
        assert tools[0].provider == "openai"


class TestToolSpecGeneration:
    """Test ToolSpec generation for all tool types."""

    def test_search_tool_specs(self):
        """Test search tool specs for all providers."""
        providers = ["google", "anthropic", "openai"]

        for provider in providers:
            spec = get_tool_spec("search", provider_preference=provider)
            assert spec.provider == provider
            assert spec.requires_network == True

    def test_code_tool_specs(self):
        """Test code execution tool specs for all providers."""
        providers = ["google", "openai", "anthropic"]

        for provider in providers:
            spec = get_tool_spec("code", provider_preference=provider)
            assert spec.provider == provider

    def test_local_tool_specs(self):
        """Test local tool specifications."""
        local_tools = ["json", "fetch"]

        for tool in local_tools:
            spec = get_tool_spec(tool)
            # Local tools have no provider
            assert spec.provider is None or spec.provider == "local"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
