#!/usr/bin/env python3
"""
Comprehensive tests for OpenAI web search implementation.

Tests accuracy against the official OpenAI web search documentation:
https://platform.openai.com/docs/assistants/tools/web-search
"""

import pytest
from llm_studio.tools.web_search import OpenAIWebSearch
from llm_studio.models.openai import OpenAIProvider
from llm_studio.models.base import ModelConfig
from llm_studio.schemas.messages import UserMessage, SystemMessage, ModelResponse


class TestOpenAIWebSearchTool:
    """Test the OpenAIWebSearch tool factory."""

    def test_basic_web_search_tool(self):
        """Test basic web search tool without any options."""
        tool = OpenAIWebSearch()
        spec = tool.spec()

        assert spec.name == "web_search"
        assert spec.provider == "openai"
        assert spec.provider_type == "web_search"
        assert spec.requires_network is True
        assert spec.provider_config is None

    def test_web_search_preview_tool(self):
        """Test web search preview tool."""
        tool = OpenAIWebSearch(preview=True)
        spec = tool.spec()

        assert spec.provider_type == "web_search_preview"

    def test_domain_filtering(self):
        """Test domain filtering with allowed_domains."""
        allowed_domains = ["openai.com", "github.com", "stackoverflow.com"]
        tool = OpenAIWebSearch(allowed_domains=allowed_domains)
        spec = tool.spec()

        assert spec.provider_config is not None
        assert "filters" in spec.provider_config
        assert spec.provider_config["filters"]["allowed_domains"] == allowed_domains

    def test_domain_filtering_limit(self):
        """Test domain filtering limit of 20 domains."""
        # Test exactly 20 domains (should work)
        allowed_domains = [f"domain{i}.com" for i in range(20)]
        tool = OpenAIWebSearch(allowed_domains=allowed_domains)
        spec = tool.spec()
        assert spec.provider_config["filters"]["allowed_domains"] == allowed_domains

        # Test more than 20 domains (should raise error)
        too_many_domains = [f"domain{i}.com" for i in range(21)]
        tool = OpenAIWebSearch(allowed_domains=too_many_domains)
        with pytest.raises(
            ValueError, match="allowed_domains can contain at most 20 domains"
        ):
            tool.spec()

    def test_user_location_basic(self):
        """Test user location with basic country setting."""
        user_location = {"country": "US"}
        tool = OpenAIWebSearch(user_location=user_location)
        spec = tool.spec()

        assert spec.provider_config is not None
        assert "user_location" in spec.provider_config
        expected_location = {"type": "approximate", "country": "US"}
        assert spec.provider_config["user_location"] == expected_location

    def test_user_location_full(self):
        """Test user location with full geographic details."""
        user_location = {
            "country": "GB",
            "city": "London",
            "region": "London",
            "timezone": "Europe/London",
        }
        tool = OpenAIWebSearch(user_location=user_location)
        spec = tool.spec()

        expected_location = {
            "type": "approximate",
            "country": "GB",
            "city": "London",
            "region": "London",
            "timezone": "Europe/London",
        }
        assert spec.provider_config["user_location"] == expected_location

    def test_combined_options(self):
        """Test web search with both domain filtering and user location."""
        tool = OpenAIWebSearch(
            allowed_domains=["pubmed.ncbi.nlm.nih.gov", "clinicaltrials.gov"],
            user_location={
                "country": "US",
                "city": "Minneapolis",
                "region": "Minnesota",
            },
            preview=True,
        )
        spec = tool.spec()

        assert spec.provider_type == "web_search_preview"
        assert spec.provider_config["filters"]["allowed_domains"] == [
            "pubmed.ncbi.nlm.nih.gov",
            "clinicaltrials.gov",
        ]
        assert spec.provider_config["user_location"]["country"] == "US"
        assert spec.provider_config["user_location"]["type"] == "approximate"


class TestOpenAIProviderWebSearchIntegration:
    """Test OpenAI provider adapter integration with web search."""

    def test_tool_preparation_basic_web_search(self):
        """Test tool preparation for basic web search."""
        provider = OpenAIProvider()
        tool = OpenAIWebSearch()
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        assert len(prepared_tools) == 1
        assert prepared_tools[0] == {"type": "web_search"}

    def test_tool_preparation_with_domain_filtering(self):
        """Test tool preparation with domain filtering."""
        provider = OpenAIProvider()
        tool = OpenAIWebSearch(allowed_domains=["example.com", "test.com"])
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        expected = {
            "type": "web_search",
            "filters": {"allowed_domains": ["example.com", "test.com"]},
        }
        assert prepared_tools[0] == expected

    def test_tool_preparation_with_user_location(self):
        """Test tool preparation with user location."""
        provider = OpenAIProvider()
        tool = OpenAIWebSearch(user_location={"country": "GB", "city": "London"})
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        expected = {
            "type": "web_search",
            "user_location": {"type": "approximate", "country": "GB", "city": "London"},
        }
        assert prepared_tools[0] == expected

    def test_tool_preparation_preview_version(self):
        """Test tool preparation for preview version."""
        provider = OpenAIProvider()
        tool = OpenAIWebSearch(preview=True)
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        assert prepared_tools[0] == {"type": "web_search_preview"}

    def test_mixed_tools_preparation(self):
        """Test preparing web search tools mixed with function tools."""
        provider = OpenAIProvider()

        # Create a mock local tool spec
        from llm_studio.schemas.tooling import ToolSpec

        local_tool = ToolSpec(
            name="test_function",
            description="A test function",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
        )

        web_search_tool = OpenAIWebSearch(allowed_domains=["openai.com"]).spec()

        prepared_tools = provider.prepare_tools([local_tool, web_search_tool])

        assert len(prepared_tools) == 2

        # Function tool should be wrapped
        assert prepared_tools[0]["type"] == "function"
        assert prepared_tools[0]["function"]["name"] == "test_function"

        # Web search tool should be provider-native
        assert prepared_tools[1]["type"] == "web_search"
        assert prepared_tools[1]["filters"]["allowed_domains"] == ["openai.com"]


class TestResponsesAPISelection:
    """Test automatic Responses API selection for web search."""

    def test_responses_api_auto_selection(self):
        """Test that Responses API is automatically selected when web search tools are present."""
        provider = OpenAIProvider()
        web_search_tool = OpenAIWebSearch().spec()

        # Mock the generate method to capture the API selection logic
        messages = [UserMessage("Test query")]
        config = ModelConfig(provider="openai", model="gpt-5")

        # The provider should detect web search and try to use Responses API
        try:
            provider.generate(messages, config, tools=[web_search_tool])
        except RuntimeError as e:
            # Should get the Responses API error, not Chat Completions
            assert "Responses API not wired" in str(e)

    def test_chat_completions_for_function_tools(self):
        """Test that Chat Completions is used for regular function tools."""
        provider = OpenAIProvider()

        from llm_studio.schemas.tooling import ToolSpec

        function_tool = ToolSpec(
            name="test_function",
            description="A test function",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
        )

        messages = [UserMessage("Test query")]
        config = ModelConfig(provider="openai", model="gpt-4")

        # Should use Chat Completions, not Responses API
        try:
            provider.generate(messages, config, tools=[function_tool])
        except RuntimeError as e:
            assert "Chat Completions not wired" in str(e)

    def test_forced_responses_api(self):
        """Test forcing Responses API via config.api setting."""
        provider = OpenAIProvider()

        messages = [UserMessage("Test query")]
        config = ModelConfig(provider="openai", model="gpt-5", api="responses")

        # Should use Responses API even without web search tools
        try:
            provider.generate(messages, config, tools=[])
        except RuntimeError as e:
            assert "Responses API not wired" in str(e)


class TestModelConfigIntegration:
    """Test ModelConfig integration with OpenAI-specific fields."""

    def test_reasoning_config(self):
        """Test reasoning configuration for deep research."""
        config = ModelConfig(
            provider="openai",
            model="gpt-5",
            reasoning={"effort": "high"},
            include=["web_search_call.action.sources"],
            tool_choice="auto",
        )

        assert config.reasoning == {"effort": "high"}
        assert config.include == ["web_search_call.action.sources"]
        assert config.tool_choice == "auto"

    def test_include_sources(self):
        """Test including sources in response."""
        config = ModelConfig(
            provider="openai", model="gpt-5", include=["web_search_call.action.sources"]
        )

        assert "web_search_call.action.sources" in config.include


def test_documentation_examples():
    """Test examples from the OpenAI documentation work with our implementation."""

    # Example 1: Basic web search
    basic_tool = OpenAIWebSearch()
    spec = basic_tool.spec()
    assert spec.provider_type == "web_search"

    # Example 2: Domain filtering for medical sources
    medical_tool = OpenAIWebSearch(
        allowed_domains=[
            "pubmed.ncbi.nlm.nih.gov",
            "clinicaltrials.gov",
            "www.who.int",
            "www.cdc.gov",
            "www.fda.gov",
        ]
    )
    spec = medical_tool.spec()
    assert len(spec.provider_config["filters"]["allowed_domains"]) == 5

    # Example 3: Geographic location (London restaurants)
    location_tool = OpenAIWebSearch(
        user_location={"country": "GB", "city": "London", "region": "London"}
    )
    spec = location_tool.spec()
    assert spec.provider_config["user_location"]["country"] == "GB"
    assert spec.provider_config["user_location"]["type"] == "approximate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
