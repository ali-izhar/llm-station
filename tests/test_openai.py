#!/usr/bin/env python3
"""
Clean OpenAI tests using smart tools and mocks.
Tests OpenAI-specific functionality without API calls.
"""

import pytest
from unittest.mock import patch
from llm_studio import Agent
from llm_studio.schemas.messages import ModelResponse


class TestOpenAISmartTools:
    """Test OpenAI with smart tools."""

    @patch("llm_studio.models.openai.OpenAIProvider.generate")
    def test_openai_search_routing(self, mock_generate):
        """Test that OpenAI agent routes search to OpenAI."""
        mock_generate.return_value = ModelResponse(
            content="Search results", tool_calls=[]
        )

        agent = Agent(provider="openai", model="gpt-4o-mini", api_key="test")
        response = agent.generate("Search for information", tools=["search"])

        # Verify OpenAI tool was used
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0].provider == "openai"

    @patch("llm_studio.models.openai.OpenAIProvider.generate")
    def test_openai_code_routing(self, mock_generate):
        """Test that OpenAI agent routes code to OpenAI."""
        mock_generate.return_value = ModelResponse(
            content="Code executed", tool_calls=[]
        )

        agent = Agent(provider="openai", model="gpt-4o-mini", api_key="test")
        response = agent.generate("Execute Python code", tools=["code"])

        # Verify OpenAI code interpreter was used
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0].provider == "openai"
        assert tools[0].provider_type == "code_interpreter"

    @patch("llm_studio.models.openai.OpenAIProvider.generate")
    def test_openai_image_routing(self, mock_generate):
        """Test that OpenAI agent routes image to OpenAI."""
        mock_generate.return_value = ModelResponse(
            content="Image generated", tool_calls=[]
        )

        agent = Agent(provider="openai", model="gpt-4o-mini", api_key="test")
        response = agent.generate("Generate an image", tools=["image"])

        # Verify OpenAI image generation was used
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0].provider == "openai"
        assert tools[0].provider_type == "image_generation"

    def test_openai_api_type_detection(self):
        """Test OpenAI API type detection logic."""
        from llm_studio.models.openai import OpenAIProvider
        from llm_studio.models.base import ModelConfig
        from llm_studio.schemas.tooling import ToolSpec

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
