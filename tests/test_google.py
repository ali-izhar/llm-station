#!/usr/bin/env python3
"""
Clean Google tests using smart tools and mocks.
Tests Google-specific functionality without API calls.
"""

import pytest
from unittest.mock import patch
from llm_station import Agent
from llm_station.schemas.messages import ModelResponse


class TestGoogleSmartTools:
    """Test Google with smart tools."""

    @patch("llm_station.models.google.GoogleProvider.generate")
    def test_google_search_routing(self, mock_generate):
        """Test that Google agent routes search to Google."""
        mock_generate.return_value = ModelResponse(
            content="Search results", tool_calls=[]
        )

        agent = Agent(provider="google", model="gemini-2.5-flash", api_key="test")
        response = agent.generate("Search for information", tools=["search"])

        # Verify Google tool was used
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0].provider == "google"

    @patch("llm_station.models.google.GoogleProvider.generate")
    def test_google_code_routing(self, mock_generate):
        """Test that Google agent routes code to Google."""
        mock_generate.return_value = ModelResponse(
            content="Code executed", tool_calls=[]
        )

        agent = Agent(provider="google", model="gemini-2.5-flash", api_key="test")
        response = agent.generate("Execute Python code", tools=["code"])

        # Verify Google code execution was used
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0].provider == "google"
        assert tools[0].provider_type == "code_execution"

    @patch("llm_station.models.google.GoogleProvider.generate")
    def test_google_url_routing(self, mock_generate):
        """Test that Google agent routes URL context to Google."""
        mock_generate.return_value = ModelResponse(
            content="URL processed", tool_calls=[]
        )

        agent = Agent(provider="google", model="gemini-2.5-flash", api_key="test")
        response = agent.generate("Process URL content", tools=["url"])

        # Verify Google URL context was used
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0].provider == "google"
        assert tools[0].provider_type == "url_context"

    def test_google_tool_preparation(self):
        """Test Google tool preparation logic."""
        from llm_station.models.google import GoogleProvider
        from llm_station.schemas.tooling import ToolSpec

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
