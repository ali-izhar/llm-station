#!/usr/bin/env python3
"""
Integration tests for the complete smart tools system.
Tests the full workflow using mocks without API calls.
"""

import pytest
from unittest.mock import Mock, patch

from llm_studio import Agent
from llm_studio.schemas.messages import ModelResponse, ToolCall
from llm_studio.tools.fetch_url import FetchUrlTool
from llm_studio.tools.json_format import JsonFormatTool
from llm_studio.tools.registry import get_tool_spec


class TestEndToEndWorkflows:
    """Test complete workflows with smart tools."""

    @patch("llm_studio.models.openai.OpenAIProvider.generate")
    def test_openai_search_workflow(self, mock_generate):
        """Test complete search workflow with OpenAI."""
        # Mock OpenAI returning search results
        mock_generate.return_value = ModelResponse(
            content="AI developments include...",
            tool_calls=[],
            grounding_metadata={
                "web_search": {"id": "search_123", "status": "completed"},
                "citations": [{"url": "https://example.com", "title": "AI News"}],
            },
        )

        agent = Agent(provider="openai", model="gpt-4o-mini", api_key="test")
        response = agent.generate("Research AI developments", tools=["search"])

        # Verify response
        assert isinstance(response.content, str)
        assert response.grounding_metadata is not None

        # Verify provider was called with correct tools
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0].provider == "openai"
        assert tools[0].provider_type in ["web_search", "web_search_preview"]

    @patch("llm_studio.models.google.GoogleProvider.generate")
    def test_google_code_workflow(self, mock_generate):
        """Test complete code execution workflow with Google."""
        # Mock Google returning code execution results
        mock_generate.return_value = ModelResponse(
            content="Calculated result: 120",
            tool_calls=[],
            grounding_metadata={
                "code_execution": [
                    {
                        "code": "import math\nresult = math.factorial(5)",
                        "result": {"output": "120"},
                    }
                ]
            },
        )

        agent = Agent(provider="google", model="gemini-2.5-flash", api_key="test")
        response = agent.generate("Calculate factorial of 5", tools=["code"])

        # Verify response
        assert "120" in response.content
        assert response.grounding_metadata is not None

        # Verify provider was called with correct tools
        args, kwargs = mock_generate.call_args
        tools = kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0].provider == "google"
        assert tools[0].provider_type == "code_execution"

    def test_local_tool_execution_workflow(self):
        """Test local tool execution workflow."""
        agent = Agent(provider="mock", model="test")

        # Mock provider returning tool call for local tool
        with patch("llm_studio.models.mock.MockProvider.generate") as mock_generate:
            mock_generate.return_value = ModelResponse(
                content="I'll format that as JSON",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="json_format",
                        arguments={"data": {"name": "test"}},
                    )
                ],
            )

            # Mock local tool execution
            with patch.object(agent, "_execute_tool") as mock_execute:
                mock_execute.return_value = Mock(content='{"name":"test"}')

                response = agent.generate("Format as JSON: name=test", tools=["json"])

                # Local tool should have been executed
                mock_execute.assert_called_once()
                assert "json_format result:" in response.content

    def test_mixed_tools_workflow(self):
        """Test workflow with both provider and local tools."""
        agent = Agent(provider="mock", model="test")

        # Test that multiple tools can be specified
        response = agent.generate(
            "Research and format results", tools=["search", "json", "code"]
        )

        assert isinstance(response.content, str)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_no_tools_workflow(self):
        """Test workflow without any tools."""
        agent = Agent(provider="mock", model="test")
        response = agent.generate("Simple question without tools")

        assert isinstance(response.content, str)
        assert response.tool_calls is None or len(response.tool_calls) == 0

    def test_invalid_tool_configuration(self):
        """Test invalid tool configuration handling."""
        agent = Agent(provider="mock", model="test")

        # Test invalid dict format
        with pytest.raises(TypeError):
            agent.generate("Test", tools=[{"invalid": "config"}])  # Missing 'name' key

    def test_provider_exclusion_all_providers(self):
        """Test error when all providers are excluded."""
        with pytest.raises(KeyError) as exc_info:
            get_tool_spec("search", exclude_providers=["google", "anthropic", "openai"])

        assert "No available providers" in str(exc_info.value)

    @patch("llm_studio.models.mock.MockProvider.generate")
    def test_provider_error_handling(self, mock_generate):
        """Test handling of provider errors."""
        # Mock provider raising an error
        mock_generate.side_effect = Exception("Provider error")

        agent = Agent(provider="mock", model="test")
        response = agent.generate("Test message")

        # Should return error in content
        assert "Provider error" in response.content

    def test_tool_spec_validation(self):
        """Test ToolSpec validation and requirements."""
        from llm_studio.tools.registry import get_tool_spec

        # Test that required tools exist
        required_tools = ["search", "code", "image", "json"]
        for tool in required_tools:
            spec = get_tool_spec(tool)
            assert spec.name is not None
            assert spec.description is not None
            assert isinstance(spec.input_schema, dict)


class TestBackwardCompatibility:
    """Test that the system maintains necessary compatibility."""

    def test_toolspec_direct_usage(self):
        """Test that ToolSpec instances still work directly."""
        from llm_studio.schemas.tooling import ToolSpec

        agent = Agent(provider="mock", model="test")

        # Create ToolSpec directly
        tool_spec = ToolSpec(
            name="custom_tool",
            description="Custom tool for testing",
            input_schema={"type": "object", "properties": {"test": {"type": "string"}}},
        )

        # Should work with agent
        response = agent.generate("Test custom tool", tools=[tool_spec])
        assert isinstance(response.content, str)

    def test_string_and_dict_tool_formats(self):
        """Test different tool specification formats."""
        agent = Agent(provider="mock", model="test")

        # String format (smart tools)
        response1 = agent.generate("Test", tools=["search"])

        # Dict format with preferences
        response2 = agent.generate(
            "Test", tools=[{"name": "search", "provider_preference": "google"}]
        )

        # Both should work
        assert isinstance(response1.content, str)
        assert isinstance(response2.content, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
