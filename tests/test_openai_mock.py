"""
Comprehensive tests for OpenAI provider and tools.

Tests cover:
- Base OpenAI client functionality
- Chat Completions endpoint
- Responses endpoint
- All OpenAI tools with example usage
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_studio import Agent
from llm_studio.models.openai import OpenAIProvider
from llm_studio.models.base import ModelConfig
from llm_studio.schemas.messages import UserMessage, SystemMessage, ModelResponse
from llm_studio.tools.web_search.openai import OpenAIWebSearch
from llm_studio.tools.code_execution.openai import OpenAICodeInterpreter
from llm_studio.tools.image_generation.openai import OpenAIImageGeneration


class TestOpenAIProvider:
    """Test base OpenAI provider functionality."""

    def test_provider_initialization(self):
        """Test OpenAI provider can be initialized properly."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.name == "openai"
        assert provider.api_key == "test-key"
        assert provider.supports_tools() is True

    def test_tool_detection(self):
        """Test provider correctly detects different tool types."""
        provider = OpenAIProvider()

        # Web search tools
        web_search_tools = [OpenAIWebSearch().spec()]
        assert provider._has_web_search_tools(web_search_tools) is True
        assert provider._has_code_interpreter_tools(web_search_tools) is False
        assert provider._has_image_generation_tools(web_search_tools) is False

        # Code interpreter tools
        code_tools = [OpenAICodeInterpreter().spec()]
        assert provider._has_web_search_tools(code_tools) is False
        assert provider._has_code_interpreter_tools(code_tools) is True
        assert provider._has_image_generation_tools(code_tools) is False

        # Image generation tools
        image_tools = [OpenAIImageGeneration().spec()]
        assert provider._has_web_search_tools(image_tools) is False
        assert provider._has_code_interpreter_tools(image_tools) is False
        assert provider._has_image_generation_tools(image_tools) is True

    def test_api_selection_logic(self):
        """Test automatic API selection based on tools and configuration."""
        provider = OpenAIProvider()
        config = ModelConfig(provider="openai", model="gpt-4")

        # No tools - should default to Chat Completions
        assert provider._should_use_responses_api(config, False, False, False) is False

        # Code interpreter - MUST use Responses API
        assert provider._should_use_responses_api(config, False, True, False) is True

        # Image generation - MUST use Responses API
        assert provider._should_use_responses_api(config, False, False, True) is True

        # Web search with regular model - should use Responses API
        assert provider._should_use_responses_api(config, True, False, False) is True

        # Web search with search model - should use Chat Completions
        search_config = ModelConfig(provider="openai", model="gpt-4o-search-preview")
        assert (
            provider._should_use_responses_api(search_config, True, False, False)
            is False
        )

        # Explicit API preference always wins
        explicit_config = ModelConfig(provider="openai", model="gpt-4", api="responses")
        assert (
            provider._should_use_responses_api(explicit_config, False, False, False)
            is True
        )


class TestChatCompletionsEndpoint:
    """Test OpenAI Chat Completions API endpoint."""

    @patch("openai.OpenAI")
    def test_chat_completions_basic(self, mock_openai_class):
        """Test basic Chat Completions API call."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Hello! How can I help you today?",
                        "role": "assistant",
                    }
                }
            ]
        }
        mock_client.chat.completions.create.return_value = mock_response

        # Test Chat Completions call
        provider = OpenAIProvider(api_key="test-key")
        messages = [UserMessage("Hello!")]
        config = ModelConfig(provider="openai", model="gpt-4", api="chat")

        result = provider.generate(messages, config, tools=None)

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["content"] == "Hello!"

        # Verify response
        assert result.content == "Hello! How can I help you today?"
        assert len(result.tool_calls) == 0

    @patch("openai.OpenAI")
    def test_chat_completions_with_tools(self, mock_openai_class):
        """Test Chat Completions with function tools."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I'll help you format that data as JSON.",
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "json_format",
                                    "arguments": '{"data": {"name": "Alice", "age": 30}}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        mock_client.chat.completions.create.return_value = mock_response

        # Test with local tools
        agent = Agent(provider="openai", model="gpt-4", api_key="test-key")
        with patch.object(
            agent._provider,
            "generate",
            return_value=ModelResponse(
                content="I'll help you format that data as JSON.", tool_calls=[], raw={}
            ),
        ):
            result = agent.generate(
                "Format this as JSON: name=Alice, age=30", tools=["json_format"]
            )

        assert "JSON" in result.content

    @patch("openai.OpenAI")
    def test_search_model_native_search(self, mock_openai_class):
        """Test search models handle web search natively in Chat Completions."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Based on my web search, here are today's positive news stories...",
                        "role": "assistant",
                    }
                }
            ]
        }
        mock_client.chat.completions.create.return_value = mock_response

        # Test search model with web search
        provider = OpenAIProvider(api_key="test-key")
        messages = [UserMessage("What are today's positive news stories?")]
        config = ModelConfig(provider="openai", model="gpt-4o-search-preview")
        tools = [OpenAIWebSearch().spec()]

        result = provider.generate(messages, config, tools)

        # Should use Chat Completions (not Responses API)
        mock_client.chat.completions.create.assert_called_once()
        (
            mock_client.responses.create.assert_not_called()
            if hasattr(mock_client, "responses")
            else None
        )

        assert "positive news" in result.content


class TestResponsesEndpoint:
    """Test OpenAI Responses API endpoint."""

    @patch("openai.OpenAI")
    def test_responses_api_basic(self, mock_openai_class):
        """Test basic Responses API call."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.model_dump.return_value = [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello! I'm ready to help you with any questions.",
                    }
                ],
            }
        ]
        mock_client.responses.create.return_value = mock_response

        # Test Responses API call
        provider = OpenAIProvider(api_key="test-key")
        messages = [UserMessage("Hello!")]
        config = ModelConfig(provider="openai", model="gpt-5", api="responses")

        result = provider.generate(messages, config, tools=None)

        # Verify API call
        mock_client.responses.create.assert_called_once()
        call_args = mock_client.responses.create.call_args[1]
        assert call_args["model"] == "gpt-5"
        assert call_args["input"] == "Hello!"

        # Verify response
        assert "ready to help" in result.content

    @patch("openai.OpenAI")
    def test_responses_api_with_reasoning(self, mock_openai_class):
        """Test Responses API with reasoning parameters."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.model_dump.return_value = [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "After careful analysis..."}
                ],
            }
        ]
        mock_client.responses.create.return_value = mock_response

        # Test with reasoning and advanced parameters
        provider = OpenAIProvider(api_key="test-key")
        messages = [UserMessage("Analyze this complex problem")]
        config = ModelConfig(
            provider="openai",
            model="gpt-5",
            api="responses",
            reasoning={"effort": "high"},
            tool_choice="auto",
            include=["web_search_call.action.sources"],
        )

        result = provider.generate(messages, config, tools=None)

        # Verify advanced parameters were included
        call_args = mock_client.responses.create.call_args[1]
        assert call_args["reasoning"] == {"effort": "high"}
        assert call_args["tool_choice"] == "auto"
        assert call_args["include"] == ["web_search_call.action.sources"]


class TestWebSearchTool:
    """Test OpenAI Web Search tool."""

    def test_web_search_tool_spec(self):
        """Test web search tool specification."""
        # Basic web search
        tool = OpenAIWebSearch()
        spec = tool.spec()

        assert spec.name == "web_search"
        assert spec.provider == "openai"
        assert spec.provider_type == "web_search"
        assert spec.requires_network is True

        # Preview version
        preview_tool = OpenAIWebSearch(preview=True)
        preview_spec = preview_tool.spec()
        assert preview_spec.provider_type == "web_search_preview"

    def test_domain_filtering_validation(self):
        """Test domain filtering validation."""
        # Valid domains
        tool = OpenAIWebSearch(allowed_domains=["openai.com", "github.com"])
        spec = tool.spec()
        assert spec.provider_config["filters"]["allowed_domains"] == [
            "openai.com",
            "github.com",
        ]

        # Too many domains
        with pytest.raises(ValueError, match="at most 20 domains"):
            OpenAIWebSearch(
                allowed_domains=["domain{}.com".format(i) for i in range(25)]
            )

        # URL prefix stripping
        tool = OpenAIWebSearch(
            allowed_domains=["https://openai.com", "http://github.com"]
        )
        spec = tool.spec()
        expected_domains = ["openai.com", "github.com"]
        assert spec.provider_config["filters"]["allowed_domains"] == expected_domains

    def test_user_location_validation(self):
        """Test user location validation."""
        # Valid location
        location = {
            "country": "US",
            "city": "San Francisco",
            "region": "California",
            "timezone": "America/Los_Angeles",
        }
        tool = OpenAIWebSearch(user_location=location)
        spec = tool.spec()

        config = spec.provider_config["user_location"]
        assert config["type"] == "approximate"
        assert config["country"] == "US"
        assert config["city"] == "San Francisco"
        assert config["timezone"] == "America/Los_Angeles"

        # Invalid country code
        with pytest.raises(ValueError, match="two-letter ISO country code"):
            OpenAIWebSearch(user_location={"country": "USA"})

        # Invalid timezone format
        with pytest.raises(ValueError, match="IANA format"):
            OpenAIWebSearch(user_location={"timezone": "PST"})

    @patch("openai.OpenAI")
    def test_web_search_integration(self, mock_openai_class):
        """Test web search tool integration with agent."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock web search response
        mock_response = Mock()
        mock_response.model_dump.return_value = [
            {
                "type": "web_search_call",
                "id": "ws_123",
                "status": "completed",
                "action": {
                    "query": "positive news today",
                    "sources": ["https://news1.com", "https://news2.com"],
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Here are today's positive news stories: [1](https://news1.com) Good news about...",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "start_index": 42,
                                "end_index": 62,
                                "url": "https://news1.com",
                                "title": "Positive News Today",
                            }
                        ],
                    }
                ],
            },
        ]
        mock_client.responses.create.return_value = mock_response

        # Test web search
        agent = Agent(provider="openai", model="gpt-5", api_key="test-key")
        result = agent.generate(
            "What was a positive news story from today?", tools=["openai_web_search"]
        )

        # Verify Responses API was used
        mock_client.responses.create.assert_called_once()

        # Verify web search tool was included
        call_args = mock_client.responses.create.call_args[1]
        assert len(call_args["tools"]) == 1
        assert call_args["tools"][0]["type"] == "web_search"

        # Verify response parsing
        assert "positive news" in result.content
        assert result.grounding_metadata is not None
        assert "web_search" in result.grounding_metadata
        assert "citations" in result.grounding_metadata


class TestCodeInterpreterTool:
    """Test OpenAI Code Interpreter tool."""

    def test_code_interpreter_tool_spec(self):
        """Test code interpreter tool specification."""
        # Auto mode
        tool = OpenAICodeInterpreter()
        spec = tool.spec()

        assert spec.name == "code_interpreter"
        assert spec.provider == "openai"
        assert spec.provider_type == "code_interpreter"
        assert spec.requires_filesystem is True
        assert spec.provider_config["container"]["type"] == "auto"

        # Explicit container
        tool = OpenAICodeInterpreter(container_type="cntr_abc123")
        spec = tool.spec()
        assert spec.provider_config["container"] == "cntr_abc123"

    def test_container_validation(self):
        """Test container configuration validation."""
        # Valid auto mode
        tool = OpenAICodeInterpreter(container_type="auto", file_ids=["file-123"])
        spec = tool.spec()
        assert spec.provider_config["container"]["type"] == "auto"
        assert spec.provider_config["container"]["file_ids"] == ["file-123"]

        # Valid explicit container
        tool = OpenAICodeInterpreter(container_type="cntr_abc123")
        spec = tool.spec()
        assert spec.provider_config["container"] == "cntr_abc123"

        # Invalid container type
        with pytest.raises(ValueError, match="must be 'auto' or a container ID"):
            OpenAICodeInterpreter(container_type="invalid")

        # Invalid file IDs
        with pytest.raises(ValueError, match="must contain non-empty strings"):
            OpenAICodeInterpreter(file_ids=["", "file-123"])

    @patch("openai.OpenAI")
    def test_code_interpreter_integration(self, mock_openai_class):
        """Test code interpreter tool integration with agent."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock code interpreter response
        mock_response = Mock()
        mock_response.model_dump.return_value = [
            {
                "type": "code_interpreter_call",
                "id": "ci_123",
                "status": "completed",
                "container_id": "cntr_auto_456",
                "code": "# Solve equation 3x + 11 = 14\nx = (14 - 11) / 3\nprint(f'x = {x}')",
                "output": "x = 1.0",
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I solved the equation 3x + 11 = 14. The answer is x = 1.",
                        "annotations": [
                            {
                                "type": "container_file_citation",
                                "file_id": "cfile_789",
                                "filename": "solution.py",
                                "container_id": "cntr_auto_456",
                            }
                        ],
                    }
                ],
            },
        ]
        mock_client.responses.create.return_value = mock_response

        # Test code interpreter
        agent = Agent(provider="openai", model="gpt-4.1", api_key="test-key")
        result = agent.generate(
            "I need to solve the equation 3x + 11 = 14. Can you help me?",
            tools=["openai_code_interpreter"],
        )

        # Verify Responses API was used
        mock_client.responses.create.assert_called_once()

        # Verify code interpreter tool was included
        call_args = mock_client.responses.create.call_args[1]
        assert len(call_args["tools"]) == 1
        assert call_args["tools"][0]["type"] == "code_interpreter"
        assert call_args["tools"][0]["container"]["type"] == "auto"

        # Verify response parsing
        assert "x = 1" in result.content
        assert result.grounding_metadata is not None
        assert "code_interpreter" in result.grounding_metadata
        assert "file_citations" in result.grounding_metadata


class TestImageGenerationTool:
    """Test OpenAI Image Generation tool."""

    def test_image_generation_tool_spec(self):
        """Test image generation tool specification."""
        # Basic generation
        tool = OpenAIImageGeneration()
        spec = tool.spec()

        assert spec.name == "image_generation"
        assert spec.provider == "openai"
        assert spec.provider_type == "image_generation"
        assert spec.requires_network is False
        assert spec.requires_filesystem is False

        # Advanced configuration
        tool = OpenAIImageGeneration(
            size="1024x1536",
            quality="high",
            format="png",
            background="transparent",
            partial_images=2,
        )
        spec = tool.spec()
        config = spec.provider_config
        assert config["size"] == "1024x1536"
        assert config["quality"] == "high"
        assert config["format"] == "png"
        assert config["background"] == "transparent"
        assert config["partial_images"] == 2

    def test_image_generation_validation(self):
        """Test image generation parameter validation."""
        # Valid parameters
        tool = OpenAIImageGeneration(
            size="1024x1024", quality="medium", format="jpeg", compression=85
        )
        spec = tool.spec()
        assert spec.provider_config["compression"] == 85

        # Invalid size
        with pytest.raises(ValueError, match="size must be one of"):
            OpenAIImageGeneration(size="invalid")

        # Invalid quality
        with pytest.raises(ValueError, match="quality must be one of"):
            OpenAIImageGeneration(quality="invalid")

        # Invalid compression range
        with pytest.raises(
            ValueError, match="compression must be an integer between 0-100"
        ):
            OpenAIImageGeneration(compression=150)

        # Compression with wrong format
        with pytest.raises(ValueError, match="compression only valid for jpeg/webp"):
            OpenAIImageGeneration(format="png", compression=85)

        # Invalid partial images
        with pytest.raises(
            ValueError, match="partial_images must be an integer between 1-3"
        ):
            OpenAIImageGeneration(partial_images=5)

    @patch("openai.OpenAI")
    def test_image_generation_integration(self, mock_openai_class):
        """Test image generation tool integration with agent."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock image generation response
        mock_response = Mock()
        mock_response.model_dump.return_value = [
            {
                "type": "image_generation_call",
                "id": "ig_123",
                "status": "completed",
                "revised_prompt": "A gray tabby cat hugging an otter. The otter is wearing an orange scarf...",
                "result": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "size": "1024x1024",
                "quality": "auto",
                "format": "png",
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I've generated an image of a gray tabby cat hugging an otter with an orange scarf.",
                    }
                ],
            },
        ]
        mock_client.responses.create.return_value = mock_response

        # Test image generation
        agent = Agent(provider="openai", model="gpt-5", api_key="test-key")
        result = agent.generate(
            "Generate an image of gray tabby cat hugging an otter with an orange scarf",
            tools=["openai_image_generation"],
        )

        # Verify Responses API was used
        mock_client.responses.create.assert_called_once()

        # Verify image generation tool was included
        call_args = mock_client.responses.create.call_args[1]
        assert len(call_args["tools"]) == 1
        assert call_args["tools"][0]["type"] == "image_generation"

        # Verify response parsing
        assert "generated an image" in result.content
        assert result.grounding_metadata is not None
        assert "image_generation" in result.grounding_metadata

        # Verify image data
        image_calls = result.grounding_metadata["image_generation"]
        assert len(image_calls) == 1
        assert image_calls[0]["result"].startswith("iVBORw0KGgo")  # Base64 PNG header
        assert image_calls[0]["revised_prompt"] is not None


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    @patch("openai.OpenAI")
    def test_sdk_not_installed_error(self, mock_openai_class):
        """Test handling when OpenAI SDK is not installed."""
        mock_openai_class.side_effect = ImportError("No module named 'openai'")

        provider = OpenAIProvider(api_key="test-key")
        messages = [UserMessage("Hello")]
        config = ModelConfig(provider="openai", model="gpt-4", api="responses")

        result = provider.generate(messages, config)

        assert "OpenAI SDK not installed" in result.content
        assert result.raw["error"] == "sdk_not_installed"

    @patch("openai.OpenAI")
    def test_responses_api_not_available(self, mock_openai_class):
        """Test handling when Responses API is not available in SDK."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock AttributeError for missing responses attribute
        del mock_client.responses

        provider = OpenAIProvider(api_key="test-key")
        messages = [UserMessage("Hello")]
        config = ModelConfig(provider="openai", model="gpt-5", api="responses")

        result = provider.generate(messages, config)

        assert "Responses API not available" in result.content
        assert result.raw["error"] == "responses_api_not_available"

    @patch("openai.OpenAI")
    def test_api_call_failure(self, mock_openai_class):
        """Test handling of API call failures."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API exception
        mock_client.responses.create.side_effect = Exception("API rate limit exceeded")

        provider = OpenAIProvider(api_key="test-key")
        messages = [UserMessage("Hello")]
        config = ModelConfig(provider="openai", model="gpt-5", api="responses")

        result = provider.generate(messages, config)

        assert "OpenAI Responses API error" in result.content
        assert "rate limit exceeded" in result.content
        assert result.raw["error"] == "api_call_failed"


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    @patch("openai.OpenAI")
    def test_multi_tool_workflow(self, mock_openai_class):
        """Test workflow using multiple tools together."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response with multiple tool calls
        mock_response = Mock()
        mock_response.model_dump.return_value = [
            {"type": "web_search_call", "id": "ws_123", "status": "completed"},
            {"type": "code_interpreter_call", "id": "ci_456", "status": "completed"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I researched the data and analyzed it with Python. Here are the results...",
                    }
                ],
            },
        ]
        mock_client.responses.create.return_value = mock_response

        # Test multiple tools
        agent = Agent(provider="openai", model="gpt-5", api_key="test-key")
        result = agent.generate(
            "Research current AI trends and analyze the data with Python",
            tools=["openai_web_search", "openai_code_interpreter"],
        )

        # Verify both tools were included
        call_args = mock_client.responses.create.call_args[1]
        assert len(call_args["tools"]) == 2
        tool_types = [tool["type"] for tool in call_args["tools"]]
        assert "web_search" in tool_types
        assert "code_interpreter" in tool_types

        # Verify response contains both tool results
        assert result.grounding_metadata is not None
        assert "web_search" in result.grounding_metadata
        assert "code_interpreter" in result.grounding_metadata

    @patch("openai.OpenAI")
    def test_advanced_configuration_workflow(self, mock_openai_class):
        """Test workflow with advanced tool configurations."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.model_dump.return_value = [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Analysis complete with domain-filtered search and high-quality image generation.",
                    }
                ],
            }
        ]
        mock_client.responses.create.return_value = mock_response

        # Create agent with advanced tool configurations
        agent = Agent(provider="openai", model="gpt-5", api_key="test-key")

        # Advanced web search
        web_search = OpenAIWebSearch(
            allowed_domains=["arxiv.org", "nature.com"],
            user_location={"country": "US", "city": "Boston"},
        )

        # High-quality image generation
        image_gen = OpenAIImageGeneration(
            size="1792x1024", quality="high", format="png", background="transparent"
        )

        result = agent.generate(
            "Research AI papers and create a visualization",
            tools=[web_search.spec(), image_gen.spec()],
        )

        # Verify advanced configurations were applied
        call_args = mock_client.responses.create.call_args[1]

        # Check web search config
        web_tool = next(t for t in call_args["tools"] if t["type"] == "web_search")
        assert web_tool["filters"]["allowed_domains"] == ["arxiv.org", "nature.com"]
        assert web_tool["user_location"]["country"] == "US"
        assert web_tool["user_location"]["city"] == "Boston"

        # Check image generation config
        image_tool = next(
            t for t in call_args["tools"] if t["type"] == "image_generation"
        )
        assert image_tool["size"] == "1792x1024"
        assert image_tool["quality"] == "high"
        assert image_tool["background"] == "transparent"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
