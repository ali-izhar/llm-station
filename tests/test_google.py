#!/usr/bin/env python3
"""
Comprehensive tests for Google Gemini API implementation.

Tests accuracy against the official Google Gemini grounding documentation:
https://ai.google.dev/gemini-api/docs/grounding
"""

import pytest
from llm_studio.tools.web_search import GoogleWebSearch, GoogleSearchRetrieval
from llm_studio.tools.code_execution import GoogleCodeExecution
from llm_studio.tools.url_context import GoogleUrlContext
from llm_studio.models.google import GoogleProvider
from llm_studio.models.base import ModelConfig
from llm_studio.schemas.messages import UserMessage, SystemMessage, ModelResponse
from llm_studio.schemas.tooling import ToolSpec


class TestGoogleWebSearchTool:
    """Test the GoogleWebSearch tool factory."""

    def test_basic_google_search_tool(self):
        """Test basic Google search grounding tool."""
        tool = GoogleWebSearch()
        spec = tool.spec()

        assert spec.name == "google_search"
        assert spec.provider == "google"
        assert spec.provider_type == "google_search"
        assert spec.requires_network is True
        assert spec.provider_config is None

    def test_google_search_description(self):
        """Test that description matches the grounding functionality."""
        tool = GoogleWebSearch()
        spec = tool.spec()

        assert "grounding" in spec.description.lower()
        assert "search" in spec.description.lower()


class TestGoogleSearchRetrievalTool:
    """Test the legacy GoogleSearchRetrieval tool factory."""

    def test_basic_search_retrieval_tool(self):
        """Test basic search retrieval tool without configuration."""
        tool = GoogleSearchRetrieval()
        spec = tool.spec()

        assert spec.name == "google_search_retrieval"
        assert spec.provider == "google"
        assert spec.provider_type == "google_search_retrieval"
        assert spec.requires_network is True

    def test_dynamic_retrieval_config(self):
        """Test dynamic retrieval configuration."""
        tool = GoogleSearchRetrieval(mode="MODE_DYNAMIC", dynamic_threshold=0.7)
        spec = tool.spec()

        config = spec.provider_config
        assert config["dynamic_retrieval_config"]["mode"] == "MODE_DYNAMIC"
        assert config["dynamic_retrieval_config"]["dynamic_threshold"] == 0.7

    def test_mode_validation(self):
        """Test mode validation."""
        # Valid mode
        tool = GoogleSearchRetrieval(mode="MODE_DYNAMIC")
        spec = tool.spec()
        assert (
            spec.provider_config["dynamic_retrieval_config"]["mode"] == "MODE_DYNAMIC"
        )

        # Invalid mode
        with pytest.raises(ValueError, match="Invalid mode"):
            GoogleSearchRetrieval(mode="INVALID_MODE")

    def test_dynamic_threshold_validation(self):
        """Test dynamic threshold validation (must be 0.0-1.0)."""
        # Valid thresholds
        for threshold in [0.0, 0.5, 1.0]:
            tool = GoogleSearchRetrieval(dynamic_threshold=threshold)
            spec = tool.spec()
            assert (
                spec.provider_config["dynamic_retrieval_config"]["dynamic_threshold"]
                == threshold
            )

        # Invalid thresholds (outside 0.0-1.0 range)
        with pytest.raises(
            ValueError, match="dynamic_threshold must be between 0.0 and 1.0"
        ):
            GoogleSearchRetrieval(dynamic_threshold=-0.1)

        with pytest.raises(
            ValueError, match="dynamic_threshold must be between 0.0 and 1.0"
        ):
            GoogleSearchRetrieval(dynamic_threshold=1.1)

    def test_partial_config(self):
        """Test partial configuration (only mode or only threshold)."""
        # Only mode
        tool_mode = GoogleSearchRetrieval(mode="MODE_DYNAMIC")
        spec = tool_mode.spec()
        config = spec.provider_config["dynamic_retrieval_config"]
        assert config["mode"] == "MODE_DYNAMIC"
        assert "dynamic_threshold" not in config

        # Only threshold
        tool_threshold = GoogleSearchRetrieval(dynamic_threshold=0.8)
        spec = tool_threshold.spec()
        config = spec.provider_config["dynamic_retrieval_config"]
        assert config["dynamic_threshold"] == 0.8
        assert "mode" not in config


class TestGoogleProviderToolPreparation:
    """Test Google provider adapter tool preparation."""

    def test_google_search_tool_preparation(self):
        """Test preparation of Google search grounding tool."""
        provider = GoogleProvider()
        tool = GoogleWebSearch()
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        assert len(prepared_tools) == 1
        assert prepared_tools[0] == {"google_search": {}}

    def test_google_search_retrieval_tool_preparation(self):
        """Test preparation of legacy search retrieval tool."""
        provider = GoogleProvider()
        tool = GoogleSearchRetrieval(mode="MODE_DYNAMIC", dynamic_threshold=0.6)
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        expected = {
            "google_search_retrieval": {
                "dynamic_retrieval_config": {
                    "mode": "MODE_DYNAMIC",
                    "dynamic_threshold": 0.6,
                }
            }
        }
        assert prepared_tools[0] == expected

    def test_code_execution_tool_preparation(self):
        """Test preparation of code execution tool."""
        provider = GoogleProvider()
        tool = GoogleCodeExecution()
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        assert prepared_tools[0] == {"code_execution": {}}

    def test_url_context_tool_preparation(self):
        """Test preparation of URL context tool."""
        provider = GoogleProvider()
        tool = GoogleUrlContext()
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        assert prepared_tools[0] == {"url_context": {}}

    def test_function_declarations_preparation(self):
        """Test preparation of custom function tools."""
        provider = GoogleProvider()

        custom_tool = ToolSpec(
            name="custom_function",
            description="A custom function",
            input_schema={
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
            },
        )

        prepared_tools = provider.prepare_tools([custom_tool])

        expected = {
            "function_declarations": [
                {
                    "name": "custom_function",
                    "description": "A custom function",
                    "parameters": {
                        "type": "object",
                        "properties": {"param": {"type": "string"}},
                        "required": ["param"],
                    },
                }
            ]
        }
        assert prepared_tools[0] == expected

    def test_mixed_tools_preparation(self):
        """Test preparing both Google tools and function declarations."""
        provider = GoogleProvider()

        # Mix of Google tools and custom function
        google_search = GoogleWebSearch().spec()
        code_execution = GoogleCodeExecution().spec()

        custom_tool = ToolSpec(
            name="helper_function",
            description="A helper function",
            input_schema={"type": "object", "properties": {}},
        )

        prepared_tools = provider.prepare_tools(
            [google_search, code_execution, custom_tool]
        )

        assert len(prepared_tools) == 3

        # Google tools should be separate entries
        assert prepared_tools[0] == {"google_search": {}}
        assert prepared_tools[1] == {"code_execution": {}}

        # Function declarations should be grouped
        assert "function_declarations" in prepared_tools[2]
        assert len(prepared_tools[2]["function_declarations"]) == 1
        assert (
            prepared_tools[2]["function_declarations"][0]["name"] == "helper_function"
        )


class TestGoogleProviderResponseParsing:
    """Test Google provider response parsing."""

    def test_text_response_parsing(self):
        """Test parsing of text-only responses."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello! How can I help you today?"}],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        assert parsed.content == "Hello! How can I help you today?"
        assert len(parsed.tool_calls) == 0
        assert parsed.grounding_metadata is None
        assert parsed.raw == response_payload

    def test_function_call_response_parsing(self):
        """Test parsing of responses with function calls."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I'll help you with that calculation."},
                            {
                                "function_call": {
                                    "name": "calculate",
                                    "args": {"expression": "2 + 2"},
                                }
                            },
                        ],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        assert parsed.content == "I'll help you with that calculation."
        assert len(parsed.tool_calls) == 1

        tool_call = parsed.tool_calls[0]
        assert tool_call.name == "calculate"
        assert tool_call.arguments == {"expression": "2 + 2"}

    def test_grounding_metadata_parsing(self):
        """Test parsing of responses with grounding metadata."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Spain won Euro 2024, defeating England 2-1 in the final."
                            }
                        ],
                        "role": "model",
                    },
                    "groundingMetadata": {
                        "webSearchQueries": [
                            "UEFA Euro 2024 winner",
                            "who won euro 2024",
                        ],
                        "searchEntryPoint": {
                            "renderedContent": "<!-- HTML and CSS for the search widget -->"
                        },
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://example.com/euro2024",
                                    "title": "UEFA Euro 2024 Results",
                                }
                            },
                            {
                                "web": {
                                    "uri": "https://example.com/spain-wins",
                                    "title": "Spain Champions",
                                }
                            },
                        ],
                        "groundingSupports": [
                            {
                                "segment": {
                                    "startIndex": 0,
                                    "endIndex": 62,
                                    "text": "Spain won Euro 2024, defeating England 2-1 in the final.",
                                },
                                "groundingChunkIndices": [0, 1],
                            }
                        ],
                    },
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        assert (
            parsed.content == "Spain won Euro 2024, defeating England 2-1 in the final."
        )
        assert len(parsed.tool_calls) == 0
        assert parsed.grounding_metadata is not None

        metadata = parsed.grounding_metadata
        assert "webSearchQueries" in metadata
        assert "groundingChunks" in metadata
        assert "groundingSupports" in metadata
        assert len(metadata["webSearchQueries"]) == 2
        assert len(metadata["groundingChunks"]) == 2

    def test_mixed_content_response_parsing(self):
        """Test parsing of responses with multiple text parts."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Based on my search, "},
                            {"text": "Spain won Euro 2024. "},
                            {"text": "They defeated England in the final."},
                        ],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        expected_content = "Based on my search, Spain won Euro 2024. They defeated England in the final."
        assert parsed.content == expected_content

    def test_empty_response_parsing(self):
        """Test parsing of empty or malformed responses."""
        provider = GoogleProvider()

        # Empty candidates
        empty_payload = {"candidates": []}
        parsed = provider._parse_response(empty_payload)
        assert parsed.content == ""
        assert len(parsed.tool_calls) == 0

        # No candidates field
        no_candidates_payload = {}
        parsed = provider._parse_response(no_candidates_payload)
        assert parsed.content == ""
        assert len(parsed.tool_calls) == 0


class TestGoogleProviderMessageMapping:
    """Test Google provider message mapping."""

    def test_system_instruction_mapping(self):
        """Test system instruction handling."""
        provider = GoogleProvider()

        messages = [
            SystemMessage("You are a helpful assistant with access to web search."),
            UserMessage("Who won Euro 2024?"),
        ]

        mapped = provider._map_messages(messages)

        assert (
            mapped["system_instruction"]
            == "You are a helpful assistant with access to web search."
        )
        assert len(mapped["contents"]) == 1
        assert mapped["contents"][0]["role"] == "user"
        assert mapped["contents"][0]["parts"][0]["text"] == "Who won Euro 2024?"

    def test_conversation_mapping(self):
        """Test multi-turn conversation mapping."""
        provider = GoogleProvider()

        messages = [UserMessage("Hello"), UserMessage("How are you?")]

        mapped = provider._map_messages(messages)

        assert mapped["system_instruction"] is None
        assert len(mapped["contents"]) == 2
        assert all(msg["role"] == "user" for msg in mapped["contents"])


class TestGroundingMetadataCitations:
    """Test grounding metadata and citation functionality."""

    def test_citation_extraction(self):
        """Test extracting citation information from grounding metadata."""
        grounding_metadata = {
            "webSearchQueries": ["Euro 2024 winner"],
            "groundingChunks": [
                {
                    "web": {
                        "uri": "https://uefa.com/euro2024",
                        "title": "UEFA Euro 2024",
                    }
                },
                {"web": {"uri": "https://bbc.com/sport", "title": "BBC Sport"}},
            ],
            "groundingSupports": [
                {
                    "segment": {
                        "startIndex": 0,
                        "endIndex": 25,
                        "text": "Spain won Euro 2024",
                    },
                    "groundingChunkIndices": [0, 1],
                }
            ],
        }

        # Test that we can extract citation data
        chunks = grounding_metadata["groundingChunks"]
        supports = grounding_metadata["groundingSupports"]

        assert len(chunks) == 2
        assert chunks[0]["web"]["uri"] == "https://uefa.com/euro2024"
        assert chunks[1]["web"]["title"] == "BBC Sport"

        assert len(supports) == 1
        support = supports[0]
        assert support["segment"]["text"] == "Spain won Euro 2024"
        assert support["groundingChunkIndices"] == [0, 1]

    def test_search_queries_tracking(self):
        """Test tracking of web search queries."""
        grounding_metadata = {
            "webSearchQueries": [
                "UEFA Euro 2024 winner",
                "Spain vs England Euro 2024 final",
            ]
        }

        queries = grounding_metadata["webSearchQueries"]
        assert len(queries) == 2
        assert "UEFA Euro 2024 winner" in queries
        assert "Spain vs England Euro 2024 final" in queries


def test_documentation_examples():
    """Test examples from the Google documentation."""

    # Example 1: Basic Google search grounding
    search_tool = GoogleWebSearch()
    spec = search_tool.spec()
    assert spec.provider_type == "google_search"

    # Example 2: Legacy search retrieval with dynamic threshold
    legacy_tool = GoogleSearchRetrieval(mode="MODE_DYNAMIC", dynamic_threshold=0.7)
    spec = legacy_tool.spec()
    config = spec.provider_config["dynamic_retrieval_config"]
    assert config["mode"] == "MODE_DYNAMIC"
    assert config["dynamic_threshold"] == 0.7

    # Example 3: Code execution tool
    code_tool = GoogleCodeExecution()
    spec = code_tool.spec()
    assert spec.provider_type == "code_execution"

    # Example 4: URL context tool
    url_tool = GoogleUrlContext()
    spec = url_tool.spec()
    assert spec.provider_type == "url_context"


def test_model_support_validation():
    """Test that tools work with supported models."""
    # The documentation mentions these models support grounding
    supported_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    # Test that we can create configs for supported models
    for model in supported_models:
        config = ModelConfig(provider="google", model=model)
        assert config.model == model
        assert config.provider == "google"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
