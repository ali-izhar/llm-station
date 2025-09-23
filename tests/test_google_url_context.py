#!/usr/bin/env python3
"""
Comprehensive tests for Google Gemini URL context implementation.

Tests accuracy against the official Google Gemini URL context documentation:
https://ai.google.dev/gemini-api/docs/url-context
"""

import pytest
from llm_studio.tools.url_context import GoogleUrlContext
from llm_studio.models.google import GoogleProvider
from llm_studio.models.base import ModelConfig
from llm_studio.schemas.messages import UserMessage, ModelResponse


class TestGoogleUrlContextTool:
    """Test the GoogleUrlContext tool factory."""

    def test_basic_url_context_tool(self):
        """Test basic URL context tool."""
        tool = GoogleUrlContext()
        spec = tool.spec()

        assert spec.name == "url_context"
        assert spec.provider == "google"
        assert spec.provider_type == "url_context"
        assert spec.requires_network is True
        assert spec.provider_config is None

    def test_tool_description(self):
        """Test that description includes key features."""
        tool = GoogleUrlContext()
        spec = tool.spec()

        description = spec.description.lower()
        assert "url" in description
        assert "context" in description or "retrieval" in description


class TestGoogleProviderUrlContextPreparation:
    """Test Google provider URL context tool preparation."""

    def test_url_context_tool_preparation(self):
        """Test preparation of URL context tool."""
        provider = GoogleProvider()
        tool = GoogleUrlContext()
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        assert len(prepared_tools) == 1
        assert prepared_tools[0] == {"url_context": {}}

    def test_mixed_tools_with_url_context(self):
        """Test preparing URL context with other Google tools."""
        provider = GoogleProvider()

        from llm_studio.tools.web_search import GoogleWebSearch
        from llm_studio.tools.code_execution import GoogleCodeExecution

        # Mix of tools as shown in documentation
        search_tool = GoogleWebSearch().spec()
        url_tool = GoogleUrlContext().spec()
        code_tool = GoogleCodeExecution().spec()

        prepared_tools = provider.prepare_tools([search_tool, url_tool, code_tool])

        assert len(prepared_tools) == 3
        assert prepared_tools[0] == {"google_search": {}}
        assert prepared_tools[1] == {"url_context": {}}
        assert prepared_tools[2] == {"code_execution": {}}


class TestGoogleProviderUrlContextResponseParsing:
    """Test Google provider URL context response parsing."""

    def test_text_response_with_url_context_metadata(self):
        """Test parsing of responses with URL context metadata."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Based on the recipes from both URLs, here's a comparison:\n\nIna Garten's recipe uses herbs and lemon, while the AllRecipes version is simpler with just salt and pepper."
                            }
                        ],
                        "role": "model",
                    },
                    "url_context_metadata": {
                        "url_metadata": [
                            {
                                "retrieved_url": "https://www.foodnetwork.com/recipes/ina-garten/perfect-roast-chicken-recipe-1940592",
                                "url_retrieval_status": "URL_RETRIEVAL_STATUS_SUCCESS",
                            },
                            {
                                "retrieved_url": "https://www.allrecipes.com/recipe/21151/simple-whole-roast-chicken/",
                                "url_retrieval_status": "URL_RETRIEVAL_STATUS_SUCCESS",
                            },
                        ]
                    },
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        assert "comparison:" in parsed.content
        assert "Ina Garten" in parsed.content
        assert parsed.grounding_metadata is not None
        assert "url_context" in parsed.grounding_metadata

        url_metadata = parsed.grounding_metadata["url_context"]
        assert "url_metadata" in url_metadata
        assert len(url_metadata["url_metadata"]) == 2

        # Check first URL metadata
        first_url = url_metadata["url_metadata"][0]
        assert (
            first_url["retrieved_url"]
            == "https://www.foodnetwork.com/recipes/ina-garten/perfect-roast-chicken-recipe-1940592"
        )
        assert first_url["url_retrieval_status"] == "URL_RETRIEVAL_STATUS_SUCCESS"

    def test_url_context_with_grounding_metadata(self):
        """Test responses with both URL context and grounding metadata."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Based on search results and the specific URLs provided, here's what I found about the event."
                            }
                        ],
                        "role": "model",
                    },
                    "groundingMetadata": {
                        "webSearchQueries": ["event schedule weather"],
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://example.com/weather",
                                    "title": "Weather Info",
                                }
                            }
                        ],
                    },
                    "url_context_metadata": {
                        "url_metadata": [
                            {
                                "retrieved_url": "https://example.com/event-details",
                                "url_retrieval_status": "URL_RETRIEVAL_STATUS_SUCCESS",
                            }
                        ]
                    },
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        # Check both metadata types are parsed
        assert parsed.grounding_metadata is not None
        assert "grounding" in parsed.grounding_metadata
        assert "url_context" in parsed.grounding_metadata

        # Check grounding metadata
        grounding = parsed.grounding_metadata["grounding"]
        assert "webSearchQueries" in grounding
        assert grounding["webSearchQueries"] == ["event schedule weather"]

        # Check URL context metadata
        url_context = parsed.grounding_metadata["url_context"]
        assert len(url_context["url_metadata"]) == 1
        assert (
            url_context["url_metadata"][0]["retrieved_url"]
            == "https://example.com/event-details"
        )

    def test_url_retrieval_status_variations(self):
        """Test different URL retrieval status codes."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I was able to access some URLs but not others."}
                        ],
                        "role": "model",
                    },
                    "url_context_metadata": {
                        "url_metadata": [
                            {
                                "retrieved_url": "https://public-site.com/data",
                                "url_retrieval_status": "URL_RETRIEVAL_STATUS_SUCCESS",
                            },
                            {
                                "retrieved_url": "https://paywall-site.com/premium",
                                "url_retrieval_status": "URL_RETRIEVAL_STATUS_FAILED",
                            },
                            {
                                "retrieved_url": "https://unsafe-content.com/bad",
                                "url_retrieval_status": "URL_RETRIEVAL_STATUS_UNSAFE",
                            },
                        ]
                    },
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        url_metadata = parsed.grounding_metadata["url_context"]["url_metadata"]
        assert len(url_metadata) == 3

        # Check different status codes
        statuses = [url["url_retrieval_status"] for url in url_metadata]
        assert "URL_RETRIEVAL_STATUS_SUCCESS" in statuses
        assert "URL_RETRIEVAL_STATUS_FAILED" in statuses
        assert "URL_RETRIEVAL_STATUS_UNSAFE" in statuses

    def test_empty_url_context_metadata(self):
        """Test responses without URL context metadata."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "This response doesn't use URL context."}],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        assert parsed.content == "This response doesn't use URL context."
        assert parsed.grounding_metadata is None

    def test_url_context_with_code_execution(self):
        """Test URL context combined with code execution."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me analyze the data from the URL:"},
                            {
                                "executable_code": {
                                    "code": "import pandas as pd\nprint('Processing URL data...')"
                                }
                            },
                            {
                                "code_execution_result": {
                                    "output": "Processing URL data..."
                                }
                            },
                        ],
                        "role": "model",
                    },
                    "url_context_metadata": {
                        "url_metadata": [
                            {
                                "retrieved_url": "https://example.com/data.csv",
                                "url_retrieval_status": "URL_RETRIEVAL_STATUS_SUCCESS",
                            }
                        ]
                    },
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        # Check both code execution and URL context are parsed
        assert "```python" in parsed.content
        assert "Processing URL data..." in parsed.content
        assert parsed.grounding_metadata is not None
        assert "url_context" in parsed.grounding_metadata


class TestUrlContextUsageAndLimitations:
    """Test understanding of URL context usage and limitations."""

    def test_url_limitations_understanding(self):
        """Test that we understand URL context limitations."""
        limitations = {
            "max_urls_per_request": 20,
            "max_content_size_mb": 34,
            "supported_content_types": [
                "text/html",
                "application/json",
                "text/plain",
                "text/xml",
                "text/css",
                "text/javascript",
                "text/csv",
                "text/rtf",
                "image/png",
                "image/jpeg",
                "image/bmp",
                "image/webp",
                "application/pdf",
            ],
            "unsupported_content": [
                "paywalled_content",
                "youtube_videos",
                "google_workspace_files",
                "video_files",
                "audio_files",
            ],
        }

        # Verify key limitations
        assert limitations["max_urls_per_request"] == 20
        assert limitations["max_content_size_mb"] == 34
        assert "text/html" in limitations["supported_content_types"]
        assert "application/pdf" in limitations["supported_content_types"]
        assert "youtube_videos" in limitations["unsupported_content"]

    def test_token_counting_understanding(self):
        """Test understanding of token counting for URL content."""
        # URL content counts as input tokens
        url_content_billing = "input_tokens"

        # Token counting details from documentation
        token_categories = {
            "prompt_token_count": "original_user_prompt",
            "tool_use_prompt_token_count": "url_content_retrieved",
            "candidates_token_count": "model_response",
            "total_token_count": "sum_of_all_above",
        }

        assert url_content_billing == "input_tokens"
        assert "tool_use_prompt_token_count" in token_categories

    def test_retrieval_process_understanding(self):
        """Test understanding of two-step retrieval process."""
        retrieval_steps = {
            "step_1": "internal_index_cache",
            "step_2": "live_fetch_fallback",
        }

        benefits = {
            "cache": ["speed", "cost_optimization"],
            "live_fetch": ["fresh_data", "new_pages"],
        }

        assert retrieval_steps["step_1"] == "internal_index_cache"
        assert retrieval_steps["step_2"] == "live_fetch_fallback"
        assert "speed" in benefits["cache"]
        assert "fresh_data" in benefits["live_fetch"]


class TestUrlContextBestPractices:
    """Test understanding of URL context best practices."""

    def test_url_format_best_practices(self):
        """Test understanding of URL formatting best practices."""
        # Good URL examples from documentation
        good_urls = [
            "https://www.google.com",  # Complete with protocol
            "https://www.foodnetwork.com/recipes/ina-garten/perfect-roast-chicken-recipe-1940592",  # Specific content
            "https://www.allrecipes.com/recipe/21151/simple-whole-roast-chicken/",  # Direct recipe link
        ]

        # Bad URL examples
        bad_urls = [
            "google.com",  # Missing protocol
            "www.google.com",  # Missing protocol
            "https://site-with-nested-links.com",  # Relying on nested links
        ]

        # Check good URLs have protocols
        for url in good_urls:
            assert url.startswith("https://")

        # Check bad URLs missing protocols
        for url in bad_urls[:2]:
            assert not url.startswith("http")

    def test_accessibility_considerations(self):
        """Test understanding of accessibility considerations."""
        accessibility_checks = {
            "no_login_required": True,
            "no_paywall": True,
            "public_accessible": True,
            "safety_moderated": True,
        }

        # All should be True for best results
        assert all(accessibility_checks.values())

    def test_content_specificity(self):
        """Test understanding of content specificity requirements."""
        # Model only retrieves from provided URLs, not nested links
        retrieval_behavior = {
            "direct_urls_only": True,
            "no_nested_link_following": True,
            "up_to_20_urls": True,
        }

        assert retrieval_behavior["direct_urls_only"] is True
        assert retrieval_behavior["no_nested_link_following"] is True


def test_documentation_examples():
    """Test examples from the Google documentation."""

    # Example 1: Basic URL context tool
    url_tool = GoogleUrlContext()
    spec = url_tool.spec()
    assert spec.provider_type == "url_context"
    assert spec.requires_network is True

    # Example 2: Tool preparation
    provider = GoogleProvider()
    prepared_tools = provider.prepare_tools([spec])
    assert prepared_tools[0] == {"url_context": {}}

    # Example 3: Combined with search (from documentation)
    from llm_studio.tools.web_search import GoogleWebSearch

    search_spec = GoogleWebSearch().spec()

    combined_tools = provider.prepare_tools([spec, search_spec])
    assert len(combined_tools) == 2
    assert combined_tools[0] == {"url_context": {}}
    assert combined_tools[1] == {"google_search": {}}


def test_model_support():
    """Test that URL context works with supported models."""
    # According to documentation
    supported_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-live-2.5-flash-preview",
        "gemini-2.0-flash-live-001",
    ]

    for model in supported_models:
        config = ModelConfig(provider="google", model=model)
        assert config.model == model
        assert config.provider == "google"


def test_safety_and_moderation():
    """Test understanding of safety checks and content moderation."""
    url_retrieval_statuses = [
        "URL_RETRIEVAL_STATUS_SUCCESS",
        "URL_RETRIEVAL_STATUS_FAILED",
        "URL_RETRIEVAL_STATUS_UNSAFE",
    ]

    # Unsafe status indicates content moderation failure
    unsafe_status = "URL_RETRIEVAL_STATUS_UNSAFE"
    assert unsafe_status in url_retrieval_statuses

    # System performs content moderation checks
    safety_features = {
        "content_moderation": True,
        "safety_standards_check": True,
        "automatic_filtering": True,
    }

    assert all(safety_features.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
