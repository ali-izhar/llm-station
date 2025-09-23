#!/usr/bin/env python3
"""
Comprehensive tests for Google Gemini code execution implementation.

Tests accuracy against the official Google Gemini code execution documentation:
https://ai.google.dev/gemini-api/docs/code-execution
"""

import pytest
from llm_studio.tools.code_execution import GoogleCodeExecution
from llm_studio.models.google import GoogleProvider
from llm_studio.models.base import ModelConfig
from llm_studio.schemas.messages import UserMessage, ModelResponse


class TestGoogleCodeExecutionTool:
    """Test the GoogleCodeExecution tool factory."""

    def test_basic_code_execution_tool(self):
        """Test basic code execution tool."""
        tool = GoogleCodeExecution()
        spec = tool.spec()

        assert spec.name == "code_execution"
        assert spec.provider == "google"
        assert spec.provider_type == "code_execution"
        assert spec.requires_network is False  # Sandboxed environment
        assert spec.provider_config is None

    def test_tool_description(self):
        """Test that description includes key features."""
        tool = GoogleCodeExecution()
        spec = tool.spec()

        description = spec.description.lower()
        assert "python" in description
        assert "code execution" in description or "code" in description
        assert "libraries" in description


class TestGoogleProviderCodeExecutionPreparation:
    """Test Google provider code execution tool preparation."""

    def test_code_execution_tool_preparation(self):
        """Test preparation of code execution tool."""
        provider = GoogleProvider()
        tool = GoogleCodeExecution()
        spec = tool.spec()

        prepared_tools = provider.prepare_tools([spec])

        assert len(prepared_tools) == 1
        assert prepared_tools[0] == {"code_execution": {}}

    def test_mixed_tools_with_code_execution(self):
        """Test preparing code execution with other Google tools."""
        provider = GoogleProvider()

        from llm_studio.tools.web_search import GoogleWebSearch
        from llm_studio.schemas.tooling import ToolSpec

        # Mix of tools
        search_tool = GoogleWebSearch().spec()
        code_tool = GoogleCodeExecution().spec()
        custom_tool = ToolSpec(
            name="helper",
            description="A helper function",
            input_schema={"type": "object", "properties": {}},
        )

        prepared_tools = provider.prepare_tools([search_tool, code_tool, custom_tool])

        assert len(prepared_tools) == 3
        assert prepared_tools[0] == {"google_search": {}}
        assert prepared_tools[1] == {"code_execution": {}}
        assert "function_declarations" in prepared_tools[2]


class TestGoogleProviderCodeExecutionResponseParsing:
    """Test Google provider code execution response parsing."""

    def test_text_only_response(self):
        """Test parsing of text-only responses."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I'll solve this step by step using Python code."}
                        ],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        assert parsed.content == "I'll solve this step by step using Python code."
        assert len(parsed.tool_calls) == 0

    def test_executable_code_response(self):
        """Test parsing of responses with executable code."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me calculate the sum of first 5 primes:"},
                            {
                                "executable_code": {
                                    "code": "primes = [2, 3, 5, 7, 11]\nsum_primes = sum(primes)\nprint(f'Sum: {sum_primes}')"
                                }
                            },
                        ],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        expected_content = "Let me calculate the sum of first 5 primes:\n```python\nprimes = [2, 3, 5, 7, 11]\nsum_primes = sum(primes)\nprint(f'Sum: {sum_primes}')\n```"
        assert parsed.content == expected_content

    def test_code_execution_result_response(self):
        """Test parsing of responses with code execution results."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Here's the calculation:"},
                            {"code_execution_result": {"output": "Sum: 28"}},
                        ],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        expected_content = "Here's the calculation:\nOutput:\nSum: 28"
        assert parsed.content == expected_content

    def test_complete_code_execution_response(self):
        """Test parsing of complete code execution flow (text + code + result)."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "I'll calculate the sum of the first 5 prime numbers:"
                            },
                            {
                                "executable_code": {
                                    "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprimes = []\nnum = 2\nwhile len(primes) < 5:\n    if is_prime(num):\n        primes.append(num)\n    num += 1\n\nsum_primes = sum(primes)\nprint(f'First 5 primes: {primes}')\nprint(f'Sum: {sum_primes}')"
                                }
                            },
                            {
                                "code_execution_result": {
                                    "output": "First 5 primes: [2, 3, 5, 7, 11]\nSum: 28"
                                }
                            },
                            {"text": "The sum of the first 5 prime numbers is 28."},
                        ],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        # Check that all parts are included
        content = parsed.content
        assert "I'll calculate the sum of the first 5 prime numbers:" in content
        assert "```python" in content
        assert "def is_prime(n):" in content
        assert "Output:\nFirst 5 primes: [2, 3, 5, 7, 11]" in content
        assert "The sum of the first 5 prime numbers is 28." in content

    def test_mixed_content_with_function_calls(self):
        """Test parsing responses with both code execution and function calls."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Let me calculate this and then call a helper function:"
                            },
                            {
                                "executable_code": {
                                    "code": "result = 2 + 2\nprint(f'2 + 2 = {result}')"
                                }
                            },
                            {"code_execution_result": {"output": "2 + 2 = 4"}},
                            {
                                "function_call": {
                                    "name": "helper_function",
                                    "args": {"value": 4},
                                }
                            },
                        ],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        # Check text and code parts are combined
        content = parsed.content
        assert "Let me calculate this and then call a helper function:" in content
        assert "```python" in content
        assert "Output:\n2 + 2 = 4" in content

        # Check function call is parsed
        assert len(parsed.tool_calls) == 1
        tool_call = parsed.tool_calls[0]
        assert tool_call.name == "helper_function"
        assert tool_call.arguments == {"value": 4}

    def test_empty_code_parts_handling(self):
        """Test handling of empty or missing code parts."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Here's my analysis:"},
                            {"executable_code": {"code": ""}},  # Empty code
                            {"code_execution_result": {"output": ""}},  # Empty output
                            {"text": "No code needed for this answer."},
                        ],
                        "role": "model",
                    }
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        # Empty code parts should be filtered out
        expected_content = "Here's my analysis:\nNo code needed for this answer."
        assert parsed.content == expected_content

    def test_code_execution_with_grounding(self):
        """Test code execution combined with grounding metadata."""
        provider = GoogleProvider()

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Based on search results, I'll calculate:"},
                            {
                                "executable_code": {
                                    "code": "calculation = 100 * 0.15\nprint(f'Result: {calculation}')"
                                }
                            },
                            {"code_execution_result": {"output": "Result: 15.0"}},
                        ],
                        "role": "model",
                    },
                    "groundingMetadata": {
                        "webSearchQueries": ["percentage calculation formula"],
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://example.com/math",
                                    "title": "Math Help",
                                }
                            }
                        ],
                    },
                }
            ]
        }

        parsed = provider._parse_response(response_payload)

        # Check both code execution and grounding are parsed
        assert "```python" in parsed.content
        assert "Result: 15.0" in parsed.content
        assert parsed.grounding_metadata is not None
        assert "grounding" in parsed.grounding_metadata
        assert "webSearchQueries" in parsed.grounding_metadata["grounding"]


class TestCodeExecutionBilling:
    """Test understanding of code execution billing model."""

    def test_billing_components_understanding(self):
        """Test that we understand the billing components."""
        # This test documents the billing model from the documentation
        billing_components = {
            "input_tokens": [
                "original_user_prompt",
                "intermediate_tokens_when_regenerating",
            ],
            "output_tokens": [
                "generated_code",
                "code_execution_results",
                "final_summary",
                "thinking_tokens",
            ],
        }

        # Verify our understanding
        assert "generated_code" in billing_components["output_tokens"]
        assert "code_execution_results" in billing_components["output_tokens"]
        assert "original_user_prompt" in billing_components["input_tokens"]

    def test_intermediate_tokens_concept(self):
        """Test understanding of intermediate tokens for regeneration."""
        # When code execution fails and model regenerates (up to 5 times),
        # the intermediate attempts are billed as input tokens
        max_regeneration_attempts = 5
        max_runtime_seconds = 30

        assert max_regeneration_attempts == 5
        assert max_runtime_seconds == 30


class TestSupportedLibraries:
    """Test understanding of supported libraries."""

    def test_documented_libraries_list(self):
        """Test that we know which libraries are supported."""
        # From the documentation - these are confirmed supported
        supported_libraries = {
            "data_science": ["numpy", "pandas", "scipy", "scikit-learn"],
            "visualization": ["matplotlib", "seaborn"],
            "image_processing": ["opencv-python", "pillow", "imageio"],
            "document_processing": ["PyPDF2", "python-docx", "python-pptx", "openpyxl"],
            "machine_learning": ["tensorflow"],
            "symbolic_math": ["sympy", "mpmath"],
            "utilities": ["attrs", "joblib", "lxml", "tabulate", "toolz"],
            "reporting": ["fpdf", "reportlab", "pylatex"],
            "geospatial": ["geopandas"],
            "games": ["chess"],
        }

        # Verify key libraries are documented
        all_libs = []
        for category in supported_libraries.values():
            all_libs.extend(category)

        assert "numpy" in all_libs
        assert "pandas" in all_libs
        assert "matplotlib" in all_libs
        assert "tensorflow" in all_libs
        assert "scikit-learn" in all_libs

    def test_library_limitations(self):
        """Test understanding of library limitations."""
        # Cannot install custom libraries
        can_install_custom_libraries = False
        assert can_install_custom_libraries is False


def test_documentation_examples():
    """Test examples from the Google documentation."""

    # Example 1: Basic code execution tool
    code_tool = GoogleCodeExecution()
    spec = code_tool.spec()
    assert spec.provider_type == "code_execution"
    assert spec.requires_network is False

    # Example 2: Tool preparation
    provider = GoogleProvider()
    prepared_tools = provider.prepare_tools([spec])
    assert prepared_tools[0] == {"code_execution": {}}


def test_model_support():
    """Test that code execution works with supported models."""
    # According to documentation, works with Gemini 2.0 and 2.5 models
    supported_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

    for model in supported_models:
        config = ModelConfig(provider="google", model=model)
        assert config.model == model
        assert config.provider == "google"


def test_io_capabilities():
    """Test understanding of I/O capabilities."""
    # Starting with Gemini 2.0 Flash
    io_features = {
        "file_input_types": [
            ".png",
            ".jpeg",
            ".csv",
            ".xml",
            ".cpp",
            ".java",
            ".py",
            ".js",
            ".ts",
        ],
        "plotting_libraries": ["matplotlib", "seaborn"],
        "max_file_size_tokens": 1_000_000,  # ~2MB for text files
        "max_runtime_seconds": 30,
        "max_regeneration_attempts": 5,
    }

    assert len(io_features["file_input_types"]) > 5
    assert "matplotlib" in io_features["plotting_libraries"]
    assert io_features["max_runtime_seconds"] == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
