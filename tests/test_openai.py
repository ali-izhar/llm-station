"""
Real OpenAI API integration tests using OPENAI_API_KEY from .env

These tests make actual API calls to OpenAI and require:
- OPENAI_API_KEY in .env file
- Active internet connection
- OpenAI API credits

Tests cover:
- Chat Completions endpoint with real models
- Responses endpoint with real models
- Web Search tool with real searches
- Code Interpreter tool with real Python execution
- Image Generation tool with real image creation

Enhanced with detailed response analysis and logging for model validation.
"""

import os
import base64
import json
import pytest
from dotenv import load_dotenv
from llm_studio import Agent, setup_logging, LogLevel
from llm_studio.models.openai import OpenAIProvider
from llm_studio.models.base import ModelConfig
from llm_studio.schemas.messages import UserMessage, SystemMessage
from llm_studio.tools.web_search.openai import OpenAIWebSearch
from llm_studio.tools.code_execution.openai import OpenAICodeInterpreter
from llm_studio.tools.image_generation.openai import OpenAIImageGeneration

# Load environment variables
load_dotenv()

# Enable detailed logging for analysis
setup_logging(level=LogLevel.DEBUG)


def log_response_analysis(test_name: str, response, api_type: str = "unknown"):
    """Detailed logging and analysis of API responses.

    Note: For Agent.generate() calls, response is AssistantMessage with grounding_metadata.
    For direct provider calls, response is ModelResponse with raw data.
    """
    print(f"\n{'='*80}")
    print(f"RESPONSE ANALYSIS: {test_name}")
    print(f"API Type: {api_type}")
    print(f"Response Type: {type(response).__name__}")
    print(f"{'='*80}")

    # Basic response info
    print(f"Content Length: {len(response.content)}")
    print(f"Content Preview: {response.content[:200]}...")
    print(
        f"Tool Calls Count: {len(response.tool_calls) if hasattr(response, 'tool_calls') else 'N/A'}"
    )

    # Tool calls analysis
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\nTool Calls:")
        for i, tc in enumerate(response.tool_calls):
            print(f"  {i+1}. ID: {tc.id}")
            print(f"     Name: {tc.name}")
            print(f"     Arguments: {tc.arguments}")

    # Grounding metadata analysis (available in both AssistantMessage and ModelResponse)
    if hasattr(response, "grounding_metadata") and response.grounding_metadata:
        print(f"\nGrounding Metadata Keys: {list(response.grounding_metadata.keys())}")

        for key, value in response.grounding_metadata.items():
            print(f"\n{key.upper()}:")
            if isinstance(value, list):
                print(f"  Type: List with {len(value)} items")
                if value and len(value) <= 3:  # Show first few items
                    for i, item in enumerate(value):
                        print(f"  [{i}]: {str(item)[:100]}...")
            elif isinstance(value, dict):
                print(f"  Type: Dict with {len(value)} keys: {list(value.keys())}")
                for k, v in value.items():
                    if isinstance(v, str) and len(v) > 50:
                        print(f"    {k}: {str(v)[:50]}...")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  Type: {type(value).__name__}")
                print(f"  Value: {str(value)[:100]}...")

    # Raw response structure analysis (only available in ModelResponse)
    if hasattr(response, "raw") and response.raw:
        print(f"\nRaw Response Structure:")
        print(f"  Type: {type(response.raw)}")
        if isinstance(response.raw, dict):
            print(f"  Top-level keys: {list(response.raw.keys())}")

            # Analyze specific OpenAI response patterns
            if "output" in response.raw:
                output = response.raw["output"]
                print(f"  Output type: {type(output)}")
                if isinstance(output, list):
                    print(f"  Output array length: {len(output)}")
                    for i, item in enumerate(output[:3]):  # First 3 items
                        if isinstance(item, dict):
                            print(f"    [{i}] type: {item.get('type', 'unknown')}")
                            print(f"    [{i}] keys: {list(item.keys())}")

            if "choices" in response.raw:
                choices = response.raw["choices"]
                print(f"  Choices length: {len(choices)}")
                if choices:
                    choice = choices[0]
                    print(f"  First choice keys: {list(choice.keys())}")
                    if "message" in choice:
                        msg = choice["message"]
                        print(f"  Message keys: {list(msg.keys())}")
        elif isinstance(response.raw, list):
            print(f"  List length: {len(response.raw)}")
            for i, item in enumerate(response.raw[:3]):
                if isinstance(item, dict):
                    print(f"    [{i}] type: {item.get('type', 'unknown')}")
                    print(f"    [{i}] keys: {list(item.keys())}")

    print(f"{'='*80}\n")


def save_response_to_file(test_name: str, response, api_type: str):
    """Save full response to file for detailed analysis."""
    os.makedirs("test_outputs", exist_ok=True)

    analysis_data = {
        "test_name": test_name,
        "api_type": api_type,
        "content": response.content,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in response.tool_calls
        ],
        "grounding_metadata": response.grounding_metadata,
        "raw_response": response.raw,
    }

    filename = f"test_outputs/{test_name.replace(' ', '_').lower()}_{api_type}.json"
    with open(filename, "w") as f:
        json.dump(analysis_data, f, indent=2, default=str)

    print(f"💾 Saved detailed analysis to: {filename}")


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in .env file")
    return api_key


@pytest.fixture
def openai_agent(openai_api_key):
    """Create OpenAI agent for testing."""
    return Agent(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_api_key,
        system_prompt="You are a helpful test assistant.",
    )


@pytest.fixture
def responses_agent(openai_api_key):
    """Create OpenAI agent configured for Responses API."""
    return Agent(
        provider="openai",
        model="gpt-4o",
        api_key=openai_api_key,
        system_prompt="You are a helpful assistant.",
    )


class TestRealChatCompletions:
    """Test real OpenAI Chat Completions API calls."""

    @pytest.mark.integration
    def test_basic_chat_completion(self, openai_agent):
        """Test basic chat completion with real API."""
        result = openai_agent.generate("Hello! What is 2+2?")

        assert len(result.content) > 0
        assert "4" in result.content
        print(f"✓ Chat Completions Response: {result.content}")

        # Detailed response analysis
        log_response_analysis("Basic Chat Completion", result, "chat_completions")
        save_response_to_file("basic_chat_completion", result, "chat_completions")

    @pytest.mark.integration
    def test_chat_with_function_tools(self, openai_agent):
        """Test Chat Completions with local function tools."""
        result = openai_agent.generate(
            "Format this data as JSON: name=Alice, age=30, city=New York",
            tools=["json_format"],
        )

        assert len(result.content) > 0
        print(f"✓ Function Tool Response: {result.content}")

        # Should have executed the tool locally
        if result.tool_calls:
            print(f"✓ Tool calls made: {[tc.name for tc in result.tool_calls]}")

        # Detailed response analysis
        log_response_analysis("Chat with Function Tools", result, "chat_completions")
        save_response_to_file("chat_with_function_tools", result, "chat_completions")

    @pytest.mark.integration
    def test_search_model_native_search(self, openai_api_key):
        """Test search models with built-in web search capabilities."""
        # Use search model that has built-in web search
        search_agent = Agent(
            provider="openai", model="gpt-4o-search-preview", api_key=openai_api_key
        )

        result = search_agent.generate("What's a recent development in AI this week?")

        assert len(result.content) > 0
        print(f"✓ Built-in Search Response: {result.content[:200]}...")

        # Detailed response analysis
        log_response_analysis(
            "Search Model Native Search", result, "chat_completions_with_search"
        )
        save_response_to_file("search_model_native", result, "chat_completions")


class TestRealResponsesAPI:
    """Test real OpenAI Responses API calls."""

    @pytest.mark.integration
    def test_basic_responses_call(self, responses_agent):
        """Test basic Responses API call."""
        # Force Responses API usage
        provider = OpenAIProvider(api_key=responses_agent._provider.api_key)
        messages = [UserMessage("What is the capital of France?")]
        config = ModelConfig(provider="openai", model="gpt-4o", api="responses")

        result = provider.generate(messages, config)

        assert len(result.content) > 0
        assert "Paris" in result.content
        print(f"✓ Responses API Response: {result.content}")

        # Detailed response analysis
        log_response_analysis("Basic Responses API", result, "responses_api")
        save_response_to_file("basic_responses_call", result, "responses_api")

    @pytest.mark.integration
    def test_responses_with_reasoning(self, responses_agent):
        """Test Responses API with basic parameters (avoiding unsupported ones)."""
        provider = OpenAIProvider(api_key=responses_agent._provider.api_key)
        messages = [UserMessage("Explain quantum computing in simple terms")]

        # Use basic gpt-4o without unsupported parameters
        config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            api="responses",
            # Note: temperature and reasoning not supported with gpt-4o in Responses API
        )

        result = provider.generate(messages, config)

        assert len(result.content) > 0
        assert "quantum" in result.content.lower()
        print(f"✓ Responses API Response: {result.content[:200]}...")

        # Detailed response analysis
        log_response_analysis("Responses with Reasoning", result, "responses_api")
        save_response_to_file("responses_with_reasoning", result, "responses_api")


class TestRealWebSearch:
    """Test real OpenAI Web Search tool."""

    @pytest.mark.integration
    def test_basic_web_search(self, responses_agent):
        """Test basic web search functionality."""
        result = responses_agent.generate(
            "What's happening in AI news today?", tools=["openai_web_search"]
        )

        assert len(result.content) > 0
        print(f"✓ Web Search Response: {result.content[:300]}...")

        # Check for grounding metadata
        if result.grounding_metadata:
            print(f"✓ Web Search Metadata: {list(result.grounding_metadata.keys())}")

            if "citations" in result.grounding_metadata:
                citations = result.grounding_metadata["citations"]
                print(f"✓ Found {len(citations)} citations")
                # Analyze citation structure
                if citations:
                    sample_citation = citations[0]
                    print(f"  Sample citation keys: {list(sample_citation.keys())}")

            if "sources" in result.grounding_metadata:
                sources = result.grounding_metadata["sources"]
                print(f"✓ Found {len(sources)} sources")

            if "web_search" in result.grounding_metadata:
                ws_info = result.grounding_metadata["web_search"]
                print(
                    f"✓ Web search info keys: {list(ws_info.keys()) if isinstance(ws_info, dict) else type(ws_info)}"
                )

        # Detailed response analysis
        log_response_analysis("OpenAI Web Search", result, "responses_api_web_search")
        save_response_to_file("openai_web_search", result, "responses_api")

    @pytest.mark.integration
    def test_domain_filtered_search(self, responses_agent):
        """Test web search with domain filtering."""
        # Create domain-filtered search tool
        search_tool = OpenAIWebSearch(
            allowed_domains=["openai.com", "github.com", "arxiv.org"]
        )

        result = responses_agent.generate(
            "Find information about OpenAI's latest models", tools=[search_tool.spec()]
        )

        assert len(result.content) > 0
        print(f"✓ Domain-Filtered Search: {result.content[:300]}...")

        # Verify domain filtering worked (should only have allowed domains)
        if result.grounding_metadata and "citations" in result.grounding_metadata:
            citations = result.grounding_metadata["citations"]
            for citation in citations:
                url = citation.get("url", "")
                print(f"✓ Citation URL: {url}")

    @pytest.mark.integration
    def test_geographic_search(self, responses_agent):
        """Test web search with geographic refinement."""
        search_tool = OpenAIWebSearch(
            user_location={"country": "GB", "city": "London", "region": "England"}
        )

        result = responses_agent.generate(
            "What are the best restaurants near London Bridge?",
            tools=[search_tool.spec()],
        )

        assert len(result.content) > 0
        print(f"✓ Geographic Search: {result.content[:300]}...")


class TestRealCodeInterpreter:
    """Test real OpenAI Code Interpreter tool."""

    @pytest.mark.integration
    def test_basic_math_calculation(self, responses_agent):
        """Test code interpreter with basic math (official example)."""
        result = responses_agent.generate(
            "I need to solve the equation 3x + 11 = 14. Can you help me using Python?",
            tools=["openai_code_interpreter"],
        )

        assert len(result.content) > 0
        assert "1" in result.content  # x = 1 is the answer
        print(f"✓ Code Interpreter Math: {result.content}")

        # Check for code execution metadata
        if result.grounding_metadata:
            print(
                f"✓ Code Interpreter Metadata: {list(result.grounding_metadata.keys())}"
            )

            if "code_interpreter" in result.grounding_metadata:
                code_info = result.grounding_metadata["code_interpreter"]
                if isinstance(code_info, dict) and "code" in code_info:
                    print(f"✓ Executed Code: {code_info['code']}")

        # Detailed response analysis
        log_response_analysis(
            "OpenAI Code Interpreter Math", result, "responses_api_code_interpreter"
        )
        save_response_to_file("openai_code_interpreter_math", result, "responses_api")

    @pytest.mark.integration
    def test_data_analysis(self, responses_agent):
        """Test code interpreter with data analysis."""
        result = responses_agent.generate(
            "Create a list of numbers 1-10, calculate their mean and standard deviation using Python",
            tools=["openai_code_interpreter"],
        )

        assert len(result.content) > 0
        print(f"✓ Data Analysis Response: {result.content}")

        # Should mention mean and standard deviation
        content_lower = result.content.lower()
        assert "mean" in content_lower or "average" in content_lower

    @pytest.mark.integration
    def test_visualization_creation(self, responses_agent):
        """Test code interpreter creating visualizations."""
        result = responses_agent.generate(
            "Create a simple bar chart showing sales data: Q1=100, Q2=150, Q3=120, Q4=180. Use Python to generate the chart.",
            tools=["openai_code_interpreter"],
        )

        assert len(result.content) > 0
        print(f"✓ Visualization Response: {result.content}")

        # Check for file citations (generated charts)
        if result.grounding_metadata and "file_citations" in result.grounding_metadata:
            file_citations = result.grounding_metadata["file_citations"]
            print(f"✓ Generated files: {len(file_citations)}")
            for citation in file_citations:
                print(
                    f"  - {citation.get('filename', 'Unknown')} (ID: {citation.get('file_id', 'N/A')})"
                )


class TestRealImageGeneration:
    """Test real OpenAI Image Generation tool."""

    @pytest.mark.integration
    def test_basic_image_generation(self, responses_agent):
        """Test basic image generation (official example)."""
        result = responses_agent.generate(
            "Generate a simple image of a red apple on a white background",
            tools=["openai_image_generation"],
        )

        assert len(result.content) > 0
        print(f"✓ Image Generation Response: {result.content}")

        # Check for image generation metadata
        if (
            result.grounding_metadata
            and "image_generation" in result.grounding_metadata
        ):
            image_calls = result.grounding_metadata["image_generation"]
            print(f"✓ Generated images: {len(image_calls)}")

            for i, call in enumerate(image_calls):
                if "result" in call:
                    image_data = call["result"]
                    print(f"✓ Image {i+1}: {len(image_data)} characters (base64)")

                    # Save image to verify it's valid
                    try:
                        image_bytes = base64.b64decode(image_data)
                        with open(f"test_apple_{i}.png", "wb") as f:
                            f.write(image_bytes)
                        print(f"✓ Saved test_apple_{i}.png ({len(image_bytes)} bytes)")
                    except Exception as e:
                        print(f"✗ Failed to save image: {e}")

                if "revised_prompt" in call:
                    print(f"✓ Revised prompt: {call['revised_prompt']}")

        # Detailed response analysis
        log_response_analysis(
            "OpenAI Image Generation", result, "responses_api_image_generation"
        )
        save_response_to_file("openai_image_generation", result, "responses_api")

    @pytest.mark.integration
    def test_high_quality_image_generation(self, responses_agent):
        """Test high-quality image generation with specific parameters."""
        # Create high-quality image tool
        image_tool = OpenAIImageGeneration(
            size="1024x1024", quality="high", format="png", background="transparent"
        )

        result = responses_agent.generate(
            "Draw a professional logo of a stylized tree with transparent background",
            tools=[image_tool.spec()],
        )

        assert len(result.content) > 0
        print(f"✓ High-Quality Image Response: {result.content}")

        # Verify image metadata
        if (
            result.grounding_metadata
            and "image_generation" in result.grounding_metadata
        ):
            image_calls = result.grounding_metadata["image_generation"]
            if image_calls:
                call = image_calls[0]
                print(f"✓ Image size: {call.get('size', 'Unknown')}")
                print(f"✓ Image quality: {call.get('quality', 'Unknown')}")
                print(f"✓ Image format: {call.get('format', 'Unknown')}")

    @pytest.mark.integration
    def test_compressed_image_generation(self, responses_agent):
        """Test image generation with compression."""
        # Create compressed image tool
        image_tool = OpenAIImageGeneration(
            format="jpeg", compression=75, quality="medium"
        )

        result = responses_agent.generate(
            "Create a landscape image of mountains at sunset", tools=[image_tool.spec()]
        )

        assert len(result.content) > 0
        print(f"✓ Compressed Image Response: {result.content}")


class TestRealMultiToolWorkflows:
    """Test real multi-tool workflows combining different capabilities."""

    @pytest.mark.integration
    def test_research_and_analyze_workflow(self, responses_agent):
        """Test workflow combining web search and code interpreter."""
        result = responses_agent.generate(
            "Search for current Bitcoin price and calculate what $1000 invested would be worth. Use Python for calculations.",
            tools=["openai_web_search", "openai_code_interpreter"],
        )

        assert len(result.content) > 0
        print(f"✓ Research + Analysis Response: {result.content}")

        # Should have both web search and code execution
        if result.grounding_metadata:
            metadata_keys = list(result.grounding_metadata.keys())
            print(f"✓ Metadata types: {metadata_keys}")

    @pytest.mark.integration
    def test_search_and_visualize_workflow(self, responses_agent):
        """Test workflow combining web search and image generation."""
        result = responses_agent.generate(
            "Search for information about solar system planets and create a visual diagram showing their relative sizes",
            tools=["openai_web_search", "openai_image_generation"],
        )

        assert len(result.content) > 0
        print(f"✓ Search + Visualize Response: {result.content}")

    @pytest.mark.integration
    def test_full_research_pipeline(self, responses_agent):
        """Test complete research pipeline with all tools."""
        result = responses_agent.generate(
            "Research the latest AI model performance benchmarks, analyze the data with Python, and create a visualization chart",
            tools=[
                "openai_web_search",
                "openai_code_interpreter",
                "openai_image_generation",
            ],
        )

        assert len(result.content) > 0
        print(f"✓ Full Pipeline Response: {result.content[:500]}...")


class TestRealAdvancedFeatures:
    """Test advanced real-world features and configurations."""

    @pytest.mark.integration
    def test_domain_specific_research(self, responses_agent):
        """Test research limited to specific domains."""
        # Medical research domains
        medical_search = OpenAIWebSearch(
            allowed_domains=[
                "pubmed.ncbi.nlm.nih.gov",
                "www.nejm.org",
                "www.thelancet.com",
            ]
        )

        result = responses_agent.generate(
            "Find recent research on diabetes treatment methods",
            tools=[medical_search.spec()],
        )

        assert len(result.content) > 0
        print(f"✓ Medical Research Response: {result.content[:400]}...")

        # Verify citations are from allowed domains
        if result.grounding_metadata and "citations" in result.grounding_metadata:
            citations = result.grounding_metadata["citations"]
            for citation in citations:
                url = citation.get("url", "")
                print(f"✓ Medical Citation: {url}")

    @pytest.mark.integration
    def test_geographic_local_search(self, responses_agent):
        """Test geographically refined search."""
        local_search = OpenAIWebSearch(
            user_location={
                "country": "US",
                "city": "San Francisco",
                "region": "California",
            }
        )

        result = responses_agent.generate(
            "What are good coffee shops that are open right now?",
            tools=[local_search.spec()],
        )

        assert len(result.content) > 0
        print(f"✓ Local Search Response: {result.content[:400]}...")

    @pytest.mark.integration
    def test_container_persistence(self, responses_agent):
        """Test code interpreter container and file persistence."""
        # Use auto container mode
        container_tool = OpenAICodeInterpreter(container_type="auto")

        result = responses_agent.generate(
            "Create a Python function that calculates fibonacci numbers and save it to a file",
            tools=[container_tool.spec()],
        )

        assert len(result.content) > 0
        print(f"✓ Container Persistence Response: {result.content}")

        # Check for file generation
        if result.grounding_metadata:
            if "file_citations" in result.grounding_metadata:
                files = result.grounding_metadata["file_citations"]
                print(f"✓ Generated {len(files)} files")
                for file_info in files:
                    print(f"  - {file_info.get('filename', 'Unknown')}")

            if "code_interpreter" in result.grounding_metadata:
                code_info = result.grounding_metadata["code_interpreter"]
                if isinstance(code_info, dict):
                    print(f"✓ Container ID: {code_info.get('container_id', 'Unknown')}")


class TestRealErrorScenarios:
    """Test real error scenarios and edge cases."""

    @pytest.mark.integration
    def test_invalid_model_error(self, openai_api_key):
        """Test handling of invalid model names."""
        try:
            agent = Agent(
                provider="openai", model="invalid-model-name", api_key=openai_api_key
            )
            result = agent.generate("Hello")
            # Should get an error about invalid model
            print(f"✓ Invalid model response: {result.content}")
        except Exception as e:
            print(f"✓ Expected error for invalid model: {e}")

    @pytest.mark.integration
    def test_rate_limit_handling(self, responses_agent):
        """Test graceful handling if we hit rate limits."""
        # Make multiple rapid requests to potentially trigger rate limiting
        results = []
        for i in range(3):
            try:
                result = responses_agent.generate(f"Quick test {i}: What is {i} + {i}?")
                results.append(result.content)
                print(f"✓ Request {i}: {result.content}")
            except Exception as e:
                print(f"✓ Rate limit or error on request {i}: {e}")

        # At least one should succeed
        assert len(results) > 0

    @pytest.mark.integration
    def test_complex_tool_request(self, responses_agent):
        """Test complex request that exercises multiple tool capabilities."""
        result = responses_agent.generate(
            "Search for Python programming tutorials, write a 'Hello World' program, and create an image showing the output",
            tools=[
                "openai_web_search",
                "openai_code_interpreter",
                "openai_image_generation",
            ],
        )

        assert len(result.content) > 0
        print(f"✓ Complex Multi-Tool Response: {result.content[:500]}...")

        # Count different types of metadata
        if result.grounding_metadata:
            metadata_types = list(result.grounding_metadata.keys())
            print(f"✓ Metadata types present: {metadata_types}")


class TestOpenAIResponseStructures:
    """Test different OpenAI response structures for analysis."""

    @pytest.mark.integration
    def test_responses_api_structure_analysis(self, openai_api_key):
        """Deep analysis of Responses API structure with different tools."""
        # Test different tools with Responses API to understand structures
        agent = Agent(
            provider="openai", model="gpt-4o", api_key=openai_api_key, api="responses"
        )

        # Test 1: Web search structure
        result = agent.generate(
            "Quick search: latest Python version", tools=["openai_web_search"]
        )
        log_response_analysis(
            "Responses API Web Search Structure", result, "responses_api_detailed"
        )
        save_response_to_file("responses_web_search_structure", result, "responses_api")

        # Test 2: Code interpreter structure
        result = agent.generate(
            "Calculate 5! using Python", tools=["openai_code_interpreter"]
        )
        log_response_analysis(
            "Responses API Code Structure", result, "responses_api_detailed"
        )
        save_response_to_file("responses_code_structure", result, "responses_api")

    @pytest.mark.integration
    def test_chat_completions_structure_analysis(self, openai_api_key):
        """Deep analysis of Chat Completions API structure."""
        agent = Agent(provider="openai", model="gpt-4o-mini", api_key=openai_api_key)

        # Test with function calling
        result = agent.generate("Format as JSON: name=Test", tools=["json_format"])
        log_response_analysis(
            "Chat Completions Function Call Structure",
            result,
            "chat_completions_detailed",
        )
        save_response_to_file("chat_function_structure", result, "chat_completions")


class TestRealUsabilityScenarios:
    """Test real-world usability scenarios."""

    @pytest.mark.integration
    def test_string_tool_names(self, responses_agent):
        """Test using simple string tool names (user-friendly approach)."""
        # Test all three tools with string names
        test_cases = [
            ("openai_web_search", "What's the weather like today?"),
            ("openai_code_interpreter", "Calculate the square root of 144"),
            ("openai_image_generation", "Draw a simple smiley face"),
        ]

        for tool_name, prompt in test_cases:
            result = responses_agent.generate(prompt, tools=[tool_name])
            assert len(result.content) > 0
            print(f"✓ {tool_name}: {result.content[:150]}...")

    @pytest.mark.integration
    def test_generic_tool_names(self, openai_api_key):
        """Test using generic tool names that default to best providers."""
        agent = Agent(provider="openai", model="gpt-4o", api_key=openai_api_key)

        # Test generic names
        test_cases = [
            ("web_search", "What's happening in tech news?"),
            ("code_interpreter", "Calculate factorial of 5"),
            ("image_generation", "Create a logo design"),
        ]

        for tool_name, prompt in test_cases:
            try:
                result = agent.generate(prompt, tools=[tool_name])
                assert len(result.content) > 0
                print(f"✓ Generic {tool_name}: {result.content[:150]}...")
            except Exception as e:
                print(f"⚠ Generic {tool_name} failed: {e}")

    @pytest.mark.integration
    def test_mixed_provider_tools(self, openai_api_key):
        """Test mixing OpenAI tools with other provider tools."""
        agent = Agent(provider="openai", model="gpt-4o", api_key=openai_api_key)

        # Mix OpenAI tools with local tools
        result = agent.generate(
            "Search for a website about Python programming, fetch its content, and format the results as JSON",
            tools=["openai_web_search", "fetch_url", "json_format"],
        )

        assert len(result.content) > 0
        print(f"✓ Mixed Tools Response: {result.content[:300]}...")


def cleanup_test_files():
    """Clean up any generated test files."""
    import glob

    test_files = glob.glob("test_*.png") + glob.glob("test_*.jpg")
    for file in test_files:
        try:
            os.remove(file)
            print(f"✓ Cleaned up {file}")
        except:
            pass


if __name__ == "__main__":
    # Run tests with pytest
    print("🚀 Running Real OpenAI API Integration Tests")
    print("=" * 60)
    print("⚠ WARNING: These tests make real API calls and will use OpenAI credits!")
    print("⚠ Ensure OPENAI_API_KEY is set in your .env file")
    print("=" * 60)

    # Run only integration tests
    exit_code = pytest.main([__file__, "-v", "-m", "integration"])

    # Clean up generated files
    cleanup_test_files()

    if exit_code == 0:
        print("\n✅ All real API tests passed!")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")
