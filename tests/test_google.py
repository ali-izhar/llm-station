"""
Real Google Gemini API integration tests using GEMINI_API_KEY from .env

These tests make actual API calls to Google Gemini and require:
- GEMINI_API_KEY or GOOGLE_API_KEY in .env file
- Active internet connection
- Google API access

Tests cover:
- Gemini 2.5+ models with native tools
- Search grounding with automatic citations
- Code execution with Python and visualization
- URL context processing for websites, PDFs, images
- Image generation with Gemini 2.5 models
- Batch processing capabilities

Enhanced with detailed response analysis and logging for model validation.
"""

import os
import json
import pytest
from dotenv import load_dotenv
from llm_studio import Agent, setup_logging, LogLevel, GoogleBatchProcessor
from llm_studio.models.google import GoogleProvider
from llm_studio.models.base import ModelConfig
from llm_studio.schemas.messages import UserMessage, SystemMessage
from llm_studio.tools.web_search.google import GoogleWebSearch, GoogleSearchRetrieval
from llm_studio.tools.code_execution.google import GoogleCodeExecution
from llm_studio.tools.url_context.google import GoogleUrlContext
from llm_studio.tools.image_generation.google import GoogleImageGeneration

# Load environment variables
load_dotenv()

# Enable detailed logging for analysis
setup_logging(level=LogLevel.DEBUG)


def log_google_response_analysis(
    test_name: str, response, api_type: str = "gemini_api"
):
    """Detailed logging and analysis of Google Gemini API responses."""
    print(f"\n{'='*80}")
    print(f"GOOGLE RESPONSE ANALYSIS: {test_name}")
    print(f"API Type: {api_type}")
    print(f"{'='*80}")

    # Basic response info
    print(f"Content Length: {len(response.content)}")
    print(f"Content Preview: {response.content[:300]}...")
    print(f"Tool Calls Count: {len(response.tool_calls)}")

    # Tool calls analysis
    if response.tool_calls:
        print(f"\nTool Calls:")
        for i, tc in enumerate(response.tool_calls):
            print(f"  {i+1}. ID: {tc.id}")
            print(f"     Name: {tc.name}")
            print(f"     Arguments: {tc.arguments}")

    # Grounding metadata analysis (Google-specific)
    if response.grounding_metadata:
        print(f"\nGrounding Metadata Keys: {list(response.grounding_metadata.keys())}")

        for key, value in response.grounding_metadata.items():
            print(f"\n{key.upper()}:")
            if isinstance(value, list):
                print(f"  Type: List with {len(value)} items")
                if value and len(value) <= 2:  # Show first few items
                    for i, item in enumerate(value):
                        print(f"  [{i}]: {str(item)[:150]}...")
            elif isinstance(value, dict):
                print(f"  Type: Dict with {len(value)} keys: {list(value.keys())}")
                for k, v in value.items():
                    if isinstance(v, str) and len(v) > 50:
                        print(f"    {k}: {str(v)[:50]}...")
                    elif isinstance(v, list):
                        print(f"    {k}: List with {len(v)} items")
                    elif isinstance(v, dict):
                        print(f"    {k}: Dict with {len(v)} keys")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  Type: {type(value).__name__}")
                print(f"  Value: {str(value)[:100]}...")

    # Raw response structure analysis (Google-specific)
    if response.raw:
        print(f"\nRaw Response Structure:")
        print(f"  Type: {type(response.raw)}")
        if isinstance(response.raw, dict):
            print(f"  Top-level keys: {list(response.raw.keys())}")

            # Analyze Google-specific response patterns
            if "candidates" in response.raw:
                candidates = response.raw["candidates"]
                print(f"  Candidates length: {len(candidates)}")
                if candidates:
                    candidate = candidates[0]
                    print(f"  First candidate keys: {list(candidate.keys())}")

                    if "content" in candidate:
                        content = candidate["content"]
                        print(f"  Content keys: {list(content.keys())}")
                        if "parts" in content:
                            parts = content["parts"]
                            print(f"  Parts length: {len(parts)}")
                            for i, part in enumerate(parts[:3]):
                                if isinstance(part, dict):
                                    print(f"    Part[{i}] keys: {list(part.keys())}")
                                    print(
                                        f"    Part[{i}] types: {[k for k in part.keys() if part.get(k)]}"
                                    )

                    # Check for grounding metadata in raw response
                    if "groundingMetadata" in candidate:
                        gm = candidate["groundingMetadata"]
                        print(f"  Raw grounding metadata keys: {list(gm.keys())}")

                    # Check for URL context metadata
                    if "url_context_metadata" in candidate:
                        ucm = candidate["url_context_metadata"]
                        print(
                            f"  Raw URL context metadata: {type(ucm)} with {len(ucm) if isinstance(ucm, (list, dict)) else 'unknown'} items"
                        )

    print(f"{'='*80}\n")


def save_google_response_to_file(test_name: str, response, api_type: str):
    """Save full Google response to file for detailed analysis."""
    os.makedirs("test_outputs", exist_ok=True)

    analysis_data = {
        "test_name": test_name,
        "api_type": api_type,
        "provider": "google",
        "content": response.content,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in response.tool_calls
        ],
        "grounding_metadata": response.grounding_metadata,
        "raw_response": response.raw,
    }

    filename = (
        f"test_outputs/google_{test_name.replace(' ', '_').lower()}_{api_type}.json"
    )
    with open(filename, "w") as f:
        json.dump(analysis_data, f, indent=2, default=str)

    print(f"💾 Saved Google analysis to: {filename}")


@pytest.fixture
def google_api_key():
    """Get Google API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file")
    return api_key


@pytest.fixture
def google_agent(google_api_key):
    """Create Google Gemini agent for testing."""
    return Agent(
        provider="google",
        model="gemini-2.5-flash",
        api_key=google_api_key,
        system_prompt="You are a helpful test assistant.",
    )


@pytest.fixture
def google_pro_agent(google_api_key):
    """Create Google Gemini Pro agent for testing."""
    return Agent(
        provider="google",
        model="gemini-2.5-pro",
        api_key=google_api_key,
        system_prompt="You are a helpful assistant with advanced reasoning capabilities.",
    )


@pytest.fixture
def google_image_agent(google_api_key):
    """Create Google Gemini agent for image generation."""
    return Agent(
        provider="google",
        model="gemini-2.5-flash-image-preview",
        api_key=google_api_key,
        system_prompt="You are a helpful assistant with image generation capabilities.",
    )


class TestRealGeminiBasic:
    """Test real Google Gemini basic API calls."""

    @pytest.mark.integration
    def test_basic_gemini_chat(self, google_agent):
        """Test basic Gemini chat functionality."""
        result = google_agent.generate("Hello! What is 2+2?")

        assert len(result.content) > 0
        assert "4" in result.content
        print(f"✓ Gemini Basic Response: {result.content}")

        # Detailed response analysis
        log_google_response_analysis("Basic Gemini Chat", result, "gemini_basic")
        save_google_response_to_file("basic_gemini_chat", result, "gemini_basic")

    @pytest.mark.integration
    def test_gemini_with_function_tools(self, google_agent):
        """Test Gemini with local function tools."""
        result = google_agent.generate(
            "Format this data as JSON: name=Bob, age=25, city=Tokyo",
            tools=["json_format"],
        )

        assert len(result.content) > 0
        print(f"✓ Function Tool Response: {result.content}")

        # Should have executed the tool locally
        if result.tool_calls:
            print(f"✓ Tool calls made: {[tc.name for tc in result.tool_calls]}")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini with Function Tools", result, "gemini_with_functions"
        )
        save_google_response_to_file("gemini_function_tools", result, "gemini_basic")

    @pytest.mark.integration
    def test_gemini_pro_reasoning(self, google_pro_agent):
        """Test Gemini Pro with complex reasoning."""
        result = google_pro_agent.generate(
            "Explain the relationship between quantum mechanics and general relativity in simple terms"
        )

        assert len(result.content) > 0
        print(f"✓ Gemini Pro Reasoning: {result.content[:300]}...")

        # Detailed response analysis
        log_google_response_analysis("Gemini Pro Reasoning", result, "gemini_pro")
        save_google_response_to_file("gemini_pro_reasoning", result, "gemini_pro")


class TestRealGeminiSearch:
    """Test real Google Gemini search grounding."""

    @pytest.mark.integration
    def test_basic_search_grounding(self, google_agent):
        """Test Gemini 2.0+ search with automatic grounding."""
        result = google_agent.generate(
            "What are the latest developments in renewable energy technology?",
            tools=["google_search"],
        )

        assert len(result.content) > 0
        print(f"✓ Search Grounding Response: {result.content[:400]}...")

        # Check for grounding metadata
        if result.grounding_metadata:
            print(f"✓ Search Metadata: {list(result.grounding_metadata.keys())}")

            if "sources" in result.grounding_metadata:
                sources = result.grounding_metadata["sources"]
                print(f"✓ Found {len(sources)} sources")

            if "citations" in result.grounding_metadata:
                citations = result.grounding_metadata["citations"]
                print(f"✓ Found {len(citations)} citations")

            if "search_entry_point" in result.grounding_metadata:
                print("✓ Search entry point available (Google Search Suggestions)")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Search Grounding", result, "gemini_search_2_0"
        )
        save_google_response_to_file("gemini_search_grounding", result, "gemini_search")

    @pytest.mark.integration
    def test_legacy_search_retrieval(self, google_agent):
        """Test legacy search retrieval (Gemini 1.5 style)."""
        search_tool = GoogleSearchRetrieval(mode="MODE_DYNAMIC", dynamic_threshold=0.7)

        result = google_agent.generate(
            "What is the current status of electric vehicle adoption globally?",
            tools=[search_tool.spec()],
        )

        assert len(result.content) > 0
        print(f"✓ Legacy Search Response: {result.content[:400]}...")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Legacy Search", result, "gemini_search_legacy"
        )
        save_google_response_to_file("gemini_legacy_search", result, "gemini_search")


class TestRealGeminiCodeExecution:
    """Test real Google Gemini code execution."""

    @pytest.mark.integration
    def test_basic_code_execution(self, google_agent):
        """Test Gemini code execution with mathematical calculation."""
        result = google_agent.generate(
            "Calculate the first 10 prime numbers using Python. Show the code and results.",
            tools=["google_code_execution"],
        )

        assert len(result.content) > 0
        print(f"✓ Code Execution Response: {result.content[:500]}...")

        # Check for code execution metadata
        if result.grounding_metadata:
            print(f"✓ Code Metadata: {list(result.grounding_metadata.keys())}")

            if "code_execution" in result.grounding_metadata:
                code_executions = result.grounding_metadata["code_execution"]
                print(f"✓ Executed {len(code_executions)} code blocks")
                for i, exec_info in enumerate(code_executions):
                    print(f"  Block {i+1}: {exec_info.get('language', 'unknown')} code")
                    if "result" in exec_info:
                        outcome = exec_info["result"].get("outcome", "unknown")
                        print(f"    Outcome: {outcome}")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Code Execution", result, "gemini_code_execution"
        )
        save_google_response_to_file("gemini_code_execution", result, "gemini_code")

    @pytest.mark.integration
    def test_data_analysis_with_visualization(self, google_agent):
        """Test Gemini code execution with data analysis and visualization."""
        result = google_agent.generate(
            """Create a dataset of 20 random numbers, calculate statistics (mean, std, median), 
            and create a histogram visualization. Show all code and results.""",
            tools=["google_code_execution"],
        )

        assert len(result.content) > 0
        print(f"✓ Data Analysis Response: {result.content[:500]}...")

        # Check for visualization generation
        if result.grounding_metadata:
            if "inline_media" in result.grounding_metadata:
                media = result.grounding_metadata["inline_media"]
                print(f"✓ Generated {len(media)} media files")
                for media_item in media:
                    print(f"  - {media_item['mime_type']} ({media_item['size']} bytes)")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Data Analysis", result, "gemini_code_with_viz"
        )
        save_google_response_to_file("gemini_data_analysis", result, "gemini_code")

    @pytest.mark.integration
    def test_iterative_code_refinement(self, google_agent):
        """Test Gemini's iterative code learning capability."""
        result = google_agent.generate(
            """Implement the bubble sort algorithm in Python, then optimize it and compare 
            performance with different input sizes. Show execution times.""",
            tools=["google_code_execution"],
        )

        assert len(result.content) > 0
        print(f"✓ Iterative Code Response: {result.content[:500]}...")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Iterative Code", result, "gemini_iterative_code"
        )
        save_google_response_to_file("gemini_iterative_code", result, "gemini_code")


class TestRealGeminiUrlContext:
    """Test real Google Gemini URL context processing."""

    @pytest.mark.integration
    def test_website_analysis(self, google_agent):
        """Test URL context with website analysis."""
        result = google_agent.generate(
            "Based on https://ai.google.dev/gemini-api/docs, summarize the key capabilities of Gemini models",
            tools=["google_url_context"],
        )

        assert len(result.content) > 0
        print(f"✓ Website Analysis Response: {result.content[:400]}...")

        # Check for URL context metadata
        if result.grounding_metadata:
            if "url_context" in result.grounding_metadata:
                print("✓ URL context metadata available")
            if "processed_urls" in result.grounding_metadata:
                processed = result.grounding_metadata["processed_urls"]
                print(f"✓ Processed {len(processed)} URLs")
                for url_info in processed:
                    print(
                        f"  - {url_info.get('url', '')[:50]}... - {url_info.get('status', 'unknown')}"
                    )

        # Detailed response analysis
        log_google_response_analysis("Gemini URL Context", result, "gemini_url_context")
        save_google_response_to_file("gemini_url_context", result, "gemini_url")

    @pytest.mark.integration
    def test_pdf_analysis(self, google_agent):
        """Test URL context with PDF analysis."""
        # Use a publicly accessible PDF for testing
        result = google_agent.generate(
            """Analyze this research paper and provide key insights: 
            https://arxiv.org/pdf/1706.03762.pdf
            Focus on the main contributions and methodology.""",
            tools=["google_url_context"],
        )

        assert len(result.content) > 0
        print(f"✓ PDF Analysis Response: {result.content[:400]}...")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini PDF Analysis", result, "gemini_pdf_context"
        )
        save_google_response_to_file("gemini_pdf_analysis", result, "gemini_url")

    @pytest.mark.integration
    def test_image_url_analysis(self, google_agent):
        """Test URL context with image analysis."""
        result = google_agent.generate(
            """Describe the components and structure in this diagram: 
            https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Trombone.svg/960px-Trombone.svg.png
            Identify the labeled parts.""",
            tools=["google_url_context"],
        )

        assert len(result.content) > 0
        print(f"✓ Image URL Analysis Response: {result.content[:400]}...")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Image URL", result, "gemini_image_url_context"
        )
        save_google_response_to_file("gemini_image_url", result, "gemini_url")


class TestRealGeminiImageGeneration:
    """Test real Google Gemini image generation."""

    @pytest.mark.integration
    def test_basic_image_generation(self, google_image_agent):
        """Test Gemini 2.5 native image generation."""
        result = google_image_agent.generate(
            "Create a photorealistic image of a blue robot reading a book in a modern library",
            tools=["google_image_generation"],
        )

        assert len(result.content) > 0
        print(f"✓ Image Generation Response: {result.content}")

        # Check for image generation metadata
        if result.grounding_metadata:
            if "image_generation" in result.grounding_metadata:
                images = result.grounding_metadata["image_generation"]
                print(f"✓ Generated {len(images)} images")
                for i, img_info in enumerate(images):
                    print(f"  Image {i+1}: {img_info.get('type', 'unknown')} format")

        # Check raw response for image parts
        if result.raw and "candidates" in result.raw:
            candidate = result.raw["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                image_parts = [p for p in parts if "image" in p or "inlineData" in p]
                print(f"✓ Found {len(image_parts)} image parts in raw response")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Image Generation", result, "gemini_image_generation"
        )
        save_google_response_to_file("gemini_image_generation", result, "gemini_image")

    @pytest.mark.integration
    def test_multi_image_story(self, google_image_agent):
        """Test Gemini multi-image story generation."""
        result = google_image_agent.generate(
            "Create a 3-part visual story about a space explorer discovering a new planet. Generate 3 images showing: 1) approach to planet, 2) landing, 3) discovery of alien life"
        )

        assert len(result.content) > 0
        print(f"✓ Multi-Image Story Response: {result.content[:500]}...")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Multi-Image Story", result, "gemini_multi_image"
        )
        save_google_response_to_file("gemini_multi_image_story", result, "gemini_image")


class TestRealGeminiCombinedWorkflows:
    """Test real multi-tool workflows with Gemini."""

    @pytest.mark.integration
    def test_research_and_analyze_workflow(self, google_agent):
        """Test workflow combining search and code execution."""
        result = google_agent.generate(
            """Research current Bitcoin price trends and create a Python analysis:
            1. Search for recent Bitcoin price data
            2. Use Python to calculate basic statistics
            3. Create a simple trend visualization
            
            Provide comprehensive analysis with data.""",
            tools=["google_search", "google_code_execution"],
        )

        assert len(result.content) > 0
        print(f"✓ Research + Analysis Response: {result.content[:500]}...")

        # Should have both search and code execution metadata
        if result.grounding_metadata:
            metadata_keys = list(result.grounding_metadata.keys())
            print(f"✓ Metadata types: {metadata_keys}")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Research + Code", result, "gemini_search_code_combo"
        )
        save_google_response_to_file("gemini_research_analyze", result, "gemini_combo")

    @pytest.mark.integration
    def test_url_search_combo(self, google_agent):
        """Test workflow combining URL context and search."""
        result = google_agent.generate(
            """Analyze the Gemini documentation at https://ai.google.dev/gemini-api/docs/models
            and then search for recent news about Gemini model improvements. 
            Provide a comprehensive summary.""",
            tools=["google_url_context", "google_search"],
        )

        assert len(result.content) > 0
        print(f"✓ URL + Search Response: {result.content[:500]}...")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini URL + Search", result, "gemini_url_search_combo"
        )
        save_google_response_to_file("gemini_url_search_combo", result, "gemini_combo")

    @pytest.mark.integration
    def test_full_multimodal_pipeline(self, google_image_agent):
        """Test complete multimodal pipeline with all tools."""
        result = google_image_agent.generate(
            """Create a comprehensive research project:
            1. Search for recent AI safety research
            2. Analyze key papers and findings using Python
            3. Create visualizations of the research trends
            4. Generate an image representing AI safety concepts
            
            Provide detailed analysis with citations.""",
            tools=["google_search", "google_code_execution", "google_image_generation"],
        )

        assert len(result.content) > 0
        print(f"✓ Full Pipeline Response: {result.content[:600]}...")

        # Detailed response analysis
        log_google_response_analysis(
            "Gemini Full Pipeline", result, "gemini_full_pipeline"
        )
        save_google_response_to_file("gemini_full_pipeline", result, "gemini_combo")


class TestRealGeminiBatch:
    """Test real Google Gemini batch processing."""

    @pytest.mark.integration
    def test_batch_text_processing(self, google_api_key):
        """Test batch processing with text analysis."""
        processor = GoogleBatchProcessor(api_key=google_api_key)

        # Create small batch for testing
        test_texts = [
            "Artificial intelligence is transforming healthcare",
            "Renewable energy adoption is accelerating globally",
            "Quantum computing promises breakthrough capabilities",
        ]

        tasks = []
        for i, text in enumerate(test_texts):
            task = processor.create_task(
                key=f"analysis-{i}",
                model="gemini-2.5-flash",
                contents=text,
                system_instruction="Provide a one-sentence summary and identify key themes",
                generation_config={"temperature": 0.1},
            )
            tasks.append(task)

        # Create batch file (don't submit to avoid costs)
        batch_file = processor.create_batch_file(tasks, "test_google_batch.jsonl")
        print(f"✓ Created batch file: {batch_file}")
        print(f"✓ {len(tasks)} tasks prepared for batch processing")

        # Analyze the batch file structure
        with open(batch_file, "r") as f:
            batch_content = f.read()
            print(f"✓ Batch file content preview:")
            lines = batch_content.strip().split("\n")
            for i, line in enumerate(lines[:2]):  # Show first 2 lines
                data = json.loads(line)
                print(
                    f"  Line {i+1}: key={data['key']}, request keys={list(data['request'].keys())}"
                )

    @pytest.mark.integration
    def test_embeddings_batch_structure(self, google_api_key):
        """Test embeddings batch structure."""
        processor = GoogleBatchProcessor(api_key=google_api_key)

        # Test embeddings batch creation
        test_texts = ["AI research", "Machine learning", "Deep learning"]

        try:
            batch_job = processor.create_embeddings_batch(
                texts=test_texts,
                model="gemini-embedding-001",
                output_dimensionality=512,
            )
            print(f"✓ Embeddings batch created: {batch_job.name}")
            print(f"✓ Batch status: {batch_job.state.value}")

        except Exception as e:
            print(f"⚠ Embeddings batch creation: {e}")
            # Still useful for understanding the structure


class TestGeminiResponseStructures:
    """Test different response structures from Gemini API."""

    @pytest.mark.integration
    def test_direct_provider_call(self, google_api_key):
        """Test direct provider call to understand raw response structure."""
        provider = GoogleProvider(api_key=google_api_key)
        messages = [UserMessage("What is machine learning?")]
        config = ModelConfig(provider="google", model="gemini-2.5-flash")

        result = provider.generate(messages, config)

        assert len(result.content) > 0
        print(f"✓ Direct Provider Response: {result.content[:300]}...")

        # Detailed response analysis
        log_google_response_analysis("Direct Provider Call", result, "gemini_direct")
        save_google_response_to_file("gemini_direct_provider", result, "gemini_direct")

    @pytest.mark.integration
    def test_provider_with_tools(self, google_api_key):
        """Test provider with tools to understand tool response structure."""
        provider = GoogleProvider(api_key=google_api_key)
        messages = [UserMessage("Search for recent Python tutorials and analyze them")]
        config = ModelConfig(provider="google", model="gemini-2.5-flash")

        # Test with search tool
        search_tool = GoogleWebSearch()
        result = provider.generate(messages, config, tools=[search_tool.spec()])

        assert len(result.content) > 0
        print(f"✓ Provider with Tools Response: {result.content[:300]}...")

        # Detailed response analysis
        log_google_response_analysis(
            "Provider with Tools", result, "gemini_provider_tools"
        )
        save_google_response_to_file("gemini_provider_tools", result, "gemini_provider")


class TestGeminiErrorScenarios:
    """Test error scenarios and edge cases with Gemini."""

    @pytest.mark.integration
    def test_invalid_model_error(self, google_api_key):
        """Test handling of invalid model names."""
        try:
            agent = Agent(
                provider="google", model="invalid-gemini-model", api_key=google_api_key
            )
            result = agent.generate("Hello")
            print(f"✓ Invalid model response: {result.content}")
        except Exception as e:
            print(f"✓ Expected error for invalid model: {e}")

    @pytest.mark.integration
    def test_tool_combination_edge_cases(self, google_agent):
        """Test edge cases with tool combinations."""
        result = google_agent.generate(
            "Research quantum computing, analyze with Python, and process this URL: https://ai.google.dev",
            tools=["google_search", "google_code_execution", "google_url_context"],
        )

        assert len(result.content) > 0
        print(f"✓ Tool Combo Edge Case: {result.content[:400]}...")

        # Detailed response analysis
        log_google_response_analysis(
            "Tool Combination Edge Case", result, "gemini_edge_case"
        )
        save_google_response_to_file("gemini_tool_combo_edge", result, "gemini_edge")


def cleanup_test_files():
    """Clean up any generated test files."""
    import glob

    test_files = (
        glob.glob("test_*.png")
        + glob.glob("test_*.jpg")
        + glob.glob("*batch*.jsonl")
        + glob.glob("google_batch_*.jsonl")
    )

    for file in test_files:
        try:
            os.remove(file)
            print(f"✓ Cleaned up {file}")
        except:
            pass


if __name__ == "__main__":
    # Run tests with pytest
    print("🚀 Running Real Google Gemini API Integration Tests")
    print("=" * 60)
    print("⚠ WARNING: These tests make real API calls and will use Google API credits!")
    print("⚠ Ensure GEMINI_API_KEY or GOOGLE_API_KEY is set in your .env file")
    print("=" * 60)

    # Run only integration tests
    exit_code = pytest.main([__file__, "-v", "-m", "integration"])

    # Clean up generated files
    cleanup_test_files()

    if exit_code == 0:
        print("\n✅ All real Google API tests passed!")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")
