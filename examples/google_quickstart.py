#!/usr/bin/env python3
"""Google Gemini Quickstart - Get started with Google Gemini 2.0+ in LLM Station"""

import os
from dotenv import load_dotenv

from llm_station import (
    Agent,
    setup_logging,
    LogLevel,
    GoogleBatchProcessor,
)
from llm_station import get_available_tools
from llm_station.cli.logging_cli import generate_log_filename


def main():
    """Quick Google Gemini setup and testing."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file")
        print("   Add: GEMINI_API_KEY=your-key-here")
        return

    # Enable logging with automatic file creation
    os.makedirs("logs", exist_ok=True)
    print("Created logs/ directory")

    # Setup logging with file output
    logger = setup_logging(level=LogLevel.INFO)
    log_file_path = generate_log_filename("google", "gemini-2.5-flash")

    # Create log file for session
    log_file = open(log_file_path, "w", encoding="utf-8")
    logger.log_file = log_file

    print(f"Logging enabled: {log_file_path}")

    # Create Gemini agent with latest model
    agent = Agent(
        provider="google",
        model="gemini-2.5-flash",  # Latest Gemini 2.0+ model
        api_key=api_key,
        system_prompt="You are a helpful research assistant with access to web search.",
    )

    # Also create an image-capable agent for testing
    image_agent = Agent(
        provider="google",
        model="gemini-2.5-flash-image-preview",  # Image generation model
        api_key=api_key,
        system_prompt="You are a helpful assistant with image generation capabilities.",
    )

    print(f"✅ Google Gemini agent created: {agent.provider_name}")

    # Show available smart tools
    tools = get_available_tools()
    smart_tools = [name for name, type_info in tools.items() if type_info == "smart"]
    primary_tools = ["search", "code", "image", "json", "fetch", "url"]
    available_primary = [t for t in primary_tools if t in smart_tools]
    print(f"🔧 Available smart tools: {len(available_primary)}")
    for tool in available_primary:
        print(f"   - {tool}")

    print(f"\nRunning comprehensive Gemini 2.0 tests...")

    # Test 1: Basic chat
    print(f"\nTest 1: Basic Chat")
    response = agent.generate("What is 2 + 2?")
    print(f"Response: {response.content}")

    # Test 2: Web search with grounding (using generic "search" tool)
    print(f"\nTest 2: Web Search with Grounding")
    response = agent.generate(
        "What are the latest developments in AI this week?", tools=["search"]
    )
    print(f"Response: {response.content[:300]}...")

    if response.grounding_metadata:
        print(f"✓ Grounding metadata: {list(response.grounding_metadata.keys())}")

        # Show sources if available
        if "sources" in response.grounding_metadata:
            sources = response.grounding_metadata["sources"]
            print(f"✓ Found {len(sources)} sources:")
            for i, source in enumerate(sources[:3]):  # Show first 3
                print(f"   {i+1}. {source}")

        # Show citations if available
        if "citations" in response.grounding_metadata:
            citations = response.grounding_metadata["citations"]
            print(f"✓ Found {len(citations)} citations")

        # Show search entry point if available
        if "search_entry_point" in response.grounding_metadata:
            print("✓ Search entry point available (Google Search Suggestions)")

    # Test 3: Code execution with data analysis
    print(f"\nTest 3: Code Execution & Data Analysis")
    response = agent.generate(
        """Calculate the sum of the first 50 prime numbers. 
        Generate and run Python code to:
        1. Find the first 50 prime numbers
        2. Calculate their sum
        3. Create a simple visualization showing the distribution
        
        Make sure you get all 50 prime numbers and show the code used.""",
        tools=["code"],
    )
    print(f"Response: {response.content[:600]}...")

    if response.grounding_metadata:
        print(f"✓ Code execution metadata: {list(response.grounding_metadata.keys())}")

        # Show code execution details
        if "code_execution" in response.grounding_metadata:
            code_executions = response.grounding_metadata["code_execution"]
            print(f"✓ Executed {len(code_executions)} code blocks")
            for i, exec_info in enumerate(code_executions):
                if "result" in exec_info:
                    outcome = exec_info["result"].get("outcome", "OK")
                    print(f"  Block {i+1}: {exec_info['language']} code - {outcome}")

        # Show generated media
        if "inline_media" in response.grounding_metadata:
            media = response.grounding_metadata["inline_media"]
            print(f"✓ Generated {len(media)} media files")
            for media_item in media:
                print(f"  - {media_item['mime_type']} ({media_item['size']} bytes)")

    # Test 4: Combined research workflow (like the documentation example)
    print(f"\nTest 4: Research Workflow (Company Analysis)")
    company = "Tesla"
    response = agent.generate(
        f"""You are an analyst conducting company research on {company}.
        
        Please research and provide:
        1. Recent news and developments
        2. Stock performance overview
        3. Key business segments
        4. Recent financial highlights
        
        Use web search to find current information and provide a concise report.""",
        tools=["search"],
    )

    print(f"Research Report: {response.content[:500]}...")

    if response.grounding_metadata:
        sources = response.grounding_metadata.get("sources", [])
        citations = response.grounding_metadata.get("citations", [])
        print(f"✓ Research used {len(sources)} sources and {len(citations)} citations")

    # Test 5: URL Context (direct content processing)
    print(f"\nTest 5: URL Context Tool")
    response = agent.generate(
        """Based on https://ai.google.dev/gemini-api/docs/models, what are the key 
        differences between Gemini 1.5, Gemini 2.0 and Gemini 2.5 models? 
        Create a brief comparison focusing on the main capabilities.""",
        tools=["url"],
    )
    print(f"URL Context Response: {response.content[:400]}...")

    if response.grounding_metadata:
        if "url_context" in response.grounding_metadata:
            print("✓ URL context metadata available")
        if "processed_urls" in response.grounding_metadata:
            processed = response.grounding_metadata["processed_urls"]
            print(f"✓ Processed {len(processed)} URLs")
            for url_info in processed:
                status = url_info.get("status", "unknown")
                content_type = url_info.get("content_type", "unknown")
                print(
                    f"  - {url_info.get('url', '')[:50]}... - {status} ({content_type})"
                )

    # Test 6: Combined tools (URL context + search)
    print(f"\nTest 6: Combined Tools (URL + Search)")
    response = agent.generate(
        """Analyze this Gemini documentation: https://ai.google.dev/gemini-api/docs/models
        and then search for recent news about Gemini model updates or improvements.
        Provide a summary combining both sources.""",
        tools=["url", "search"],
    )
    print(f"Combined Tools Response: {response.content[:400]}...")

    if response.grounding_metadata:
        tools_used = list(response.grounding_metadata.keys())
        print(f"✓ Tools used: {tools_used}")

    # Test 7: Image Generation (using dedicated image model)
    print(f"\nTest 7: Image Generation")
    try:
        response = image_agent.generate(
            """Create a simple, clean image of a robot reading a book in a library. 
            The robot should have a friendly appearance with blue accents.""",
            tools=["image"],
        )
        print(f"Image Generation Response: {response.content[:300]}...")

        if response.grounding_metadata:
            if "image_generation" in response.grounding_metadata:
                images = response.grounding_metadata["image_generation"]
                print(f"✓ Generated {len(images)} images")
                for i, img_info in enumerate(images):
                    print(f"  Image {i+1}: {img_info.get('type', 'unknown')} format")
            else:
                print(
                    "✓ Image generation requested (check response.raw for image data)"
                )

        # Note about accessing actual images
        print(
            "✓ To access actual images, use response.raw and look for parts with as_image() method"
        )

    except Exception as e:
        print(f"⚠ Image generation test failed: {e}")
        print(f"  Using model: {image_agent._base_config.model}")
        print("  Note: Ensure you have access to Gemini 2.5 image generation models")

    # Test 8: Generic tool names (should default to Google)
    print(f"\nTest 8: Generic Tool Names")
    response = agent.generate(
        "Search for recent Python programming tutorials",
        tools=["search"],  # Will auto-route to Google search for Google agent
    )
    print(f"Generic search response: {response.content[:200]}...")

    # Test 9: Batch Processing Demo
    print(f"\nTest 9: Batch Processing Demo")
    try:
        processor = GoogleBatchProcessor(api_key=api_key)

        # Create sample batch tasks for analysis
        sample_topics = [
            "Renewable energy trends in 2025",
            "Electric vehicle market growth",
            "AI impact on healthcare",
        ]

        tasks = []
        for i, topic in enumerate(sample_topics):
            task = processor.create_task(
                key=f"analysis-{i}",
                model="gemini-2.5-flash",
                contents=f"Provide a brief analysis of: {topic}",
                system_instruction="You are a research analyst. Provide concise insights.",
                generation_config={"temperature": 0.2},
            )
            tasks.append(task)

        # Create batch file (don't submit to avoid costs)
        batch_file = processor.create_batch_file(tasks, "google_batch_demo.jsonl")
        print(f"✅ Created Google batch file: {batch_file}")
        print(f"    {len(tasks)} tasks ready for batch processing")
        print(f"    Submit with: processor.submit_batch(tasks)")

        # Demo inline batch (smaller jobs)
        inline_tasks = []
        for i, topic in enumerate(sample_topics[:2]):  # Just 2 for demo
            task = processor.create_task(
                key=f"inline-{i}",
                model="gemini-2.5-flash",
                contents=f"Summarize in one sentence: {topic}",
            )
            inline_tasks.append(task)

        print(f"✅ Prepared {len(inline_tasks)} inline tasks")
        print(f"    Submit with: processor.submit_inline_batch(inline_tasks)")

    except Exception as e:
        print(f"Batch processing demo error: {e}")


if __name__ == "__main__":
    main()
