#!/usr/bin/env python3
"""Anthropic Claude Quickstart - Get started with Claude in LLM Studio"""

import os
from dotenv import load_dotenv

from llm_studio import (
    Agent,
    setup_logging,
    LogLevel,
    AnthropicBatchProcessor,
    UserMessage,
)
from llm_studio.tools.registry import list_all_tools
from llm_studio.cli.logging_cli import generate_log_filename


def main():
    """Quick Anthropic Claude setup and testing."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in .env file")
        print("   Add: ANTHROPIC_API_KEY=your-key-here")
        return

    # Enable logging with automatic file creation
    os.makedirs("logs", exist_ok=True)
    print("Created logs/ directory")

    # Setup logging with file output
    logger = setup_logging(level=LogLevel.INFO)
    log_file_path = generate_log_filename("claude", "claude-sonnet-4")

    # Create log file for session
    log_file = open(log_file_path, "w", encoding="utf-8")
    logger.log_file = log_file

    print(f"Logging enabled: {log_file_path}")

    # Create Claude agent
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        max_tokens=2048,  # Required for Anthropic
        system_prompt="You are a helpful assistant with access to web search and analysis tools.",
    )

    print(f"✅ Anthropic Claude agent created: {agent.provider_name}")

    # Show available Claude tools
    tools = list_all_tools()
    claude_tools = [name for name, type_info in tools.items() if "anthropic" in name]
    print(f"🔧 Available Claude tools: {len(claude_tools)}")
    for tool in claude_tools:
        print(f"   - {tool}")

    print(f"\nRunning comprehensive Claude tests...")

    # Test 1: Basic chat
    print(f"\nTest 1: Basic Chat")
    response = agent.generate("What is 2 + 2?")
    print(f"Response: {response.content}")

    # Test 2: Web search with citations
    print(f"\nTest 2: Web Search with Citations")
    response = agent.generate(
        "What are the latest developments in renewable energy technology?",
        tools=["anthropic_web_search"],
    )
    print(f"Response: {response.content[:400]}...")

    if response.grounding_metadata:
        print(f"✓ Grounding metadata: {list(response.grounding_metadata.keys())}")

        # Show search info
        if "web_search" in response.grounding_metadata:
            searches = response.grounding_metadata["web_search"]
            print(f"✓ Performed {len(searches)} web searches")

        # Show sources and citations
        if "sources" in response.grounding_metadata:
            sources = response.grounding_metadata["sources"]
            print(f"✓ Found {len(sources)} sources:")
            for i, source in enumerate(sources[:3]):  # Show first 3
                print(f"   {i+1}. {source}")

        if "citations" in response.grounding_metadata:
            citations = response.grounding_metadata["citations"]
            print(f"✓ Generated {len(citations)} citations")

        # Show usage info
        if "usage" in response.grounding_metadata:
            usage = response.grounding_metadata["usage"]
            print(
                f"✓ Token usage: {usage.get('input_tokens', 0)} input, {usage.get('output_tokens', 0)} output"
            )
            if "web_search_requests" in usage:
                print(f"✓ Search requests: {usage['web_search_requests']}")

    # Test 3: Local tool (should always work)
    print(f"\nTest 3: Local JSON Tool")
    response = agent.generate(
        "Format this data as JSON: name=Claude, provider=Anthropic, version=4",
        tools=["json_format"],
    )
    print(f"Response: {response.content[:200]}...")

    if response.grounding_metadata and "usage" in response.grounding_metadata:
        usage = response.grounding_metadata["usage"]
        session_usage = usage.get("session_usage", {})
        print(f"✓ Session tokens: {session_usage.get('session_input_tokens', 0)}")

    # Test 4: Token management demonstration
    print(f"\nTest 4: Token Management Demo")
    # Show current token usage
    if response.grounding_metadata and "usage" in response.grounding_metadata:
        usage = response.grounding_metadata["usage"]
        session_usage = usage.get("session_usage", {})
        total_tokens = session_usage.get("session_input_tokens", 0)
        print(f"📊 Current session usage: {total_tokens}/9500 tokens")
        print(f"📈 Requests made: {session_usage.get('session_requests', 0)}")
        print(
            f"✅ Token tracking: {'Under limit' if total_tokens < 9500 else 'Approaching limit'}"
        )

    # Test 5: Web search research workflow
    print(f"\nTest 5: Research Workflow with Web Search")
    try:
        # Add delay to avoid rate limits
        import time

        time.sleep(2)

        response = agent.generate(
            "Research the latest Python 3.12 features and provide a summary with citations. Focus on new language features.",
            tools=["anthropic_web_search"],
        )

        print(f"Research workflow response: {response.content[:400]}...")

        if response.grounding_metadata:
            tools_used = []
            if "web_search" in response.grounding_metadata:
                tools_used.append(
                    f"search ({len(response.grounding_metadata['web_search'])})"
                )
            print(f"✓ Tools used: {', '.join(tools_used)}")
    except Exception as e:
        print(f"⚠ Research workflow failed: {e}")
        print("  Note: May be rate limited or API access issue")

    # Test 5: Final token usage summary
    print(f"\nTest 5: Final Usage Summary")
    if response.grounding_metadata and "usage" in response.grounding_metadata:
        usage = response.grounding_metadata["usage"]
        session_usage = usage.get("session_usage", {})
        print(f"📊 Final session statistics:")
        print(f"   Total input tokens: {session_usage.get('session_input_tokens', 0)}")
        print(
            f"   Total output tokens: {session_usage.get('session_output_tokens', 0)}"
        )
        print(f"   Total requests: {session_usage.get('session_requests', 0)}")
        print(f"   Token limit: 9500 per minute")
        print(
            f"   Status: {'✅ Under limit' if session_usage.get('session_input_tokens', 0) < 9500 else '⚠ Approaching limit'}"
        )

    # Test 6: Batch Processing Demo
    print(f"\nTest 6: Batch Processing Demo")
    try:
        processor = AnthropicBatchProcessor(api_key=api_key)

        # Create sample batch requests
        sample_topics = [
            "Artificial intelligence in healthcare",
            "Renewable energy adoption trends",
            "Quantum computing applications",
        ]

        requests = []
        for i, topic in enumerate(sample_topics):
            request = processor.create_request(
                custom_id=f"analysis-{i}",
                model="claude-sonnet-4-20250514",
                messages=[UserMessage(f"Provide a brief analysis of: {topic}")],
                system="You are an expert analyst. Provide concise, insightful analysis.",
                max_tokens=1024,
                temperature=0.2,
            )
            requests.append(request)

        print(f"✅ Created {len(requests)} batch requests")
        print(f"    Ready for submission with: processor.create_batch_job(requests)")
        print(f"    Note: Batch processing offers 50% cost savings")

        # Demo request structure
        sample_request = requests[0]
        print(f"✅ Sample request structure:")
        print(f"    Custom ID: {sample_request.custom_id}")
        print(f"    Model: {sample_request.model}")
        print(f"    Max tokens: {sample_request.max_tokens}")

    except Exception as e:
        print(f"Batch processing demo error: {e}")

    # Close log file
    if logger.log_file:
        logger.log_file.close()
        logger.log_file = None

    print(f"\n✅ Anthropic Claude quickstart complete!")
    print(f"🔍 Web search with automatic citations")
    print(f"📄 Web content fetching and analysis")
    print(f"💻 Code execution with bash and file operations (beta)")
    print(f"📊 Research workflows with rich metadata")
    print(f"🛠️ All server-side tools integrated")
    print(f"📁 Batch processing for large-scale operations")
    print(f"Session logged to: {log_file_path}")
    print(f"Full documentation: CLAUDE.md")


if __name__ == "__main__":
    main()
