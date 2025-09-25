#!/usr/bin/env python3
"""OpenAI Quickstart - Get started with OpenAI in LLM Studio"""

import os
from dotenv import load_dotenv

from llm_studio import Agent, setup_logging, LogLevel
from llm_studio.tools.registry import list_all_tools


def main():
    """Quick OpenAI setup and testing."""
    print("🚀 OpenAI Quickstart")
    print("=" * 40)

    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("   Add: OPENAI_API_KEY=your-key-here")
        return

    # Enable logging
    setup_logging(level=LogLevel.INFO)

    # Create agent
    agent = Agent(
        provider="openai",
        model="gpt-4o-mini",
        api_key=api_key,
        system_prompt="You are a helpful assistant.",
    )

    print(f"✅ OpenAI agent created: {agent.provider_name}")

    # Show available tools
    tools = list_all_tools()
    openai_tools = [name for name, type_info in tools.items() if "openai" in name]
    print(f"🔧 Available OpenAI tools: {len(openai_tools)}")
    for tool in openai_tools:
        print(f"   - {tool}")

    # Test 1: Basic chat
    print(f"\n💬 Test 1: Basic Chat")
    response = agent.generate("What is 2 + 2?")
    print(f"Response: {response.content}")

    # Test 2: Web search
    print(f"\n🔍 Test 2: Web Search")
    response = agent.generate(
        "What's happening in AI news today?", tools=["openai_web_search"]
    )
    print(f"Response: {response.content[:200]}...")
    if response.grounding_metadata:
        print(f"✓ Search metadata: {list(response.grounding_metadata.keys())}")

    # Test 3: Code execution
    print(f"\n🐍 Test 3: Code Execution")
    response = agent.generate(
        "Calculate the factorial of 5 using Python", tools=["openai_code_interpreter"]
    )
    print(f"Response: {response.content[:200]}...")
    if response.grounding_metadata:
        print(f"✓ Code metadata: {list(response.grounding_metadata.keys())}")

    # Test 4: Image generation
    print(f"\n🎨 Test 4: Image Generation")
    response = agent.generate(
        "Draw a simple red circle", tools=["openai_image_generation"]
    )
    print(f"Response: {response.content[:200]}...")
    if (
        response.grounding_metadata
        and "image_generation" in response.grounding_metadata
    ):
        images = response.grounding_metadata["image_generation"]
        print(f"✓ Generated {len(images)} images")

    print(f"\n✅ OpenAI quickstart complete!")
    print(f"📁 Check logs/ directory for session logs")
    print(f"📖 See OPENAI.md for full documentation")


if __name__ == "__main__":
    main()
