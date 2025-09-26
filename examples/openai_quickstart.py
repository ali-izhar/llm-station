#!/usr/bin/env python3
"""OpenAI Quickstart - Get started with OpenAI in LLM Station"""

import os
from dotenv import load_dotenv

from llm_station import (
    Agent,
    setup_logging,
    LogLevel,
    OpenAIBatchProcessor,
    SystemMessage,
    UserMessage,
)
from llm_station import get_available_tools
from llm_station.cli.logging_cli import generate_log_filename


def main():
    """Quick OpenAI setup and testing."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("   Add: OPENAI_API_KEY=your-key-here")
        return

    # Enable logging with automatic file creation
    os.makedirs("logs", exist_ok=True)
    print("Created logs/ directory")

    # Setup logging with file output
    logger = setup_logging(level=LogLevel.INFO)
    log_file_path = generate_log_filename("openai", "gpt-4o-mini")

    # Create log file for session
    log_file = open(log_file_path, "w", encoding="utf-8")
    logger.log_file = log_file

    print(f"Logging enabled: {log_file_path}")

    # Create agent
    agent = Agent(
        provider="openai",
        model="gpt-4o-mini",
        api_key=api_key,
        system_prompt="You are a helpful assistant.",
    )

    print(f"✅ OpenAI agent created: {agent.provider_name}")

    # Show available smart tools
    tools = get_available_tools()
    smart_tools = [name for name, type_info in tools.items() if type_info == "smart"]
    primary_tools = ["search", "code", "image", "json", "fetch", "url"]
    available_primary = [t for t in primary_tools if t in smart_tools]
    print(f"🔧 Available smart tools: {len(available_primary)}")
    for tool in available_primary:
        print(f"   - {tool}")

    print(f"\nRunning comprehensive tests...")

    # Test 1: Basic chat
    print(f"\nTest 1: Basic Chat")
    response = agent.generate("What is 2 + 2?")
    print(f"Response: {response.content}")

    # Test 2: Web search (using generic "search" tool)
    print(f"\nTest 2: Web Search")
    response = agent.generate("What's happening in AI news today?", tools=["search"])
    print(f"Response: {response.content[:200]}...")
    if response.grounding_metadata:
        print(f"✓ Search metadata: {list(response.grounding_metadata.keys())}")

    # Test 3: Code execution (using generic "code" tool)
    print(f"\nTest 3: Code Execution")
    response = agent.generate(
        "Calculate the factorial of 5 using Python", tools=["code"]
    )
    print(f"Response: {response.content[:200]}...")
    if response.grounding_metadata:
        print(f"✓ Code metadata: {list(response.grounding_metadata.keys())}")

    # Test 4: Image generation (using generic "image" tool)
    print(f"\nTest 4: Image Generation")

    # Try with different models that support image generation
    image_models = ["gpt-5", "gpt-4.1", "gpt-4o"]

    for model in image_models:
        print(f"\n   Testing with {model}...")
        try:
            # Create agent with image-compatible model
            image_agent = Agent(
                provider="openai",
                model=model,
                api_key=api_key,
                system_prompt="You are a helpful assistant.",
            )

            response = image_agent.generate("Draw a simple red circle", tools=["image"])

            print(f"   Response: {response.content[:150]}...")

            if (
                response.grounding_metadata
                and "image_generation" in response.grounding_metadata
            ):
                images = response.grounding_metadata["image_generation"]
                print(f"   ✅ {model}: Generated {len(images)} images")

                # Save the first successful image
                if images:
                    import base64

                    image_data = base64.b64decode(images[0]["result"])
                    with open(f"test_circle_{model.replace('-', '_')}.png", "wb") as f:
                        f.write(image_data)
                    print(f"    Saved test_circle_{model.replace('-', '_')}.png")
                    break  # Stop after first success
            else:
                print(f"    {model}: No image metadata generated")

        except Exception as e:
            print(f"   ❌ {model}: {str(e)[:100]}...")
            continue
    else:
        print(
            "   Image generation may require specific model access or API organization verification"
        )

    # Test 5: Batch Processing (demo)
    print(f"\nTest 5: Batch Processing Demo")
    try:
        processor = OpenAIBatchProcessor(api_key=api_key)

        # Create sample batch tasks
        sample_texts = [
            "Summarize: The future of AI is bright with many opportunities.",
            "Categorize: This is a positive review of our AI product.",
            "Analyze: Customer feedback shows high satisfaction with our service.",
        ]

        tasks = []
        for i, text in enumerate(sample_texts):
            task = processor.create_task(
                custom_id=f"quickstart-{i}",
                model="gpt-4o-mini",
                messages=[
                    SystemMessage("You are a helpful assistant."),
                    UserMessage(text),
                ],
                temperature=0.1,
            )
            tasks.append(task)

        # Create batch file (don't submit to avoid costs)
        batch_file = processor.create_batch_file(tasks, "quickstart_batch.jsonl")
        print(f"✅ Created batch file: {batch_file}")
        print(f"    {len(tasks)} tasks ready for batch processing")
        print(f"    Submit with: processor.submit_batch(tasks)")

    except Exception as e:
        print(f"Batch processing demo: {e}")

    # Close log file
    if logger.log_file:
        logger.log_file.close()
        logger.log_file = None

    print(f"\n✅ OpenAI quickstart complete!")
    print(f"Session logged to: {log_file_path}")
    print(f"Batch example: quickstart_batch.jsonl")
    print(f"Full documentation: OPENAI.md")


if __name__ == "__main__":
    main()
