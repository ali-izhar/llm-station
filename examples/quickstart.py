#!/usr/bin/env python3
"""
Quick start example for llm_studio.

Demonstrates basic agent usage with the mock provider.
"""

from llm_studio import Agent


def main() -> None:
    print("🚀 LLM Studio Quickstart")
    print("=" * 30)

    # Use mock provider for offline demonstration
    agent = Agent(
        provider="mock", model="mock-001", system_prompt="You are a helpful assistant."
    )

    # Simple echo behavior example
    print("1. Basic conversation:")
    result = agent.generate(prompt="Hello! How are you?")
    print(f"   User: Hello! How are you?")
    print(f"   Agent: {result.content}")
    print()

    # Tool call demo using naive pattern the mock understands
    print("2. Tool calling:")
    result2 = agent.generate(
        prompt='Please call fetch_url({"url":"https://httpbin.org/json"}) and analyze.',
        tools=["fetch_url"],
    )
    print(f"   User: Please call fetch_url...")
    print(f"   Agent: {result2.content}")
    if result2.tool_calls:
        print(f"   Tool calls: {[tc.name for tc in result2.tool_calls]}")
        for tc in result2.tool_calls:
            print(f"     - {tc.name}: {tc.arguments}")
    print()

    print("✅ Quickstart complete!")
    print("💡 Try 'python examples/agent_demo.py' for a comprehensive demo.")


if __name__ == "__main__":
    main()
