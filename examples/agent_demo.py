#!/usr/bin/env python3
"""
Comprehensive demo of llm_studio agent functionality.

This script demonstrates:
1. Basic agent usage with the mock provider
2. Tool calling with local tools
3. Provider-specific tools (web search, code execution, etc.)
4. Different providers and their capabilities
"""

from llm_studio import Agent
from llm_studio.tools.web_search import (
    OpenAIWebSearch,
    AnthropicWebSearch,
    GoogleWebSearch,
)
from llm_studio.tools.web_fetch import AnthropicWebFetch
from llm_studio.tools.code_execution import GoogleCodeExecution
from llm_studio.tools.url_context import GoogleUrlContext


def demo_basic_agent():
    """Demo basic agent functionality with mock provider."""
    print("🤖 DEMO 1: Basic Agent (Mock Provider)")
    print("=" * 50)

    agent = Agent(
        provider="mock", model="mock-001", system_prompt="You are a helpful assistant."
    )

    # Simple echo behavior
    result = agent.generate("Hello! What's the weather like today?")
    print(f"User: Hello! What's the weather like today?")
    print(f"Agent: {result.content}")
    print()


def demo_tool_calling():
    """Demo tool calling with local tools."""
    print("🔧 DEMO 2: Tool Calling (Mock Provider + Local Tools)")
    print("=" * 60)

    agent = Agent(
        provider="mock",
        model="mock-001",
        system_prompt="You are a helpful assistant that uses tools.",
    )

    # Tool call using the mock provider's pattern recognition
    result = agent.generate(
        prompt='Please call fetch_url({"url":"https://httpbin.org/json"}) to get some data.',
        tools=["fetch_url", "json_format"],
    )

    print(f"User: Please call fetch_url...")
    print(f"Agent content: {result.content}")
    if result.tool_calls:
        print(f"Tool calls made: {[tc.name for tc in result.tool_calls]}")
        for tc in result.tool_calls:
            print(f"  - {tc.name}: {tc.arguments}")
    print()


def demo_provider_tools():
    """Demo provider-specific tools (without actual network calls)."""
    print("🌐 DEMO 3: Provider-Specific Tools")
    print("=" * 40)

    # OpenAI Web Search
    print("OpenAI Web Search Tool:")
    openai_ws = OpenAIWebSearch(
        allowed_domains=["openai.com", "github.com"],
        user_location={"country": "US"},
    ).spec()
    print(f"  Provider: {openai_ws.provider}")
    print(f"  Type: {openai_ws.provider_type}")
    print(f"  Config: {openai_ws.provider_config}")
    print()

    # Anthropic Web Search + Web Fetch
    print("Anthropic Web Tools:")
    anthropic_ws = AnthropicWebSearch(
        allowed_domains=["anthropic.com"], max_uses=5
    ).spec()
    anthropic_wf = AnthropicWebFetch(
        max_content_tokens=2000, citations={"enabled": True}
    ).spec()
    print(
        f"  Web Search - Provider: {anthropic_ws.provider}, Type: {anthropic_ws.provider_type}"
    )
    print(
        f"  Web Fetch - Provider: {anthropic_wf.provider}, Type: {anthropic_wf.provider_type}"
    )
    print()

    # Google Tools
    print("Google/Gemini Tools:")
    google_ws = GoogleWebSearch().spec()
    google_ce = GoogleCodeExecution().spec()
    google_uc = GoogleUrlContext().spec()
    print(f"  Web Search: {google_ws.provider_type}")
    print(f"  Code Execution: {google_ce.provider_type}")
    print(f"  URL Context: {google_uc.provider_type}")
    print()


def demo_agent_configurations():
    """Demo different agent configurations."""
    print("⚙️  DEMO 4: Different Agent Configurations")
    print("=" * 50)

    # Mock provider with custom config
    mock_agent = Agent(
        provider="mock",
        model="mock-001",
        system_prompt="You are a creative writing assistant.",
        temperature=0.8,
        max_tokens=100,
    )
    print(f"Mock Agent - Provider: {mock_agent.provider_name}")

    # Note: These would work with real API keys
    print("\nOther provider examples (require API keys):")
    print("  OpenAI Agent: Agent(provider='openai', model='gpt-4', api_key='...')")
    print(
        "  Anthropic Agent: Agent(provider='anthropic', model='claude-3-sonnet', api_key='...')"
    )
    print(
        "  Google Agent: Agent(provider='google', model='gemini-1.5-pro', api_key='...')"
    )
    print()


def demo_tool_mixing():
    """Demo mixing local and provider-specific tools."""
    print("🔄 DEMO 5: Mixed Tool Usage")
    print("=" * 35)

    agent = Agent(
        provider="mock",
        model="mock-001",
        system_prompt="You are a research assistant with access to various tools.",
    )

    # Mix of local tools and provider tools
    local_tools = ["fetch_url", "json_format"]
    provider_tools = [OpenAIWebSearch().spec(), GoogleCodeExecution().spec()]

    all_tools = local_tools + provider_tools

    print(f"Available tools: {len(all_tools)} total")
    print("  Local tools:", local_tools)
    print("  Provider tools:", [t.name for t in provider_tools])

    # Simulate a complex query (would work with real providers)
    result = agent.generate(
        prompt="Research the latest AI developments and create a summary report.",
        tools=all_tools,
    )
    print(f"\nAgent response: {result.content}")
    print()


def demo_dynamic_agent_usage():
    """Demo dynamic agent creation and tool testing."""
    print("🎯 DEMO 6: Dynamic Agent Usage & Tool Testing")
    print("=" * 55)

    # Dynamic provider selection
    providers = ["mock", "openai", "anthropic", "google"]
    selected_provider = "mock"  # In real usage, this could be user input

    print(f"Creating agent with provider: {selected_provider}")
    agent = Agent(
        provider=selected_provider,
        model=(
            "mock-001" if selected_provider == "mock" else f"{selected_provider}-model"
        ),
        system_prompt="You are an AI assistant with tool capabilities.",
        temperature=0.7,
    )
    print(f"✅ Agent created: {agent.provider_name}")
    print()

    # Test local tools dynamically
    print("🔧 Testing Local Tools:")
    local_tools = ["json_format", "fetch_url"]

    for tool_name in local_tools:
        print(f"  Testing {tool_name}...")
        if tool_name == "json_format":
            result = agent.generate(
                prompt=f'call {tool_name}({{"data": {{"test": "value", "number": 42}}}})',
                tools=[tool_name],
                max_tool_rounds=1,
            )
        else:  # fetch_url
            result = agent.generate(
                prompt=f'call {tool_name}({{"url": "https://httpbin.org/json"}})',
                tools=[tool_name],
                max_tool_rounds=1,
            )

        print(
            f"    ✓ Tool calls detected: {len(result.tool_calls) if result.tool_calls else 0}"
        )
        if result.tool_calls:
            print(f"      Tool: {result.tool_calls[0].name}")
            print(f"      Args: {result.tool_calls[0].arguments}")
    print()

    # Test provider-specific tools
    print("🌐 Testing Provider-Specific Tools:")
    provider_tool_configs = {
        "openai": OpenAIWebSearch(allowed_domains=["openai.com"]).spec(),
        "anthropic": AnthropicWebSearch(max_uses=3).spec(),
        "google": GoogleCodeExecution().spec(),
    }

    for provider, tool_spec in provider_tool_configs.items():
        print(f"  {provider.capitalize()} Tool: {tool_spec.name}")
        print(f"    Provider: {tool_spec.provider}")
        print(f"    Type: {tool_spec.provider_type}")
        print(f"    Config: {tool_spec.provider_config}")
    print()


def demo_real_world_scenarios():
    """Demo real-world usage scenarios."""
    print("🌍 DEMO 7: Real-World Usage Scenarios")
    print("=" * 45)

    scenarios = [
        {
            "name": "Content Research",
            "description": "Research and summarize information from multiple sources",
            "provider": "google",
            "tools": [GoogleWebSearch().spec(), GoogleUrlContext().spec()],
            "prompt": "Research recent developments in AI safety and provide a summary",
        },
        {
            "name": "Data Analysis",
            "description": "Analyze data with code and web context",
            "provider": "google",
            "tools": [GoogleCodeExecution().spec(), GoogleUrlContext().spec()],
            "prompt": "Analyze the data from these CSV URLs and create visualizations",
        },
        {
            "name": "Web Research",
            "description": "Domain-filtered web search with citations",
            "provider": "openai",
            "tools": [
                OpenAIWebSearch(allowed_domains=["arxiv.org", "nature.com"]).spec()
            ],
            "prompt": "Find recent AI research papers and summarize key findings",
        },
        {
            "name": "Content Fetching",
            "description": "Fetch and analyze specific web content",
            "provider": "anthropic",
            "tools": [AnthropicWebFetch(max_content_tokens=1500).spec()],
            "prompt": "Fetch content from documentation URLs and explain the concepts",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Provider: {scenario['provider']}")
        print(f"   Tools: {[t.name for t in scenario['tools']]}")
        print(f"   Example prompt: {scenario['prompt']}")
        print()

    print("💡 Note: These scenarios work with real API keys!")
    print("   Replace 'mock' provider with actual providers and API keys.")
    print()


def demo_advanced_features():
    """Demo advanced features like tool choice and configuration."""
    print("⚡ DEMO 8: Advanced Features")
    print("=" * 35)

    # Advanced model configuration
    print("🔧 Advanced Model Configuration:")
    advanced_configs = {
        "OpenAI Responses API": {
            "provider": "openai",
            "model": "gpt-5",
            "api": "responses",
            "reasoning": {"effort": "high"},
            "include": ["web_search_call.action.sources"],
            "tool_choice": "auto",
        },
        "Anthropic with Tool Choice": {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
            "temperature": 0.3,
        },
        "Google Legacy Search": {
            "provider": "google",
            "model": "gemini-1.5-flash",
            "temperature": 0.8,
        },
    }

    for name, config in advanced_configs.items():
        print(f"  {name}:")
        for key, value in config.items():
            print(f"    {key}: {value}")
        print()

    # Tool choice examples
    print("🎛️  Tool Choice Options:")
    tool_choices = {
        "auto": "Let model decide whether to use tools",
        "any": "Force model to use at least one tool",
        "none": "Disable all tool usage",
        "specific_tool": "Force model to use a specific tool",
    }

    for choice, description in tool_choices.items():
        print(f"  {choice}: {description}")
    print()


def demo_interactive_testing():
    """Demo interactive tool testing."""
    print("🧪 DEMO 9: Interactive Tool Testing")
    print("=" * 40)

    print("You can test tools interactively like this:")
    print()

    # Show how to test individual tools
    test_examples = [
        {
            "tool": "json_format",
            "test_input": '{"data": {"users": ["Alice", "Bob"], "count": 2}}',
            "description": "Test JSON formatting with sample data",
        },
        {
            "tool": "fetch_url",
            "test_input": '{"url": "https://httpbin.org/headers"}',
            "description": "Test URL fetching with httpbin service",
        },
    ]

    agent = Agent(provider="mock", model="mock-001", system_prompt="Test assistant")

    for test in test_examples:
        print(f"Testing {test['tool']}: {test['description']}")
        print(f"  Command: call {test['tool']}({test['test_input']})")

        result = agent.generate(
            prompt=f"call {test['tool']}({test['test_input']})",
            tools=[test["tool"]],
            max_tool_rounds=1,
        )

        print(
            f"  Result: {len(result.tool_calls) if result.tool_calls else 0} tool calls detected"
        )
        if result.tool_calls:
            tc = result.tool_calls[0]
            print(f"    Tool: {tc.name}")
            print(f"    Args: {tc.arguments}")
        print()

    print("💡 Pro tip: Use patterns like 'call toolname({...})' with mock provider")
    print("   Real providers understand natural language requests!")
    print()


def main():
    """Run all demos."""
    print("🚀 LLM Studio Agent Demonstration")
    print("=" * 70)
    print("This demo shows the agent capabilities using the mock provider.")
    print("Real providers (OpenAI, Anthropic, Google) would need API keys.")
    print("=" * 70)
    print()

    demo_basic_agent()
    demo_tool_calling()
    demo_provider_tools()
    demo_agent_configurations()
    demo_tool_mixing()
    demo_dynamic_agent_usage()
    demo_real_world_scenarios()
    demo_advanced_features()
    demo_interactive_testing()

    print("✅ Demo complete! The framework is working correctly.")
    print("🔑 Next steps:")
    print("   1. Get API keys for OpenAI, Anthropic, and/or Google")
    print("   2. Wire real providers by replacing RuntimeError calls with SDK calls")
    print("   3. Test with real models and tools")
    print("   4. Build your AI applications!")
    print()
    print("📖 See README.md for installation and usage instructions.")


if __name__ == "__main__":
    main()
