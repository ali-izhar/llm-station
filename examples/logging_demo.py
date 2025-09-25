#!/usr/bin/env python3
"""
Agent Logging System Demo

Demonstrates comprehensive logging of agent interactions with different
verbosity levels and output formats.

Shows:
- Input queries and tool selection
- Provider API calls and responses
- Tool execution with inputs/outputs
- Final results and metadata
- Step-by-step thought process
"""

import os
from dotenv import load_dotenv
from llm_studio import Agent, setup_logging, LogLevel, LogFormat
from llm_studio.tools.web_search.openai import OpenAIWebSearch
from llm_studio.tools.code_execution.openai import OpenAICodeInterpreter


def demo_basic_logging():
    """Demo basic logging with standard verbosity."""
    print("🔍 DEMO 1: Basic Logging (Standard Level)")
    print("=" * 60)

    # Setup info logging
    logger = setup_logging(level=LogLevel.INFO, format=LogFormat.CONSOLE)

    # Create agent
    load_dotenv()
    agent = Agent(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt="You are a helpful math assistant.",
    )

    # Make a request with logging
    result = agent.generate(
        "What is 15 * 24? Please calculate this step by step.", tools=["json_format"]
    )

    print(f"\nFinal result: {result.content}")


def demo_detailed_logging():
    """Demo detailed logging with tool calls."""
    print("\n🔍 DEMO 2: Detailed Logging with Tools")
    print("=" * 60)

    # Setup debug logging
    logger = setup_logging(level=LogLevel.DEBUG, format=LogFormat.CONSOLE)

    load_dotenv()
    agent = Agent(
        provider="openai",
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt="You are a research assistant with access to web search and code execution.",
    )

    # Complex request with multiple tools
    result = agent.generate(
        "Search for the current price of Bitcoin and calculate what a $1000 investment would be worth",
        tools=["openai_web_search", "openai_code_interpreter"],
    )

    print(f"\nFinal result: {result.content}")


def demo_debug_logging():
    """Demo debug logging with all details."""
    print("\n🔍 DEMO 3: Debug Logging (Maximum Detail)")
    print("=" * 60)

    # Setup debug logging
    logger = setup_logging(level=LogLevel.DEBUG, format=LogFormat.CONSOLE)

    load_dotenv()
    agent = Agent(
        provider="openai", model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create advanced web search tool
    search_tool = OpenAIWebSearch(
        allowed_domains=["coindesk.com", "bloomberg.com"],
        user_location={"country": "US", "city": "New York"},
    )

    result = agent.generate(
        "Find the latest cryptocurrency news from financial sources",
        tools=[search_tool.spec()],
    )

    print(f"\nFinal result: {result.content[:200]}...")


def demo_json_logging():
    """Demo JSON structured logging."""
    print("\n🔍 DEMO 4: JSON Structured Logging")
    print("=" * 60)

    # Setup JSON logging
    logger = setup_logging(level=LogLevel.INFO, format=LogFormat.JSON)

    load_dotenv()
    agent = Agent(
        provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
    )

    result = agent.generate(
        "Format this data: name=Alice, age=30, city=Boston", tools=["json_format"]
    )

    # Export session as JSON
    if logger.current_session:
        session_json = logger.export_session(LogFormat.JSON)
        print("\n📊 Session JSON Export:")
        print(session_json[:500] + "..." if len(session_json) > 500 else session_json)


def demo_markdown_export():
    """Demo markdown export for documentation."""
    print("\n🔍 DEMO 5: Markdown Export")
    print("=" * 60)

    # Setup logging for markdown export
    logger = setup_logging(level=LogLevel.DEBUG, format=LogFormat.CONSOLE)

    load_dotenv()
    agent = Agent(
        provider="openai",
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt="You are a data scientist.",
    )

    # Create code interpreter
    code_tool = OpenAICodeInterpreter(container_type="auto")

    result = agent.generate(
        "Create a simple data visualization showing monthly sales: Jan=100, Feb=150, Mar=120",
        tools=[code_tool.spec()],
    )

    # Export as markdown
    if logger.current_session:
        markdown = logger.export_session(LogFormat.MARKDOWN)
        print("\n📝 Markdown Export:")
        print(markdown[:800] + "..." if len(markdown) > 800 else markdown)


def demo_context_manager():
    """Demo using logging with context manager."""
    print("\n🔍 DEMO 6: Context Manager Logging")
    print("=" * 60)

    from llm_studio.logging import AgentLoggerContext, AgentLogger

    load_dotenv()
    logger = AgentLogger(level=LogLevel.INFO, format=LogFormat.CONSOLE)

    # Use context manager for automatic session management
    with AgentLoggerContext(
        logger=logger,
        provider="openai",
        model="gpt-4o-mini",
        input_query="Calculate fibonacci sequence up to 10 numbers",
        tools_requested=["openai_code_interpreter"],
        system_prompt="You are a math tutor.",
    ) as ctx_logger:

        agent = Agent(
            provider="openai",
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            system_prompt="You are a math tutor.",
        )

        result = agent.generate(
            "Calculate fibonacci sequence up to 10 numbers",
            tools=["openai_code_interpreter"],
        )

    print(f"\nContext manager completed. Result: {result.content[:100]}...")


def demo_logging_levels():
    """Demo different logging levels side by side."""
    print("\n🔍 DEMO 7: Logging Levels Comparison")
    print("=" * 60)

    load_dotenv()

    levels = [LogLevel.ERROR, LogLevel.INFO, LogLevel.DEBUG]

    for level in levels:
        print(f"\n--- {level.value.upper()} LEVEL ---")

        logger = setup_logging(level=level, format=LogFormat.CONSOLE)
        agent = Agent(
            provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
        )

        result = agent.generate(
            "What is the square root of 144?", tools=["json_format"]
        )


def main():
    """Run all logging demos."""
    print("🚀 Agent Logging System Demonstration")
    print("=" * 80)
    print("This demo shows the comprehensive logging capabilities.")
    print("⚠ Note: Some demos make real API calls if OPENAI_API_KEY is set.")
    print("=" * 80)

    try:
        demo_basic_logging()
        demo_detailed_logging()
        demo_debug_logging()
        demo_json_logging()
        demo_markdown_export()
        demo_context_manager()
        demo_logging_levels()

        print("\n✅ All logging demos completed successfully!")
        print("\n💡 Usage Tips:")
        print("  - Use LogLevel.INFO for normal development")
        print("  - Use LogLevel.DEBUG for debugging tool issues")
        print("  - Use LogLevel.DEBUG for full API inspection")
        print("  - Export sessions as JSON/Markdown for documentation")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure OPENAI_API_KEY is set in .env for API demos")


if __name__ == "__main__":
    main()
