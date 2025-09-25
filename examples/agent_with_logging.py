#!/usr/bin/env python3
"""
Agent with Logging Example

Demonstrates how to use the --log flag and logging system for
comprehensive agent interaction tracking.

Usage:
    python agent_with_logging.py --log
    python agent_with_logging.py --log --log-level detailed
    python agent_with_logging.py --log --log-format json --log-file session.log
"""

import argparse
import os
import sys
from dotenv import load_dotenv

from llm_studio import Agent, setup_logging, LogLevel, LogFormat
from llm_studio.cli import add_logging_args, configure_logging_from_args
from llm_studio.tools.web_search.openai import OpenAIWebSearch
from llm_studio.tools.code_execution.openai import OpenAICodeInterpreter
from llm_studio.tools.image_generation.openai import OpenAIImageGeneration


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with logging options."""
    parser = argparse.ArgumentParser(
        description="LLM Studio Agent with Comprehensive Logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic chat with logging
  python agent_with_logging.py -l "Hello, how are you?"
  
  # Research with warning-level logging  
  python agent_with_logging.py -l --log-level warn \\
    "Research AI trends and analyze the data"
    
  # Complex workflow with debug logging
  python agent_with_logging.py -l --log-level debug \\
    "Search for climate data, analyze with Python, create visualization"
    
  # Save logs to custom file
  python agent_with_logging.py -lf research.log \\
    "Research quantum computing developments"
    
  # JSON logs (auto-saved to logs/ directory)
  python agent_with_logging.py -l --log-format json \\
    "Analyze data and create visualization"
        """,
    )

    parser.add_argument(
        "query",
        nargs="?",
        default="What are the latest developments in AI?",
        help="Query to send to the agent (default: AI developments question)",
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google", "mock"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )

    parser.add_argument(
        "--model", help="Model name (default: provider-specific default)"
    )

    parser.add_argument(
        "--tools",
        nargs="*",
        help="Tools to make available (e.g., openai_web_search openai_code_interpreter)",
    )

    parser.add_argument("--system-prompt", help="System prompt for the agent")

    # Add logging arguments
    add_logging_args(parser)

    return parser


def get_default_model(provider: str) -> str:
    """Get default model for provider."""
    defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "google": "gemini-1.5-flash",
        "mock": "mock-001",
    }
    return defaults.get(provider, "gpt-4o-mini")


def get_api_key(provider: str) -> str:
    """Get API key for provider."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
    }

    if provider == "mock":
        return "mock-key"

    key_name = key_map.get(provider)
    if not key_name:
        raise ValueError(f"Unknown provider: {provider}")

    key = os.getenv(key_name)
    if not key:
        raise ValueError(f"{key_name} not found in environment variables")

    return key


def infer_tools_from_query(query: str) -> list:
    """Intelligently infer tools needed based on query content."""
    query_lower = query.lower()
    tools = []

    # Check for web search keywords
    search_keywords = [
        "search",
        "find",
        "news",
        "research",
        "latest",
        "current",
        "recent",
    ]
    if any(keyword in query_lower for keyword in search_keywords):
        tools.append("openai_web_search")

    # Check for code/calculation keywords
    code_keywords = [
        "calculate",
        "analyze",
        "data",
        "python",
        "code",
        "math",
        "statistics",
    ]
    if any(keyword in query_lower for keyword in code_keywords):
        tools.append("openai_code_interpreter")

    # Check for image/visual keywords
    image_keywords = ["image", "draw", "create", "visual", "chart", "graph", "diagram"]
    if any(keyword in query_lower for keyword in image_keywords):
        tools.append("openai_image_generation")

    # Check for data formatting keywords
    format_keywords = ["json", "format", "structure"]
    if any(keyword in query_lower for keyword in format_keywords):
        tools.append("json_format")

    return tools


def main():
    """Main CLI interface."""
    parser = create_parser()
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    try:
        # Get model and API key
        model = args.model or get_default_model(args.provider)
        api_key = get_api_key(args.provider)

        # Configure logging with provider/model info for auto-filename generation
        cleanup_logging = configure_logging_from_args(args, args.provider, model)

        # Create agent
        agent = Agent(
            provider=args.provider,
            model=model,
            api_key=api_key,
            system_prompt=args.system_prompt,
        )

        # Determine tools to use
        tools = args.tools
        if not tools:
            # Auto-infer tools from query
            tools = infer_tools_from_query(args.query)
            if tools and getattr(args, "log", False):
                print(f"🔧 Auto-detected tools: {', '.join(tools)}")

        # Generate response
        print(f"\n🤖 Processing query: {args.query}")
        if tools:
            print(f"🔧 Using tools: {', '.join(tools)}")

        result = agent.generate(args.query, tools=tools)

        print(f"\n✅ Final Result:")
        print(f"{result.content}")

        # Show metadata if available
        if result.grounding_metadata:
            print(f"\n📊 Metadata Available:")
            for key, value in result.grounding_metadata.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {type(value).__name__}")

    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        # Cleanup logging
        if cleanup_logging:
            cleanup_logging()


if __name__ == "__main__":
    main()
