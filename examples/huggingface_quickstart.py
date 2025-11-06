#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace Provider Quick Start Example

This example demonstrates how to use the HuggingFace provider with web search
capabilities via SerpAPI integration.

Setup:
1. Set environment variables:
   - HUGGING_FACE_API_TOKEN: Your HuggingFace API token
   - HUGGING_FACE_ENDPOINT_URL: Your HuggingFace Inference Endpoint URL
   - SERPAPI_API_KEY: Your SerpAPI key (optional, for web search)

2. Or pass them as parameters to Agent initialization.
"""

import os
import sys
from dotenv import load_dotenv
from llm_station import Agent

# Set UTF-8 encoding for Windows terminal
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Load environment variables
load_dotenv()


def main():
    """Demonstrate HuggingFace provider usage."""
    
    # Example 1: Basic usage without web search
    print("=" * 60)
    print("Example 1: Basic HuggingFace Usage (No Web Search)")
    print("=" * 60)
    
    agent = Agent(
        provider="huggingface",
        model="openai/gpt-oss-20b",  # Model name
        api_key=os.getenv("HUGGING_FACE_API_TOKEN"),
        # Pass provider-specific parameters via **kwargs
        endpoint_url=os.getenv("HUGGING_FACE_ENDPOINT_URL"),
        # Optional: configure reasoning level (used during generation)
        reasoning_level="medium",  # "low", "medium", or "high"
    )
    
    response = agent.generate("Explain quantum computing in simple terms.")
    print("\nResponse:")
    print(response.content)
    print("\n")
    
    # Example 2: With web search enabled
    print("=" * 60)
    print("Example 2: HuggingFace with Web Search")
    print("=" * 60)
    
    agent_with_search = Agent(
        provider="huggingface",
        model="openai/gpt-oss-20b",
        api_key=os.getenv("HUGGING_FACE_API_TOKEN"),
        # Pass provider-specific parameters via **kwargs
        endpoint_url=os.getenv("HUGGING_FACE_ENDPOINT_URL"),
        serpapi_key=os.getenv("SERPAPI_API_KEY"),  # Required for web search
        reasoning_level="high",  # Used during generation
    )
    
    # Use the search tool
    response = agent_with_search.generate(
        "What are the latest developments in artificial intelligence?",
        tools=["search"]  # Smart tool routing will use huggingface_web_search
    )
    
    print("\nResponse:")
    print(response.content)
    
    # Check for sources in grounding metadata
    if response.grounding_metadata and "sources" in response.grounding_metadata:
        sources = response.grounding_metadata["sources"]
        print(f"\nFound {len(sources)} sources:")
        for i, source in enumerate(sources[:3], 1):
            print(f"  {i}. {source.get('title', 'No title')}")
            print(f"     {source.get('url', '')}")
    
    print("\n")
    
    # Example 3: Explicit HuggingFace web search tool configuration
    print("=" * 60)
    print("Example 3: Custom Web Search Configuration")
    print("=" * 60)
    
    from llm_station.tools.web_search.huggingface import HuggingFaceWebSearch
    
    # Create custom web search tool
    custom_search = HuggingFaceWebSearch(
        use_web_search=True,
        extract_full_text=True,  # Extract full article text
        max_urls=2,  # Extract from top 2 URLs
        num_search_results=5,  # Get 5 search results
    )
    
    response = agent_with_search.generate(
        "Current trends in machine learning research",
        tools=[custom_search.spec()]  # Use custom tool spec
    )
    
    print("\nResponse:")
    print(response.content)
    print("\n")


if __name__ == "__main__":
    main()

