# LLM Studio

A unified, provider-agnostic agent framework for OpenAI, Google Gemini, and Anthropic Claude with tool integration.

## Quick Start

### Install
```bash
# Install from PyPI (recommended)
pip install llm-studio

# Optional: Install with specific provider support
pip install llm-studio[openai]     # OpenAI only
pip install llm-studio[anthropic]  # Anthropic only  
pip install llm-studio[google]     # Google only
pip install llm-studio[all]        # All providers

# Development install
git clone https://github.com/your-repo/llm_studio.git
cd llm_studio
pip install -e .[dev]
```

### Set Up API Keys
```bash
# Add to .env file
echo "OPENAI_API_KEY=your-openai-key" >> .env
echo "GEMINI_API_KEY=your-gemini-key" >> .env  
echo "ANTHROPIC_API_KEY=your-anthropic-key" >> .env
```

### Use Any Provider
```python
from llm_studio import Agent
from dotenv import load_dotenv
import os

load_dotenv()

# Same interface, any provider
agent = Agent(
    provider="openai", # or "google", "anthropic"
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Basic chat
response = agent.generate("What is machine learning?")
print(response.content)

# Use tools with simple, memorable names
response = agent.generate(
    "Search for recent AI developments",
    tools=["search"]  # Auto-routes to best search provider
)

print(response.content)
if response.grounding_metadata:
    sources = response.grounding_metadata.get("sources", [])
    print(f"Found {len(sources)} sources")

# Multiple tools work together
response = agent.generate(
    "Research AI trends, analyze the data, and create a summary",
    tools=["search", "code", "json"]
)
```

## Documentation

- **[OPENAI.md](OPENAI.md)** - OpenAI provider setup and tools
- **[GOOGLE.md](GOOGLE.md)** - Google Gemini provider setup and tools  
- **[CLAUDE.md](CLAUDE.md)** - Anthropic Claude provider setup and tools
