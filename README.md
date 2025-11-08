# LLM Station

A unified, provider-agnostic agent framework for OpenAI, Google Gemini, Anthropic Claude, and HuggingFace with tool integration and batch processing.

## Quick Start

### Install
```bash
# Install from PyPI (recommended)
pip install llm-station

# Optional: Install with specific provider support
pip install llm-station[openai]     # OpenAI only
pip install llm-station[anthropic]  # Anthropic only  
pip install llm-station[google]     # Google only
pip install llm-station[all]        # All providers

# Development install
git clone https://github.com/ali-izhar/llm-station.git
cd llm-station
pip install -e .[dev]
```

### Set Up API Keys
```bash
# Add to .env file
echo "OPENAI_API_KEY=your-openai-key" >> .env
echo "GEMINI_API_KEY=your-gemini-key" >> .env  
echo "ANTHROPIC_API_KEY=your-anthropic-key" >> .env
```

### Basic Usage
```python
from llm_station import Agent
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

## Testing Your Installation

### Quick Test
```python
from llm_station import Agent

# Test with mock provider (no API key needed)
agent = Agent(provider="mock", model="test")
response = agent.generate("Hello, world!")
print(response.content)  # Should work without API calls
```

### Run Test Suite
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_integration.py -v

# Run without integration tests (no API calls)
pytest tests/ -m "not integration"
```

## Available Tools

### Smart Tools (Auto-Routed)
- `search` - Web search with citations (routes to provider's search tool)
- `code` - Code execution (routes to provider's code execution tool)
- `image` - Image generation (routes to provider's image tool)
- `url` - URL content processing (Google only)
- `json` - JSON formatting (local tool)
- `fetch` - URL fetching (local tool)

### Tool Aliases
- `websearch`, `web_search` → `search`
- `python`, `execute`, `compute`, `run` → `code`
- `draw`, `create_image`, `generate_image` → `image`
- `format_json`, `json_format` → `json`
- `download` → `fetch`

## Advanced Features

### Structured Output
```python
# Get JSON responses with schema validation
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "score": {"type": "number"}
    },
    "required": ["summary", "sentiment"]
}

response = agent.generate(
    "Analyze this text: 'I love this product!'",
    structured_schema=schema
)
# response.content will be valid JSON matching the schema
```

### Batch Processing
```python
from llm_station.batch import OpenAIBatchProcessor
from llm_station import UserMessage

processor = OpenAIBatchProcessor(api_key=api_key)

# Create batch tasks
tasks = [
    processor.create_task(
        custom_id=f"task-{i}",
        model="gpt-4o-mini",
        messages=[UserMessage(f"Analyze: {text}")]
    )
    for i, text in enumerate(texts)
]

# Submit and wait for results
batch_job = processor.submit_batch(tasks)
results = processor.get_completed_results(batch_job.id, wait=True)

for result in results:
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Success: {result.response}")
```

See [PROVIDER.md](PROVIDER.md) for detailed batch processing examples for each provider.

### Logging
```python
from llm_station.logging import setup_logging, LogLevel, LogFormat

# Setup logging
logger = setup_logging(
    level=LogLevel.DEBUG,
    format=LogFormat.JSON,  # or CONSOLE, MARKDOWN
    enabled=True
)

# Use agent - all interactions are logged
agent = Agent(provider="openai", model="gpt-4o-mini", api_key=api_key)
response = agent.generate("Test", tools=["search"])

# Get session log
session_log = logger.get_session_log()
print(session_log)  # JSON formatted log
```

### CLI Logging
```bash
# Use logging from command line
python -m llm_station.cli.logging_cli --log --log-level debug --log-format json
```

## Examples

### Multi-Tool Workflow
```python
agent = Agent(provider="google", model="gemini-2.5-flash", api_key=api_key)

response = agent.generate(
    "Research renewable energy trends, create a data visualization, and format as JSON",
    tools=["search", "code", "json"]
)

print(response.content)
if response.grounding_metadata:
    # Access search sources
    sources = response.grounding_metadata.get("sources", [])
    # Access code execution results
    code_results = response.grounding_metadata.get("code_execution", [])
```

### Provider Switching
```python
# Same code works with any provider
providers = ["openai", "google", "anthropic"]
models = ["gpt-4o-mini", "gemini-2.5-flash", "claude-sonnet-4-20250514"]

for provider, model in zip(providers, models):
    agent = Agent(provider=provider, model=model, api_key=api_keys[provider])
    response = agent.generate("What is AI?", tools=["search"])
    print(f"{provider}: {response.content[:100]}...")
```

## Documentation

- **[PROVIDER.md](PROVIDER.md)** - Complete provider documentation with examples for OpenAI, Google Gemini, and Anthropic Claude
  - Setup instructions for each provider
  - Supported models and tools
  - JSON response formats
  - Batch processing examples
  - Advanced features

## Troubleshooting

### Common Issues

**ImportError: No module named 'openai'**
```bash
pip install llm-station[openai]  # Install provider dependencies
```

**API Key Not Found**
```python
# Make sure .env file exists and is loaded
from dotenv import load_dotenv
load_dotenv()  # Loads .env file
```

**Tool Not Found**
```python
# Check available tools
from llm_station.tools import list_tools
print(list_tools())  # Shows all registered tools
```

**Rate Limit Errors**
- Anthropic provider includes automatic rate limiting
- For other providers, implement your own retry logic
- Use batch APIs for high-volume processing

## Contributing

Contributions welcome! Please see our contributing guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
