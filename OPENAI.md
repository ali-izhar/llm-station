# OpenAI Provider Documentation

## Setup Instructions

### 1. Install & Configure
```bash
pip install openai python-dotenv
pip install -e .
echo "OPENAI_API_KEY=your-key" >> .env
```

### 2. Create Agent
```python
from llm_station import Agent
import os

agent = Agent(
    provider="openai",
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### 3. Make Tool Calls
```python
# Basic chat
response = agent.generate("What is AI?")

# With smart tools - simple, memorable names
response = agent.generate("Search AI news", tools=["search"])
response = agent.generate("Calculate factorial", tools=["code"])
response = agent.generate("Generate image", tools=["image"])
response = agent.generate("Format as JSON", tools=["json"])
```

## Supported Models & Tools

### Models
- `gpt-4o-mini` - Chat Completions, function calling
- `gpt-4o` - Chat Completions + Responses API, all tools
- `gpt-4o-search-preview` - Built-in web search
- `gpt-5` - Responses API, reasoning capabilities

### Available Tools
- `search` - Web search with citations (uses OpenAI web search)
- `code` - Code execution in containers (uses OpenAI Code Interpreter) 
- `image` - Image generation and editing (uses OpenAI image generation)
- `json` - JSON formatting (local tool)
- `fetch` - URL fetching (local tool)

### Tool Aliases (Alternative Names)
- `websearch`, `web_search` → `search`
- `python`, `execute`, `compute`, `run` → `code`
- `draw`, `create_image`, `generate_image` → `image`
- `format_json`, `json_format` → `json`
- `download` → `fetch`

## Smart Tools System

### Overview
The smart tools system provides generic, memorable tool names that automatically route to the best available provider. When using an OpenAI agent, smart tools will automatically use OpenAI's implementations.

### Usage Examples
```python
# Smart tools automatically use OpenAI implementations
response = agent.generate("Research AI trends", tools=["search"])
# → Routes to OpenAI web search via Responses API

response = agent.generate("Calculate statistics", tools=["code"])  
# → Routes to OpenAI Code Interpreter via Responses API

response = agent.generate("Create artwork", tools=["image"])
# → Routes to OpenAI image generation via Responses API

# Multiple tools work together
response = agent.generate(
    "Research AI news, analyze the data, and create a summary",
    tools=["search", "code", "json"]
)
```

### Advanced Configuration
```python
# Provider preference (optional - already using OpenAI)
response = agent.generate(
    "Search for information", 
    tools=[{"name": "search", "provider_preference": "openai"}]
)

# Multiple smart tools
response = agent.generate(
    "Research, analyze, and visualize data",
    tools=["search", "code", "image", "json"]
)

# Tool discovery
from llm_station import get_available_tools, get_tool_info
tools = get_available_tools()
search_info = get_tool_info("search")
```

### Multiple Tools
```python
# Combine tools for complex workflows
response = agent.generate(
    "Research the latest AI developments, analyze trends, and create visualizations",
    tools=["search", "code", "image", "json"]
)
```

## JSON Response Formats

### Basic Chat (Chat Completions API)
```json
{
  "content": "AI is a branch of computer science...",
  "tool_calls": [],
  "grounding_metadata": null
}
```

### Web Search (Responses API)
```json
{
  "content": "Recent AI developments include...",
  "grounding_metadata": {
    "web_search": {
      "id": "ws_123",
      "status": "completed",
      "query": "AI news"
    },
    "citations": [
      {
        "url": "https://source.com",
        "title": "AI Breakthrough",
        "start_index": 100,
        "end_index": 200
      }
    ],
    "sources": ["https://source1.com", "https://source2.com"]
  }
}
```

### Code Interpreter (Responses API)
```json
{
  "content": "Calculation complete...",
  "grounding_metadata": {
    "code_interpreter": {
      "id": "ci_456",
      "container_id": "cntr_789",
      "code": "import math\nresult = math.factorial(10)",
      "output": "3628800"
    },
    "file_citations": [
      {
        "file_id": "cfile_123",
        "filename": "chart.png",
        "container_id": "cntr_789"
      }
    ]
  }
}
```

### Image Generation (Responses API)
```json
{
  "content": "I've created an image...",
  "grounding_metadata": {
    "image_generation": [
      {
        "id": "ig_345",
        "result": "base64_image_data",
        "revised_prompt": "optimized prompt",
        "size": "1024x1024",
        "quality": "high"
      }
    ]
  }
}
```

### Function Calling (Chat Completions API)
```json
{
  "content": "Here's the formatted JSON...",
  "tool_calls": [
    {
      "id": "call_678",
      "name": "json_format",
      "arguments": {
        "data": "name=Alice, age=30"
      }
    }
  ],
  "grounding_metadata": null
}
```

### Multi-Tool Response (Responses API)
```json
{
  "content": "Complete analysis with research, code, and images...",
  "grounding_metadata": {
    "web_search": {"id": "ws_123", "query": "research topic"},
    "code_interpreter": {"id": "ci_456", "output": "analysis results"},
    "image_generation": [{"id": "ig_789", "result": "base64_data"}],
    "citations": [{"url": "...", "title": "..."}],
    "sources": ["https://..."],
    "file_citations": [{"filename": "chart.png"}]
  }
}
```
