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
from llm_studio import Agent
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

# With tools
response = agent.generate("Search AI news", tools=["openai_web_search"])
response = agent.generate("Calculate factorial", tools=["openai_code_interpreter"])
response = agent.generate("Generate image", tools=["openai_image_generation"])
```

## Supported Models & Tools

### Models
- `gpt-4o-mini` - Chat Completions, function calling
- `gpt-4o` - Chat Completions + Responses API, all tools
- `gpt-4o-search-preview` - Built-in web search
- `gpt-5` - Responses API, reasoning capabilities

### Tools
- `openai_web_search` - Web search with citations (Responses API)
- `openai_code_interpreter` - Python execution in containers (Responses API)
- `openai_image_generation` - Image generation/editing (Responses API)
- `json_format` - Local JSON formatting
- `fetch_url` - Local URL fetching

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
