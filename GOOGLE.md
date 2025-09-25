# Google Gemini Provider Documentation

## Setup Instructions

### 1. Install & Configure
```bash
pip install -U google-genai python-dotenv
pip install -e .
echo "GEMINI_API_KEY=your-key" >> .env
```

### 2. Create Agent
```python
from llm_studio import Agent
import os

agent = Agent(
    provider="google",
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Image generation agent
image_agent = Agent(
    provider="google",
    model="gemini-2.5-flash-image-preview",
    api_key=os.getenv("GEMINI_API_KEY")
)
```

### 3. Make Tool Calls
```python
# Basic chat
response = agent.generate("What is AI?")

# With smart tools - simple, memorable names
response = agent.generate("Search AI news", tools=["search"])
response = agent.generate("Calculate with Python", tools=["code"])
response = agent.generate("Analyze URL content", tools=["url"])
response = agent.generate("Format as JSON", tools=["json"])
response = image_agent.generate("Generate image", tools=["image"])
```

## Supported Models & Tools

### Models
- `gemini-2.5-flash` - Fast, versatile, best price-performance
- `gemini-2.5-pro` - Maximum capability, complex reasoning
- `gemini-2.5-flash-image-preview` - Image generation and editing
- `gemini-2.0-flash` - Previous generation, fast responses
- `gemini-1.5-pro` - Legacy, large context window

### Available Tools
- `search` - Web search with automatic grounding (uses Google search)
- `code` - Code execution with data analysis (uses Google code execution)
- `url` - URL content processing (uses Google URL context)
- `image` - Image generation (uses Google native generation in Gemini 2.5+)
- `json` - JSON formatting (local tool)
- `fetch` - URL fetching (local tool)

### Tool Aliases (Alternative Names)
- `websearch`, `web_search` → `search`
- `python`, `execute`, `compute`, `run` → `code`
- `draw`, `create_image`, `generate_image` → `image`
- `webpage`, `url_context` → `url`
- `format_json`, `json_format` → `json`
- `download` → `fetch`

## Smart Tools System

### Overview
The smart tools system provides generic, memorable tool names that automatically route to the best available provider. When using a Google agent, smart tools will automatically use Google's implementations where available.

### Usage Examples
```python
# Smart tools automatically use Google implementations
response = agent.generate("Research quantum computing", tools=["search"])
# → Routes to Google search with automatic grounding

response = agent.generate("Create data visualization", tools=["code"])
# → Routes to Google code execution with matplotlib support

response = agent.generate("Process this URL content", tools=["url"])
# → Routes to Google URL context for content extraction

response = image_agent.generate("Create artwork", tools=["image"])
# → Routes to Google image generation (Gemini 2.5+)

# Combined workflow
response = agent.generate(
    "Research renewable energy, analyze trends, and create report",
    tools=["search", "code", "json"]
)
```

### Why Google Tools Excel
- **Search**: Gemini 2.0+ provides the most advanced search grounding
- **Code**: Best data analysis and visualization capabilities  
- **URL**: Advanced content processing and extraction
- **Image**: Native integration in Gemini 2.5+ models

### Cross-Provider Compatibility
```python
# Same tools work with any provider
openai_agent = Agent(provider="openai", model="gpt-4o-mini", api_key=openai_key)
google_agent = Agent(provider="google", model="gemini-2.5-flash", api_key=google_key)

# Identical interface, different implementations
openai_response = openai_agent.generate("Research AI", tools=["search"])
google_response = google_agent.generate("Research AI", tools=["search"])
```

## JSON Response Formats

### Basic Chat
```json
{
  "content": "Quantum computing uses quantum mechanics...",
  "tool_calls": [],
  "grounding_metadata": null
}
```

### Search Grounding (Gemini 2.0+)
```json
{
  "content": "Recent developments include...",
  "grounding_metadata": {
    "grounding": {
      "grounding_chunks": [
        {
          "web": {
            "uri": "https://source.com",
            "title": "Article Title",
            "snippet": "Content excerpt..."
          }
        }
      ],
      "web_search_queries": ["search query"],
      "search_entry_point": {"rendered_content": "<html>..."}
    },
    "sources": ["https://source1.com", "https://source2.com"],
    "citations": [
      {
        "url": "https://source.com",
        "title": "Title",
        "snippet": "Excerpt"
      }
    ],
    "search_entry_point": "<html>Google Search Suggestions</html>"
  }
}
```

### Code Execution
```json
{
  "content": "**Execution Output:**\n```\n120\n```\n**Generated Image** (image/png)",
  "grounding_metadata": {
    "code_execution": [
      {
        "code": "import math\nresult = math.factorial(5)",
        "language": "python",
        "result": {
          "output": "120",
          "outcome": "OUTCOME_OK"
        }
      }
    ],
    "inline_media": [
      {
        "mime_type": "image/png",
        "data": "base64_data",
        "size": 21071
      }
    ]
  }
}
```

### URL Context
```json
{
  "content": "Based on the website analysis...",
  "grounding_metadata": {
    "url_context": [
      {
        "url": "https://example.com",
        "status": "success",
        "content_type": "text/html"
      }
    ],
    "processed_urls": [
      {
        "url": "https://example.com",
        "status": "success",
        "content_type": "text/html",
        "size": 50000
      }
    ]
  }
}
```

### Image Generation (Gemini 2.5+)
```json
{
  "content": "I've created an image of a robot...",
  "grounding_metadata": {
    "image_generation": [
      {
        "type": "native_generation",
        "available": true,
        "format": "PIL_Image"
      }
    ]
  },
  "note": "Access images via response.raw.candidates[0].content.parts[].as_image()"
}
```

### Multi-Tool Response
```json
{
  "content": "Comprehensive analysis with research, code, and visualizations...",
  "grounding_metadata": {
    "grounding": {"grounding_chunks": [...], "web_search_queries": [...]},
    "sources": ["https://..."],
    "citations": [{"url": "...", "title": "..."}],
    "url_context": [{"url": "...", "status": "success"}],
    "processed_urls": [{"url": "...", "content_type": "text/html"}],
    "code_execution": [{"code": "...", "result": {"outcome": "OUTCOME_OK"}}],
    "inline_media": [{"mime_type": "image/png", "size": 34567}]
  }
}
```

## Batch Processing
```python
from llm_studio import GoogleBatchProcessor

processor = GoogleBatchProcessor(api_key=api_key)
tasks = [processor.create_task(key=f"task-{i}", model="gemini-2.5-flash", contents=text) for i, text in enumerate(texts)]
batch_job = processor.submit_batch(tasks)
results = processor.get_completed_results(batch_job.name)
```
