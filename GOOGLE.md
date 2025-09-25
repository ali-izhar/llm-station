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

# With tools
response = agent.generate("Search AI news", tools=["google_search"])
response = agent.generate("Calculate with Python", tools=["google_code_execution"])
response = agent.generate("Analyze URL", tools=["google_url_context"])
response = image_agent.generate("Generate image", tools=["google_image_generation"])
```

## Supported Models & Tools

### Models
- `gemini-2.5-flash` - Fast, versatile, best price-performance
- `gemini-2.5-pro` - Maximum capability, complex reasoning
- `gemini-2.5-flash-image-preview` - Image generation and editing
- `gemini-2.0-flash` - Previous generation, fast responses
- `gemini-1.5-pro` - Legacy, large context window

### Tools
- `google_search` - Search with automatic grounding (Gemini 2.0+)
- `google_search_retrieval` - Legacy search with threshold (Gemini 1.5)
- `google_code_execution` - Python execution with visualization
- `google_url_context` - Direct URL content processing
- `google_image_generation` - Native image generation (2.5+ models)
- `json_format` - Local JSON formatting
- `fetch_url` - Local URL fetching

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
