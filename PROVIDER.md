# Provider Documentation

Complete documentation for all supported LLM providers in LLM Station.

## Table of Contents

- [OpenAI](#openai-provider)
- [Google Gemini](#google-gemini-provider)
- [Anthropic Claude](#anthropic-claude-provider)

---

## OpenAI Provider

### Setup Instructions

#### 1. Install & Configure
```bash
pip install openai python-dotenv
pip install -e .
echo "OPENAI_API_KEY=your-key" >> .env
```

#### 2. Create Agent
```python
from llm_station import Agent
import os

agent = Agent(
    provider="openai",
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

#### 3. Make Tool Calls
```python
# Basic chat
response = agent.generate("What is AI?")

# With smart tools - simple, memorable names
response = agent.generate("Search AI news", tools=["search"])
response = agent.generate("Calculate factorial", tools=["code"])
response = agent.generate("Generate image", tools=["image"])
response = agent.generate("Format as JSON", tools=["json"])
```

### Supported Models & Tools

#### Models
- `gpt-4o-mini` - Chat Completions, function calling
- `gpt-4o` - Chat Completions + Responses API, all tools
- `gpt-4o-search-preview` - Built-in web search
- `gpt-5` - Responses API, reasoning capabilities

#### Available Tools
- `search` - Web search with citations (uses OpenAI web search)
- `code` - Code execution in containers (uses OpenAI Code Interpreter) 
- `image` - Image generation and editing (uses OpenAI image generation)
- `json` - JSON formatting (local tool)
- `fetch` - URL fetching (local tool)

#### Tool Aliases (Alternative Names)
- `websearch`, `web_search` → `search`
- `python`, `execute`, `compute`, `run` → `code`
- `draw`, `create_image`, `generate_image` → `image`
- `format_json`, `json_format` → `json`
- `download` → `fetch`

### Smart Tools System

The smart tools system provides generic, memorable tool names that automatically route to the best available provider. When using an OpenAI agent, smart tools will automatically use OpenAI's implementations.

#### Usage Examples
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

### JSON Response Formats

#### Basic Chat (Chat Completions API)
```json
{
  "content": "AI is a branch of computer science...",
  "tool_calls": [],
  "grounding_metadata": null
}
```

#### Web Search (Responses API)
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

#### Code Interpreter (Responses API)
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

#### Image Generation (Responses API)
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

---

## Google Gemini Provider

### Setup Instructions

#### 1. Install & Configure
```bash
pip install -U google-genai python-dotenv
pip install -e .
echo "GEMINI_API_KEY=your-key" >> .env
```

#### 2. Create Agent
```python
from llm_station import Agent
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

#### 3. Make Tool Calls
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

### Supported Models & Tools

#### Models
- `gemini-2.5-flash` - Fast, versatile, best price-performance
- `gemini-2.5-pro` - Maximum capability, complex reasoning
- `gemini-2.5-flash-image-preview` - Image generation and editing
- `gemini-2.0-flash` - Previous generation, fast responses
- `gemini-1.5-pro` - Legacy, large context window

#### Available Tools
- `search` - Web search with automatic grounding (uses Google search)
- `code` - Code execution with data analysis (uses Google code execution)
- `url` - URL content processing (uses Google URL context)
- `image` - Image generation (uses Google native generation in Gemini 2.5+)
- `json` - JSON formatting (local tool)
- `fetch` - URL fetching (local tool)

#### Tool Aliases (Alternative Names)
- `websearch`, `web_search` → `search`
- `python`, `execute`, `compute`, `run` → `code`
- `draw`, `create_image`, `generate_image` → `image`
- `webpage`, `url_context` → `url`
- `format_json`, `json_format` → `json`
- `download` → `fetch`

### Smart Tools System

The smart tools system provides generic, memorable tool names that automatically route to the best available provider. When using a Google agent, smart tools will automatically use Google's implementations where available.

#### Usage Examples
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

#### Why Google Tools Excel
- **Search**: Gemini 2.0+ provides the most advanced search grounding
- **Code**: Best data analysis and visualization capabilities  
- **URL**: Advanced content processing and extraction
- **Image**: Native integration in Gemini 2.5+ models

### JSON Response Formats

#### Basic Chat
```json
{
  "content": "Quantum computing uses quantum mechanics...",
  "tool_calls": [],
  "grounding_metadata": null
}
```

#### Search Grounding (Gemini 2.0+)
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

#### Code Execution
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

#### Image Generation (Gemini 2.5+)
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

### Batch Processing
```python
from llm_station.batch import GoogleBatchProcessor

processor = GoogleBatchProcessor(api_key=api_key)

# Create batch tasks
tasks = [
    processor.create_task(
        key=f"task-{i}",
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": text}]}]
    )
    for i, text in enumerate(texts)
]

# Submit batch (creates file, uploads, and creates job)
batch_job = processor.submit_batch(tasks)

# Wait for completion and get results
completed_job = processor.wait_for_completion(batch_job.name)
results = processor.download_results(completed_job)
```

---

## Anthropic Claude Provider

### Setup Instructions

#### 1. Install & Configure
```bash
pip install anthropic python-dotenv
pip install -e .
echo "ANTHROPIC_API_KEY=your-key" >> .env
```

#### 2. Create Agent
```python
from llm_station import Agent
import os

agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Advanced configuration
advanced_agent = Agent(
    provider="anthropic",
    model="claude-opus-4-1-20250805",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
    max_tokens=4096,
    system_prompt="You are an expert research assistant."
)
```

#### 3. Make Tool Calls
```python
# Basic chat
response = agent.generate("What is quantum computing?")

# With smart tools - simple, memorable names
response = agent.generate("Search for TypeScript 5.5 updates", tools=["search"])
response = agent.generate("Fetch content from https://docs.example.com", tools=["fetch"])
response = agent.generate("Calculate statistics with Python", tools=["code"])
response = agent.generate("Format results as JSON", tools=["json"])
```

### Supported Models & Tools

#### Models
- `claude-opus-4-1-20250805` - Latest model, full capabilities
- `claude-opus-4-20250514` - Opus model with all tools
- `claude-sonnet-4-20250514` - Sonnet model, balanced performance
- `claude-3-7-sonnet-20250219` - Sonnet 3.7 with tool support
- `claude-3-5-haiku-latest` - Fast model with basic tools

#### Available Tools
- `search` - Web search with citations (uses Anthropic web search)
- `code` - Code execution with bash and file operations (uses Anthropic execution)
- `fetch` - Web content fetching (local tool)
- `json` - JSON formatting (local tool)

#### Tool Aliases (Alternative Names)
- `websearch`, `web_search` → `search`
- `python`, `execute`, `compute`, `run` → `code`
- `format_json`, `json_format` → `json`
- `download` → `fetch`

### Smart Tools System

The smart tools system provides generic, memorable tool names that automatically route to the best available provider. When using a Claude agent, smart tools will automatically use Anthropic's implementations where available.

#### Usage Examples
```python
# Smart tools automatically use Anthropic implementations
response = agent.generate("Research AI safety", tools=["search"])
# → Routes to Anthropic web search with citations

response = agent.generate("Analyze data trends", tools=["code"])
# → Routes to Anthropic code execution (if beta access available)

response = agent.generate("Format results", tools=["json"])
# → Uses local json_format tool

# Combined research workflow
response = agent.generate(
    "Research renewable energy, analyze trends, and create report",
    tools=["search", "code", "json"]
)
```

#### Why Anthropic Tools Excel
- **Search**: Real-time web search with automatic citations and domain filtering
- **Code**: Bash + Python execution with file manipulation and container persistence
- **Fetch**: Advanced web content fetching with security controls (when available)
- **Token Management**: Built-in rate limiting and usage tracking

### JSON Response Formats

#### Basic Chat
```json
{
  "content": "Quantum computing is a revolutionary computing paradigm...",
  "tool_calls": [],
  "grounding_metadata": {
    "usage": {
      "input_tokens": 21,
      "output_tokens": 305
    },
    "response_info": {
      "id": "msg_01HCDu5LRGeP2o7s2xGmxyx8",
      "model": "claude-sonnet-4-20250514",
      "stop_reason": "end_turn"
    }
  }
}
```

#### Web Search Response
```json
{
  "content": "I'll search for the latest TypeScript 5.5 information...",
  "tool_calls": [],
  "grounding_metadata": {
    "web_search": [
      {
        "id": "srvtoolu_01WYG3ziw53XMcoyKL4XcZmE",
        "name": "web_search",
        "query": "TypeScript 5.5 updates features",
        "status": "completed",
        "type": "server_tool"
      }
    ],
    "sources": [
      "https://devblogs.microsoft.com/typescript/announcing-typescript-5-5/"
    ],
    "citations": [
      {
        "url": "https://devblogs.microsoft.com/typescript/announcing-typescript-5-5/",
        "title": "Announcing TypeScript 5.5",
        "cited_text": "TypeScript 5.5 brings performance improvements..."
      }
    ],
    "usage": {
      "input_tokens": 105,
      "output_tokens": 512,
      "server_tool_use": {
        "web_search_requests": 1
      }
    }
  }
}
```

#### Code Execution Response (Beta)
```json
{
  "content": "I'll calculate the statistics for you...",
  "tool_calls": [],
  "grounding_metadata": {
    "code_execution": [
      {
        "id": "srvtoolu_01CodeExec789",
        "name": "bash_code_execution",
        "type": "server_tool",
        "status": "completed",
        "command": "python3 -c \"import statistics; data=[1,2,3,4,5,6,7,8,9,10]; print(f'Mean: {statistics.mean(data)}')\"",
        "execution_type": "bash",
        "result": {
          "tool_use_id": "srvtoolu_01CodeExec789",
          "result_type": "bash_code_execution_result",
          "stdout": "Mean: 5.5",
          "stderr": "",
          "return_code": 0,
          "execution_type": "bash"
        }
      }
    ],
    "usage": {
      "input_tokens": 45,
      "output_tokens": 180
    }
  }
}
```

### Advanced Features

#### Container Reuse
```python
# First request creates container
response1 = agent.generate(
    "Create a data analysis script and save it to analysis.py",
    tools=["code"]
)

# Extract container ID for reuse
container_id = response1.grounding_metadata["response_info"]["container"]["id"]

# Reuse container in subsequent requests
from llm_station.tools.code import AnthropicCode
container_tool = AnthropicCode(container_id=container_id)
response2 = agent.generate(
    "Run the analysis.py script on new data",
    tools=[container_tool.spec()]
)
```

#### Domain-Filtered Search
```python
from llm_station.tools.search import AnthropicSearch

# Academic research search
academic_search = AnthropicSearch(
    allowed_domains=["arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ieee.org"],
    max_uses=5
)

response = agent.generate(
    "Find recent research on quantum computing",
    tools=[academic_search.spec()]
)
```

### Batch API for Large-Scale Processing

Anthropic's Message Batches API provides high-throughput, cost-effective processing:

#### Benefits
- **50% cost savings** compared to standard API prices
- **High throughput**: Up to 100,000 requests or 256MB per batch
- **24-hour completion window** with most batches finishing within 1 hour
- **All Messages API features**: Tools, vision, multi-turn conversations

#### Basic Batch Processing
```python
from llm_station.batch import AnthropicBatchProcessor
from llm_station import UserMessage

processor = AnthropicBatchProcessor(api_key=claude_key)

# Create batch requests
requests = []
for i, topic in enumerate(research_topics):
    request = processor.create_request(
        custom_id=f"research-{i}",
        model="claude-sonnet-4-20250514",
        messages=[UserMessage(f"Research: {topic}")],
        system="You are a research analyst. Provide comprehensive analysis.",
        max_tokens=2048
    )
    requests.append(request)

# Submit batch
batch_job = processor.create_batch_job(requests)
print(f"Batch submitted: {batch_job.id}")

# Wait for completion and get results
results = processor.download_results(
    processor.wait_for_completion(batch_job.id)
)
for result in results:
    if result.result_type.value == "succeeded":
        print(f"{result.custom_id}: {result.message}")
```

---

## Cross-Provider Compatibility

All providers support the same smart tool interface, making it easy to switch between providers:

```python
# Same tools work with any provider
openai_agent = Agent(provider="openai", model="gpt-4o-mini", api_key=openai_key)
google_agent = Agent(provider="google", model="gemini-2.5-flash", api_key=google_key)
claude_agent = Agent(provider="anthropic", model="claude-sonnet-4", api_key=claude_key)

# Identical interface, different implementations
openai_response = openai_agent.generate("Research AI", tools=["search"])
google_response = google_agent.generate("Research AI", tools=["search"])
claude_response = claude_agent.generate("Research AI", tools=["search"])
```

