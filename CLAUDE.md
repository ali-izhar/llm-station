# Anthropic Claude Provider Documentation

## Setup Instructions

### 1. Install & Configure
```bash
pip install anthropic python-dotenv
pip install -e .
echo "ANTHROPIC_API_KEY=your-key" >> .env
```

### 2. Create Agent
```python
from llm_studio import Agent
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

### 3. Make Tool Calls
```python
# Basic chat
response = agent.generate("What is quantum computing?")

# With smart tools - simple, memorable names
response = agent.generate("Search for TypeScript 5.5 updates", tools=["search"])
response = agent.generate("Fetch content from https://docs.example.com", tools=["fetch"])
response = agent.generate("Calculate statistics with Python", tools=["code"])
response = agent.generate("Format results as JSON", tools=["json"])
```

## Supported Models & Tools

### Models
- `claude-opus-4-1-20250805` - Latest model, full capabilities
- `claude-opus-4-20250514` - Opus model with all tools
- `claude-sonnet-4-20250514` - Sonnet model, balanced performance
- `claude-3-7-sonnet-20250219` - Sonnet 3.7 with tool support
- `claude-3-5-haiku-latest` - Fast model with basic tools

### Available Tools
- `search` - Web search with citations (uses Anthropic web search)
- `code` - Code execution with bash and file operations (uses Anthropic execution)
- `fetch` - Web content fetching (local tool)
- `json` - JSON formatting (local tool)

### Tool Aliases (Alternative Names)
- `websearch`, `web_search` → `search`
- `python`, `execute`, `compute`, `run` → `code`
- `format_json`, `json_format` → `json`
- `download` → `fetch`

## Smart Tools System

### Overview
The smart tools system provides generic, memorable tool names that automatically route to the best available provider. When using a Claude agent, smart tools will automatically use Anthropic's implementations where available.

### Usage Examples
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

### Why Anthropic Tools Excel
- **Search**: Real-time web search with automatic citations and domain filtering
- **Code**: Bash + Python execution with file manipulation and container persistence
- **Fetch**: Advanced web content fetching with security controls (when available)
- **Token Management**: Built-in rate limiting and usage tracking

### Cross-Provider Compatibility
```python
# Same tools work with any provider
claude_agent = Agent(provider="anthropic", model="claude-sonnet-4", api_key=claude_key)
openai_agent = Agent(provider="openai", model="gpt-4o-mini", api_key=openai_key)

# Identical interface, different implementations
claude_response = claude_agent.generate("Research AI", tools=["search"])
openai_response = openai_agent.generate("Research AI", tools=["search"])
```

### Advanced Features
```python
# Provider preference (optional - already using Claude)
response = agent.generate(
    "Search for information",
    tools=[{"name": "search", "provider_preference": "anthropic"}]
)

# Tool discovery
from llm_studio import get_available_tools, get_tool_info
tools = get_available_tools()
search_info = get_tool_info("search")
```

## JSON Response Formats

### Basic Chat
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

### Web Search Response
```json
{
  "content": "I'll search for the latest TypeScript 5.5 information...\n\nBased on the search results, TypeScript 5.5 includes several important updates...",
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
      "https://devblogs.microsoft.com/typescript/announcing-typescript-5-5/",
      "https://github.com/microsoft/TypeScript/releases"
    ],
    "search_results": [
      {
        "url": "https://devblogs.microsoft.com/typescript/announcing-typescript-5-5/",
        "title": "Announcing TypeScript 5.5",
        "page_age": "May 15, 2024"
      }
    ],
    "citations": [
      {
        "url": "https://devblogs.microsoft.com/typescript/announcing-typescript-5-5/",
        "title": "Announcing TypeScript 5.5",
        "cited_text": "TypeScript 5.5 brings performance improvements and new language features...",
        "encrypted_index": "Eo8BCioIAhgBIiQyYjQ0OWJmZi1lNm.."
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

### Web Fetch Response
```json
{
  "content": "I've fetched the content from the documentation. Here's what I found...",
  "tool_calls": [],
  "grounding_metadata": {
    "web_fetch": [
      {
        "id": "srvtoolu_01ABC123def456ghi789",
        "name": "web_fetch",
        "status": "completed",
        "type": "server_tool"
      }
    ],
    "usage": {
      "input_tokens": 1250,
      "output_tokens": 800
    },
    "response_info": {
      "id": "msg_01FetchExample123",
      "model": "claude-sonnet-4-20250514",
      "stop_reason": "end_turn"
    }
  }
}
```

### Code Execution Response (Beta)
```json
{
  "content": "I'll calculate the statistics for you.\n\n**Execution Output:**\n```\nMean: 5.5\nStandard Deviation: 2.87\n```\n\nThe calculations show that for the dataset [1,2,3,4,5,6,7,8,9,10], the mean is 5.5 and the standard deviation is approximately 2.87.",
  "tool_calls": [],
  "grounding_metadata": {
    "code_execution": [
      {
        "id": "srvtoolu_01CodeExec789",
        "name": "bash_code_execution",
        "type": "server_tool",
        "status": "completed",
        "command": "python3 -c \"import statistics; data=[1,2,3,4,5,6,7,8,9,10]; print(f'Mean: {statistics.mean(data)}'); print(f'Std Dev: {statistics.stdev(data):.2f}')\"",
        "execution_type": "bash",
        "result": {
          "tool_use_id": "srvtoolu_01CodeExec789",
          "result_type": "bash_code_execution_result",
          "stdout": "Mean: 5.5\nStandard Deviation: 2.87",
          "stderr": "",
          "return_code": 0,
          "execution_type": "bash"
        }
      }
    ],
    "usage": {
      "input_tokens": 45,
      "output_tokens": 180
    },
    "response_info": {
      "id": "msg_01CodeExample456",
      "model": "claude-opus-4-1-20250805",
      "stop_reason": "end_turn",
      "container": {
        "id": "container_01ABC123",
        "type": "code_execution"
      }
    }
  }
}
```

### File Operations Response (Beta)
```json
{
  "content": "I've created the config.yaml file and updated the port as requested.\n\n**File Created:**\n```yaml\ndatabase:\n  host: localhost\n  port: 3306\n  name: myapp\n```\n\nThe configuration file has been successfully created and the port updated from 5432 to 3306.",
  "tool_calls": [],
  "grounding_metadata": {
    "code_execution": [
      {
        "id": "srvtoolu_01FileOp123",
        "name": "text_editor_code_execution",
        "type": "server_tool",
        "status": "completed",
        "command": "create",
        "path": "config.yaml",
        "execution_type": "file_operation",
        "result": {
          "tool_use_id": "srvtoolu_01FileOp123",
          "result_type": "text_editor_code_execution_result",
          "file_type": "text",
          "content": "database:\n  host: localhost\n  port: 3306\n  name: myapp",
          "num_lines": 4,
          "is_file_update": false,
          "execution_type": "file_operation"
        }
      }
    ],
    "response_info": {
      "container": {
        "id": "container_01FileExample",
        "type": "code_execution"
      }
    }
  }
}
```

### Function Calling Response (Local Tools)
```json
{
  "content": "Here's the data formatted as JSON:\n\n```json\n{\n  \"name\": \"Alice\",\n  \"age\": 30,\n  \"city\": \"New York\"\n}\n```",
  "tool_calls": [
    {
      "id": "call_anthropic_123",
      "name": "json_format",
      "arguments": {
        "data": "name=Alice, age=30, city=New York",
        "format": "structured"
      }
    }
  ],
  "grounding_metadata": {
    "usage": {
      "input_tokens": 25,
      "output_tokens": 85
    }
  }
}
```

### Multi-Tool Response
```json
{
  "content": "I'll research the topic, fetch specific documentation, and analyze the data...",
  "tool_calls": [],
  "grounding_metadata": {
    "web_search": [
      {
        "id": "srvtoolu_search456",
        "query": "renewable energy developments 2024",
        "status": "completed"
      }
    ],
    "web_fetch": [
      {
        "id": "srvtoolu_fetch789", 
        "status": "completed"
      }
    ],
    "code_execution": [
      {
        "id": "srvtoolu_code012",
        "execution_type": "bash",
        "result": {"return_code": 0, "stdout": "Analysis complete"}
      }
    ],
    "sources": ["https://energy.gov/...", "https://iea.org/..."],
    "citations": [
      {
        "url": "https://energy.gov/renewable-report",
        "title": "Renewable Energy Report 2024",
        "cited_text": "Global renewable capacity increased by 15% in 2024..."
      }
    ],
    "usage": {
      "input_tokens": 890,
      "output_tokens": 1240,
      "server_tool_use": {
        "web_search_requests": 2
      }
    }
  }
}
```

## Advanced Features

### Container Reuse
```python
# First request creates container
response1 = agent.generate(
    "Create a data analysis script and save it to analysis.py",
    tools=["anthropic_code_execution"]
)

# Extract container ID for reuse
container_id = response1.grounding_metadata["response_info"]["container"]["id"]

# Reuse container in subsequent requests
container_tool = AnthropicCodeExecution(container_id=container_id)
response2 = agent.generate(
    "Run the analysis.py script on new data",
    tools=[container_tool.spec()]
)
```

### Domain-Filtered Search
```python
from llm_studio.tools.web_search.anthropic import AnthropicWebSearch

# Academic research search
academic_search = AnthropicWebSearch(
    allowed_domains=["arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ieee.org"],
    max_uses=5
)

response = agent.generate(
    "Find recent research on quantum computing",
    tools=[academic_search.spec()]
)
```

### Secure Content Fetching  
```python
from llm_studio.tools.web_fetch.anthropic import AnthropicWebFetch

# Secure fetch with content limits
secure_fetch = AnthropicWebFetch(
    allowed_domains=["docs.python.org", "github.com"],
    max_content_tokens=5000,
    citations={"enabled": True}
)

response = agent.generate(
    "Analyze the Python documentation at https://docs.python.org/3/library/",
    tools=[secure_fetch.spec()]
)
```
## Usage Examples

### Research Workflow
```python
response = agent.generate("""
Research renewable energy developments:
1. Search for latest industry news
2. Fetch specific reports from energy.gov
3. Analyze data trends with Python
4. Create summary report

Include citations and data sources.
""", tools=["search", "fetch", "code"])

# Access comprehensive metadata
if response.grounding_metadata:
    searches = response.grounding_metadata.get("web_search", [])
    fetches = response.grounding_metadata.get("web_fetch", [])
    executions = response.grounding_metadata.get("code_execution", [])
    citations = response.grounding_metadata.get("citations", [])
    
    print(f"Performed {len(searches)} searches, {len(fetches)} fetches")
    print(f"Executed {len(executions)} code operations")
    print(f"Generated {len(citations)} citations")
```

### Data Analysis Pipeline
```python
# Upload data file first (requires Files API)
response = agent.generate("""
Analyze the uploaded CSV data:
1. Check file structure and preview data
2. Calculate basic statistics 
3. Create visualizations
4. Generate analysis report
5. Save results to new files

Provide comprehensive analysis with charts.
""", tools=["anthropic_code_execution"])

# Access execution results
if response.grounding_metadata:
    executions = response.grounding_metadata.get("code_execution", [])
    container = response.grounding_metadata["response_info"].get("container", {})
    
    print(f"Container ID: {container.get('id')}")
    for execution in executions:
        if execution.get("result", {}).get("return_code") == 0:
            print(f"✅ {execution['execution_type']} successful")
```

### Content Research & Analysis
```python
response = agent.generate("""
Research and analyze AI safety:
1. Search for recent AI safety research papers
2. Fetch content from key research institutions
3. Analyze trends and extract key findings
4. Create structured summary

Focus on 2024 developments with proper citations.
""", tools=["search", "fetch"])

# Rich metadata access
metadata = response.grounding_metadata
sources = metadata.get("sources", [])
citations = metadata.get("citations", []) 
usage = metadata.get("usage", {})

print(f"Research used {len(sources)} sources")
print(f"Generated {len(citations)} citations")
print(f"Cost: {usage.get('web_search_requests', 0)} searches")
```

## Batch API for Large-Scale Processing

Anthropic's Message Batches API provides high-throughput, cost-effective processing:

### Benefits
- **50% cost savings** compared to standard API prices
- **High throughput**: Up to 100,000 requests or 256MB per batch
- **24-hour completion window** with most batches finishing within 1 hour
- **All Messages API features**: Tools, vision, multi-turn conversations

### Basic Batch Processing

```python
from llm_studio import AnthropicBatchProcessor

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
results = processor.submit_and_wait(requests)
for result in results:
    if result.result_type.value == "succeeded":
        print(f"{result.custom_id}: {result.message}")
```

### Research Batch with Tools

```python
# Batch research with web search
batch_job = processor.process_research_batch(
    topics=[
        "Renewable energy developments 2024",
        "AI safety research progress", 
        "Quantum computing breakthroughs"
    ],
    model="claude-opus-4-1-20250805",
    max_tokens=3072
)

results = processor.download_results(
    processor.wait_for_completion(batch_job.id)
)
```

### Content Analysis Batch

```python
# Batch content analysis
batch_job = processor.process_content_analysis_batch(
    content_items=content_list,
    analysis_type="sentiment",
    model="claude-sonnet-4-20250514"
)
```

### Code Analysis Batch (Beta)

```python
# Batch code analysis with execution
from llm_studio.batch.anthropic_batch import create_code_analysis_batch

batch_job = create_code_analysis_batch(
    code_samples=["def fibonacci(n):", "class DataProcessor:", "async function api()"],
    processor=processor,
    model="claude-opus-4-1-20250805"
)
```

### Batch Job Management

```python
# List all batch jobs
jobs = processor.list_batch_jobs(limit=10)
for job in jobs:
    print(f"Job: {job.id} - Status: {job.processing_status.value}")
    print(f"  Requests: {job.request_counts}")

# Monitor specific job
batch_job = processor.get_batch_status("msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d")
print(f"Status: {batch_job.processing_status.value}")

# Cancel if needed
cancelled_job = processor.cancel_batch_job("msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d")
```
