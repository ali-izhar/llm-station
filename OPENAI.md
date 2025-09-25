# OpenAI Provider Documentation

Complete guide to using OpenAI models and tools with LLM Studio.

## 🚀 Quick Start

### Setup
```bash
# Install OpenAI SDK
pip install openai

# Set API key in .env
echo "OPENAI_API_KEY=your-openai-key" >> .env
```

### Basic Usage
```python
from llm_studio import Agent
from dotenv import load_dotenv
import os

load_dotenv()

# Create OpenAI agent
agent = Agent(
    provider="openai",
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    system_prompt="You are a helpful assistant."
)

# Simple conversation
response = agent.generate("What is machine learning?")
print(response.content)
```

## 🤖 OpenAI Models

### Available Models
- **GPT-4o**: Latest multimodal model
- **GPT-4o-mini**: Cost-effective variant
- **GPT-5**: Advanced reasoning model
- **o3/o4-mini**: Deep research models
- **gpt-4o-search-preview**: Built-in web search
- **gpt-4.1**: Advanced model series

### Model Selection
```python
# Standard models
agent = Agent(provider="openai", model="gpt-4o-mini", api_key=key)

# Search models (built-in web search)
search_agent = Agent(provider="openai", model="gpt-4o-search-preview", api_key=key)

# Reasoning models
reasoning_agent = Agent(provider="openai", model="gpt-5", api_key=key)
```

## 🛠️ OpenAI Tools

### Available Tools
- **`openai_web_search`** - Web search with domain filtering and citations
- **`openai_web_search_preview`** - Preview version of web search
- **`openai_code_interpreter`** - Python code execution in sandboxed containers
- **`openai_image_generation`** - AI image generation and editing

### Quick Tool Usage
```python
# Web search
response = agent.generate("Search for AI news", tools=["openai_web_search"])

# Code execution
response = agent.generate("Calculate factorial of 10", tools=["openai_code_interpreter"])

# Image generation
response = agent.generate("Draw a sunset landscape", tools=["openai_image_generation"])

# Multiple tools
response = agent.generate(
    "Research AI trends, analyze with Python, create visualization",
    tools=["openai_web_search", "openai_code_interpreter", "openai_image_generation"]
)
```

## 🔍 Web Search Tool

### Basic Web Search
```python
from llm_studio.tools.web_search.openai import OpenAIWebSearch

# Basic search
tool = OpenAIWebSearch()
response = agent.generate("Search query", tools=[tool.spec()])
```

### Domain-Filtered Search
```python
# Medical research domains (up to 20 domains)
search_tool = OpenAIWebSearch(
    allowed_domains=[
        "pubmed.ncbi.nlm.nih.gov",
        "clinicaltrials.gov",
        "www.who.int",
        "www.cdc.gov",
        "www.fda.gov"
    ]
)

response = agent.generate(
    "Find research on diabetes treatment",
    tools=[search_tool.spec()]
)

# Access citations and sources
if response.grounding_metadata:
    citations = response.grounding_metadata.get("citations", [])
    sources = response.grounding_metadata.get("sources", [])
    for citation in citations:
        print(f"Source: {citation['url']} - {citation['title']}")
```

### Geographic Search
```python
# Location-based search refinement
local_search = OpenAIWebSearch(
    user_location={
        "country": "US",          # Two-letter ISO code
        "city": "San Francisco",  # Free text
        "region": "California",   # Free text
        "timezone": "America/Los_Angeles"  # IANA timezone
    }
)

response = agent.generate(
    "Find local restaurants near me",
    tools=[local_search.spec()]
)
```

### Search Types
1. **Non-reasoning search**: Fast lookups, direct query passing
2. **Agentic search**: Reasoning models manage search process
3. **Deep research**: Extended investigations (hundreds of sources, several minutes)

## 🐍 Code Interpreter Tool

### Basic Code Execution
```python
from llm_studio.tools.code_execution.openai import OpenAICodeInterpreter

# Auto container (recommended)
tool = OpenAICodeInterpreter()
response = agent.generate(
    "Solve the equation 3x + 11 = 14 using Python",
    tools=[tool.spec()]
)
```

### Container Management
```python
# Auto container with files
tool = OpenAICodeInterpreter(
    container_type="auto",
    file_ids=["file-abc123", "file-def456"]
)

# Explicit container
tool = OpenAICodeInterpreter(container_type="cntr_abc123")

response = agent.generate(
    "Analyze the uploaded data and create visualizations",
    tools=[tool.spec()]
)

# Access container info and generated files
if response.grounding_metadata:
    code_info = response.grounding_metadata.get("code_interpreter", {})
    file_citations = response.grounding_metadata.get("file_citations", [])
    
    print(f"Container: {code_info.get('container_id')}")
    for file_info in file_citations:
        print(f"Generated file: {file_info['filename']}")
```

### Use Cases
- **Data analysis**: Process CSV, JSON, Excel files
- **Visualization**: Create charts, graphs, plots
- **Math calculations**: Solve equations, statistics
- **Image processing**: Crop, resize, transform images (o3/o4 models)
- **File generation**: Create reports, data files

## 🎨 Image Generation Tool

### Basic Image Generation
```python
from llm_studio.tools.image_generation.openai import OpenAIImageGeneration

# Basic generation
tool = OpenAIImageGeneration()
response = agent.generate(
    "Generate an image of a red apple on white background",
    tools=[tool.spec()]
)

# Access generated images
if response.grounding_metadata:
    image_calls = response.grounding_metadata["image_generation"]
    for call in image_calls:
        image_base64 = call["result"]          # Base64-encoded image
        revised_prompt = call["revised_prompt"] # Auto-optimized prompt
        
        # Save image
        import base64
        image_data = base64.b64decode(image_base64)
        with open("generated_image.png", "wb") as f:
            f.write(image_data)
```

### Advanced Configuration
```python
# High-quality with specific options
image_tool = OpenAIImageGeneration(
    size="1024x1536",        # Portrait format
    quality="high",          # Maximum detail
    format="png",           # Lossless format
    background="transparent" # For logos
)

# Compressed for web
web_tool = OpenAIImageGeneration(
    format="jpeg",
    compression=85,          # 0-100 compression level
    quality="medium"
)

# Streaming progress
streaming_tool = OpenAIImageGeneration(
    partial_images=2,        # 1-3 progressive renders
    quality="high"
)
```

### Multi-turn Image Editing
```python
# Initial generation
response1 = agent.generate("Draw a cat", tools=["openai_image_generation"])

# Edit using previous response context
from llm_studio.tools.image_generation.openai import OpenAIImageGenerationAdvanced
edit_tool = OpenAIImageGenerationAdvanced(
    previous_response_id=response1.id,  # Reference previous response
    quality="high"
)

response2 = agent.generate("Make it more realistic", tools=[edit_tool.spec()])
```

## 🔄 Batch API

### Async Batch Processing (Lower Costs)
```python
from llm_studio import OpenAIBatchProcessor, BatchTask, SystemMessage, UserMessage

# Create batch processor
processor = OpenAIBatchProcessor(api_key=openai_key)

# Create batch tasks
tasks = []
for i, description in enumerate(movie_descriptions):
    task = processor.create_task(
        custom_id=f"movie-{i}",
        model="gpt-4o-mini",
        messages=[
            SystemMessage("Categorize this movie and provide a summary"),
            UserMessage(description)
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    tasks.append(task)

# Submit batch job
batch_job = processor.submit_batch(tasks, metadata={"project": "movie_analysis"})
print(f"Batch job submitted: {batch_job.id}")

# Wait for completion (up to 24h, usually faster)
completed_job = processor.wait_for_completion(batch_job.id)

# Download results
results = processor.download_results(completed_job)
for result in results:
    print(f"{result.custom_id}: {result.response}")
```

### Batch CLI Tool
```bash
# Create sample input file
python examples/batch_cli.py sample

# Create batch job from text file
python examples/batch_cli.py create sample_movies.txt --model gpt-4o-mini

# Check batch status
python examples/batch_cli.py status batch_abc123

# Download results when completed
python examples/batch_cli.py results batch_abc123 --output results.jsonl

# List recent batch jobs
python examples/batch_cli.py list
```

### Vision Batch Processing
```python
# Image captioning batch
file_path = processor.process_image_batch(
    image_urls=["https://img1.jpg", "https://img2.jpg"],
    texts=["Furniture item 1", "Furniture item 2"], 
    system_prompt="Generate short captions for furniture images",
    model="gpt-4o-mini",
    max_tokens=300
)

# Submit and process
file_id = processor.upload_batch_file(file_path)
batch_job = processor.create_batch_job(file_id)
```

## 🌐 API Selection (Automatic)

### Chat Completions vs Responses API
```python
# Chat Completions API (default)
agent = Agent(provider="openai", model="gpt-4o-mini", api_key=key)
response = agent.generate("Hello")  # Uses Chat Completions

# Responses API (automatic with tools)
response = agent.generate("Search web", tools=["openai_web_search"])        # → Responses API
response = agent.generate("Run Python", tools=["openai_code_interpreter"])  # → Responses API  
response = agent.generate("Generate image", tools=["openai_image_generation"]) # → Responses API

# Built-in search models (Chat Completions with native search)
search_agent = Agent(provider="openai", model="gpt-4o-search-preview", api_key=key)
response = search_agent.generate("Search web")  # → Chat Completions with built-in search
```

### When Each API is Used
- **Chat Completions**: Default, local tools, built-in search models
- **Responses API**: Required for web search, code interpreter, image generation tools

## 🔍 Professional Logging

### Enable Logging with Clean CLI
```bash
# Basic logging (info level, auto-saves to logs/)
python examples/agent_with_logging.py -l "Search for AI news"

# Warning-level logging (errors + warnings)
python examples/agent_with_logging.py -l --log-level warn "Research task"

# Debug logging with full API details
python examples/agent_with_logging.py -l --log-level debug "Complex workflow"

# Save to custom log file
python examples/agent_with_logging.py -lf my_session.log "Custom logging"

# JSON logs (auto-saved to logs/YYYYMMDD_HHMMSS_openai_model.log)
python examples/agent_with_logging.py -l --log-format json "Data analysis"
```

### Logging in Code
```python
from llm_studio import Agent, setup_logging, LogLevel

# Enable logging with professional levels
setup_logging(level=LogLevel.INFO)        # General information (default)
setup_logging(level=LogLevel.DEBUG)       # Detailed debugging information

# Use agent - automatic logging + timestamped file in logs/
agent = Agent(provider="openai", model="gpt-4o", api_key=key)
result = agent.generate("Complex task", tools=["openai_web_search", "openai_code_interpreter"])

# Logs show: tool selection, API calls, provider tool execution, metadata, timing
```

### Log Levels
- **error**: Only errors and critical issues
- **warn**: Warnings and errors
- **info**: General information (default)
- **debug**: Detailed debugging information

## 🎯 Complete Examples

### Research Assistant with All Tools
```python
# OpenAI agent with comprehensive tools
agent = Agent(provider="openai", model="gpt-4o", api_key=openai_key)

# Advanced web search with domain filtering
from llm_studio.tools.web_search.openai import OpenAIWebSearch
search_tool = OpenAIWebSearch(
    allowed_domains=["arxiv.org", "nature.com"],
    user_location={"country": "US", "city": "Boston"}
)

response = agent.generate(
    "Search for recent AI research, analyze the trends with Python, and create a visualization",
    tools=[search_tool.spec(), "openai_code_interpreter", "openai_image_generation"]
)

# Access rich metadata
if response.grounding_metadata:
    # Web search results
    if "web_search" in response.grounding_metadata:
        web_info = response.grounding_metadata["web_search"]
        print(f"Search query: {web_info['query']}")
    
    # Citations from search
    if "citations" in response.grounding_metadata:
        citations = response.grounding_metadata["citations"]
        print(f"Found {len(citations)} citations")
    
    # Generated files from code execution
    if "file_citations" in response.grounding_metadata:
        files = response.grounding_metadata["file_citations"]
        for file_info in files:
            print(f"Generated: {file_info['filename']}")
    
    # Generated images
    if "image_generation" in response.grounding_metadata:
        images = response.grounding_metadata["image_generation"]
        for img in images:
            print(f"Image: {len(img['result'])} chars (base64)")
```

### High-Quality Image Generation
```python
from llm_studio.tools.image_generation.openai import OpenAIImageGeneration

# Professional logo generation
logo_tool = OpenAIImageGeneration(
    size="1792x1024",       # Wide landscape
    quality="high",         # Maximum detail
    format="png",          # Lossless
    background="transparent" # For logos
)

response = agent.generate(
    "Create a professional AI company logo with transparent background",
    tools=[logo_tool.spec()]
)

# Save generated images
if response.grounding_metadata and "image_generation" in response.grounding_metadata:
    import base64
    for i, call in enumerate(response.grounding_metadata["image_generation"]):
        image_data = base64.b64decode(call["result"])
        with open(f"logo_{i}.png", "wb") as f:
            f.write(image_data)
        print(f"✅ Saved logo_{i}.png")
        print(f"Revised prompt: {call['revised_prompt']}")
```

### Domain-Specific Research
```python
# Medical research with domain filtering
medical_search = OpenAIWebSearch(
    allowed_domains=[
        "pubmed.ncbi.nlm.nih.gov",
        "www.nejm.org",
        "www.thelancet.com"
    ]
)

response = agent.generate(
    "Find recent research on quantum computing applications in drug discovery",
    tools=[medical_search.spec()]
)

# Verify citations are from allowed domains
if response.grounding_metadata and "citations" in response.grounding_metadata:
    for citation in response.grounding_metadata["citations"]:
        print(f"✓ Citation: {citation['url']}")
```

### Code Execution with Data Analysis
```python
from llm_studio.tools.code_execution.openai import OpenAICodeInterpreter

# Data analysis with container persistence
analysis_tool = OpenAICodeInterpreter(
    container_type="auto",
    file_ids=["file-dataset123"]  # Pre-uploaded data files
)

response = agent.generate(
    "Analyze the uploaded sales data, calculate trends, and create visualizations",
    tools=[analysis_tool.spec()]
)

# Access analysis results
if response.grounding_metadata:
    if "code_interpreter" in response.grounding_metadata:
        code_info = response.grounding_metadata["code_interpreter"]
        print(f"Container: {code_info.get('container_id')}")
        print(f"Code executed: {code_info.get('code', '')[:100]}...")
    
    if "file_citations" in response.grounding_metadata:
        files = response.grounding_metadata["file_citations"]
        print(f"Generated {len(files)} files:")
        for file_info in files:
            print(f"  - {file_info['filename']} (ID: {file_info['file_id']})")
```

## 🔄 Batch API for Large-Scale Processing

### Movie Categorization Batch
```python
from llm_studio import OpenAIBatchProcessor, SystemMessage, UserMessage

processor = OpenAIBatchProcessor(api_key=openai_key)

# Create batch tasks for movie categorization
tasks = []
system_prompt = '''
Extract movie categories and provide a 1-sentence summary.
Output JSON: {"categories": ["genre1", "genre2"], "summary": "..."}
'''

for i, description in enumerate(movie_descriptions):
    task = processor.create_task(
        custom_id=f"movie-{i}",
        model="gpt-4o-mini",
        messages=[
            SystemMessage(system_prompt),
            UserMessage(description)
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    tasks.append(task)

# Submit batch (24h completion, usually faster)
batch_job = processor.submit_batch(tasks)
print(f"Batch submitted: {batch_job.id}")

# Wait for completion and get results
results = processor.get_completed_results(batch_job.id, wait=True)
for result in results:
    content = result.response["body"]["choices"][0]["message"]["content"]
    print(f"{result.custom_id}: {content}")
```

### Image Captioning Batch
```python
# Vision batch processing
tasks = []
caption_prompt = "Generate short, descriptive captions for furniture images"

for i, (image_url, title) in enumerate(zip(image_urls, titles)):
    task = processor.create_task(
        custom_id=f"furniture-{i}",
        model="gpt-4o-mini",
        messages=[
            SystemMessage(caption_prompt),
            UserMessage([
                {"type": "text", "text": title},
                {"type": "image_url", "image_url": {"url": image_url}}
            ])
        ],
        temperature=0.2,
        max_tokens=300
    )
    tasks.append(task)

# Process batch
batch_job = processor.submit_batch(tasks)
```

### Batch CLI Operations
```bash
# Create sample input
python examples/batch_cli.py sample

# Submit batch job
python examples/batch_cli.py create sample_movies.txt --model gpt-4o-mini --json-output

# Monitor status
python examples/batch_cli.py status batch_abc123

# Download when complete
python examples/batch_cli.py results batch_abc123 --output movie_results.jsonl --show-sample

# List all jobs
python examples/batch_cli.py list --limit 20

# Cancel if needed
python examples/batch_cli.py cancel batch_abc123
```

## 📊 Response Metadata Structure

### Complete Metadata Access
```python
response.grounding_metadata = {
    "web_search": {              # Web search call info
        "id": "ws_123",
        "status": "completed", 
        "query": "search query",
        "search_type": "search"
    },
    "code_interpreter": {        # Code execution info
        "id": "ci_456",
        "container_id": "cntr_789",
        "code": "print('hello')",
        "output": "hello"
    },
    "image_generation": [{       # Generated images array
        "id": "ig_abc",
        "result": "base64_image_data",
        "revised_prompt": "optimized prompt",
        "size": "1024x1024",
        "quality": "high"
    }],
    "citations": [{             # URL citations from search
        "url": "https://source.com",
        "title": "Article Title",
        "start_index": 100,
        "end_index": 200
    }],
    "sources": [                # All sources consulted
        "https://source1.com",
        "https://source2.com"
    ],
    "file_citations": [{        # Generated files from code
        "file_id": "cfile_123",
        "filename": "chart.png",
        "container_id": "cntr_456"
    }]
}
```

## ⚙️ Advanced Configuration

### Model-Specific Parameters
```python
# Responses API with reasoning (gpt-5, o3)
from llm_studio.models.base import ModelConfig

# Advanced configuration (via provider directly)
from llm_studio.models.openai import OpenAIProvider
provider = OpenAIProvider(api_key=openai_key)

config = ModelConfig(
    provider="openai",
    model="gpt-5", 
    api="responses",
    reasoning={"effort": "high"},
    include=["web_search_call.action.sources"],
    tool_choice="auto"
)

response = provider.generate(messages, config, tools)
```

### Error Handling
```python
try:
    response = agent.generate("Query", tools=["openai_code_interpreter"])
    print(response.content)
    
    # Access metadata safely
    if response.grounding_metadata:
        if "image_generation" in response.grounding_metadata:
            images = response.grounding_metadata["image_generation"]
            for img in images:
                image_data = img["result"]  # Base64 encoded
                
except Exception as e:
    print(f"Error: {e}")
```

## 🧪 Testing

### Real API Tests
```bash
# Test all OpenAI functionality with real API calls
pytest tests/test_openai.py -v -m integration

# Test specific functionality
pytest tests/test_openai.py::TestRealWebSearch -v -m integration
pytest tests/test_openai.py::TestRealCodeInterpreter -v -m integration
pytest tests/test_openai.py::TestRealImageGeneration -v -m integration
```

### Mock Tests (No API Costs)
```bash
# Unit tests with mocked responses
pytest tests/test_openai_mock.py -v
```

## 💡 Best Practices

### Tool Selection Guidelines
- **Web search**: Use `openai_web_search` for current information
- **Code execution**: Use `openai_code_interpreter` for data analysis, math, visualization
- **Image generation**: Use `openai_image_generation` for visual content
- **Batch processing**: Use Batch API for large-scale async operations

### Cost Optimization
- **Batch API**: Use for non-urgent, large-scale processing (significant cost savings)
- **Search models**: Use `gpt-4o-search-preview` for built-in search (no external tools)
- **Model selection**: Use `gpt-4o-mini` for cost-effective tasks

### Performance Tips
- **Streaming images**: Use `partial_images=2` for faster visual feedback
- **Container reuse**: Use explicit containers for multiple related code execution tasks
- **Domain filtering**: Limit web search to relevant domains for better results

## 🚀 Production Deployment

### Environment Setup
```python
import os
from dotenv import load_dotenv
from llm_studio import Agent, setup_logging, LogLevel

load_dotenv()

# Production agent with logging
setup_logging(level=LogLevel.INFO)
agent = Agent(
    provider="openai",
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    system_prompt="You are a helpful AI assistant."
)

# Production-ready with error handling
try:
    response = agent.generate(
        "User query here",
        tools=["openai_web_search", "openai_code_interpreter"]
    )
    
    # Process response and metadata
    result = {
        "content": response.content,
        "metadata": response.grounding_metadata,
        "has_citations": bool(response.grounding_metadata and "citations" in response.grounding_metadata),
        "tools_used": len(response.grounding_metadata.keys()) if response.grounding_metadata else 0
    }
    
except Exception as e:
    print(f"Production error: {e}")
```

---

**OpenAI integration is production-ready with comprehensive tool support, professional logging, batch processing, and robust error handling for enterprise applications.**
