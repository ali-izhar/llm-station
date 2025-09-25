# OpenAI Provider Documentation

> Run `python examples/openai_quickstart.py` to test everything instantly!

LLM Studio uses a clean, extensible architecture where OpenAI is one provider among many:

```python
# Same interface works with any provider
agent = Agent(provider="openai", model="gpt-4o", api_key=openai_key) # OpenAI
agent = Agent(provider="anthropic", model="claude-3", api_key=anthropic_key) # Anthropic  
agent = Agent(provider="google", model="gemini-1.5", api_key=google_key) # Google

# Logging, runtime, and utilities work identically across all providers
```

 OpenAI-Specific Features:

- **API selection**: Automatic Chat Completions vs Responses API routing
- **Tools**: Web search, code interpreter, image generation, batch processing
- **Metadata**: Citations, sources, generated files, containers

## Quick Start

```bash
# Install dependencies
pip install openai python-dotenv

# Set API key in .env
echo "OPENAI_API_KEY=your-openai-key" >> .env

# Run quickstart
python examples/openai_quickstart.py
```

## Usage
```python
from llm_studio import Agent, setup_logging, LogLevel
from dotenv import load_dotenv
import os

load_dotenv()

# Enable logging (auto-saves to logs/)
setup_logging(level=LogLevel.INFO)

# Create agent
agent = Agent(
    provider="openai",
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    system_prompt="You are a helpful assistant."
)

# Basic chat
response = agent.generate("What is 2 + 2?")
print(response.content)

# Web search (automatically uses Responses API)
response = agent.generate("What's in AI news?", tools=["openai_web_search"])
print(response.content)

# Code execution 
response = agent.generate("Calculate factorial of 5", tools=["openai_code_interpreter"])
print(response.content)

# Image generation
# Note: check if your api key is compatible with image-generation models like GPT-Image-1
response = agent.generate("Draw a red circle", tools=["openai_image_generation"])
print(response.content)
```

## OpenAI Models

LLM Studio uses **dynamic pattern-based detection** instead of hardcoded model lists, ensuring compatibility with all current and future OpenAI models:

```python
# Any OpenAI model works automatically
agent = Agent(provider="openai", model="your-preferred-model", api_key=key)

# Pattern-based feature detection:
# - Models ending in "-search" → Built-in search capabilities
# - Models containing "gpt-5", "o3", "o4" → Reasoning support
# - Models containing "gpt-4o" → Temperature restrictions in Responses API
```

## OpenAI Tools

### Available Tools
- **`openai_web_search`** - Web search with citations (Responses API)
- **`openai_web_search_preview`** - Preview version of web search
- **`openai_code_interpreter`** - Python code execution in containers (Responses API) 
- **`openai_image_generation`** - AI image generation and editing (Responses API)

```python
# Just use tool names
response = agent.generate("Search for AI news", tools=["openai_web_search"])
response = agent.generate("Calculate factorial", tools=["openai_code_interpreter"]) 
response = agent.generate("Draw a sunset", tools=["openai_image_generation"])

# Multiple tools in one request
response = agent.generate(
    "Research AI trends, analyze data, create chart",
    tools=["openai_web_search", "openai_code_interpreter", "openai_image_generation"]
)

# Access rich metadata
if response.grounding_metadata:
    print(f"Tools used: {list(response.grounding_metadata.keys())}")
    if "citations" in response.grounding_metadata:
        print(f"Found {len(response.grounding_metadata['citations'])} citations")
```

## Web Search Tool

```python
from llm_studio.tools.web_search.openai import OpenAIWebSearch

# Basic search
tool = OpenAIWebSearch()
response = agent.generate("Search query", tools=[tool.spec()])

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

## Code Interpreter Tool

```python
from llm_studio.tools.code_execution.openai import OpenAICodeInterpreter

# Auto container (recommended)
tool = OpenAICodeInterpreter()
response = agent.generate(
    "Solve the equation 3x + 11 = 14 using Python",
    tools=[tool.spec()]
)

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

## Image Generation Tool

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

## Batch API

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


### Vision Batch Processing
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

## Manual API Control
```python
# Force specific API when needed
agent = Agent(
    provider="openai",
    model="gpt-4o",
    api_key=key,
    api="responses"  # Force Responses API
)

# Or via provider_kwargs for additional control
agent = Agent(
    provider="openai",
    model="gpt-5", 
    api_key=key,
    api="responses",
    reasoning={"effort": "high"},
    tool_choice="auto"
)
```

## Batch API for Large-Scale Processing

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

## Response Metadata Structure

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

## Advanced Configuration

### OpenAI-Specific Parameters
```python
# Advanced OpenAI configuration using provider_kwargs
agent = Agent(
    provider="openai",
    model="gpt-5",
    api_key=openai_key,
    # OpenAI-specific parameters via provider_kwargs
    api="responses",                              # Force Responses API
    reasoning={"effort": "high"},                 # Reasoning level
    include=["web_search_call.action.sources"],   # Include sources
    tool_choice="auto",                          # Tool selection
    instructions="Custom instructions for model" # Responses API instructions
)

response = agent.generate("Complex query", tools=["openai_web_search"])
```

### Provider-Agnostic vs Provider-Specific
```python
# Provider-agnostic parameters (work with all providers)
agent = Agent(
    provider="openai",  # or "anthropic", "google"
    model="your-model",
    api_key=key,
    temperature=0.7,     # Universal parameter
    max_tokens=1000,     # Universal parameter
    top_p=0.9           # Universal parameter
)

# OpenAI-specific parameters (only for OpenAI)
agent = Agent(
    provider="openai",
    model="gpt-5",
    api_key=key,
    # OpenAI-specific via provider_kwargs:
    api="responses",                    # API selection
    reasoning={"effort": "medium"},     # Reasoning configuration
    tool_choice="auto"                 # Tool usage preference
)
```
