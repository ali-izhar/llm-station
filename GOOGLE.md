# Google Gemini Provider Documentation

> Run `python examples/google_quickstart.py` to test everything instantly!

LLM Studio provides comprehensive integration with Google's Gemini models, featuring advanced multimodal capabilities and native tool support:

```python
# Same interface works with any provider
agent = Agent(provider="google", model="gemini-2.5-flash", api_key=gemini_key)  # Google
agent = Agent(provider="openai", model="gpt-4o", api_key=openai_key)  # OpenAI
agent = Agent(provider="anthropic", model="claude-3", api_key=anthropic_key)  # Anthropic

# Logging, runtime, and utilities work identically across all providers
```

Google-Specific Features:

- **Advanced Search Grounding**: Automatic web search with citations and sources
- **Native Code Execution**: Python code generation and execution with visualization
- **URL Context Processing**: Direct analysis of websites, PDFs, and images
- **Native Image Generation**: Gemini 2.5+ multimodal image creation and editing
- **Multimodal Understanding**: Process text, images, video, audio, and documents

## Quick Start

```bash
# Install dependencies
pip install -U google-genai python-dotenv

# Set API key in .env
echo "GEMINI_API_KEY=your-gemini-key" >> .env
# or
echo "GOOGLE_API_KEY=your-google-key" >> .env

# Run quickstart
python examples/google_quickstart.py
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
    provider="google",
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    system_prompt="You are a helpful research assistant."
)

# Basic chat
response = agent.generate("What is quantum computing?")
print(response.content)

# Web search with automatic grounding
response = agent.generate("What's happening in AI this week?", tools=["google_search"])
print(response.content)

# Code execution with visualization
response = agent.generate("Calculate factorial of 8 and plot the results", tools=["google_code_execution"])
print(response.content)

# URL content analysis
response = agent.generate("Analyze this documentation: https://ai.google.dev/gemini-api/docs", tools=["google_url_context"])
print(response.content)

# Image generation (requires gemini-2.5-flash-image-preview)
image_agent = Agent(provider="google", model="gemini-2.5-flash-image-preview", api_key=api_key)
response = image_agent.generate("Create an image of a robot in a library", tools=["google_image_generation"])
print(response.content)
```

## Google Gemini Models

LLM Studio supports all current and future Gemini models through the `google-genai` SDK:

```python
# Latest stable models (recommended)
agent = Agent(provider="google", model="gemini-2.5-flash", api_key=key)  # Best price-performance
agent = Agent(provider="google", model="gemini-2.5-pro", api_key=key)   # Maximum capability

# Specialized models
agent = Agent(provider="google", model="gemini-2.5-flash-image-preview", api_key=key)  # Image generation
agent = Agent(provider="google", model="gemini-2.5-flash-live", api_key=key)          # Live interactions

# Legacy models (still supported)
agent = Agent(provider="google", model="gemini-1.5-pro", api_key=key)
agent = Agent(provider="google", model="gemini-1.5-flash", api_key=key)
```

### Model Capabilities

| Model | Search | Code Exec | URL Context | Image Gen | Best For |
|-------|---------|-----------|-------------|-----------|-----------|
| `gemini-2.5-pro` | ✅ | ✅ | ✅ | ❌ | Complex reasoning, large context |
| `gemini-2.5-flash` | ✅ | ✅ | ✅ | ❌ | General use, best price-performance |
| `gemini-2.5-flash-image-preview` | ✅ | ✅ | ✅ | ✅ | Image generation and editing |
| `gemini-2.0-flash` | ✅ | ✅ | ✅ | ❌ | Fast responses, tool use |
| `gemini-1.5-pro` | ✅* | ✅ | ✅ | ❌ | Legacy, large context |

*Uses legacy search grounding with dynamic threshold

## Google Tools

### Search Grounding Tool

```python
from llm_studio.tools.web_search.google import GoogleWebSearch

# Basic search with automatic grounding (Gemini 2.0+)
tool = GoogleWebSearch()
response = agent.generate("Latest AI developments", tools=[tool.spec()])

# Access grounding metadata
if response.grounding_metadata:
    sources = response.grounding_metadata.get("sources", [])
    citations = response.grounding_metadata.get("citations", [])
    search_entry_point = response.grounding_metadata.get("search_entry_point")
    
    print(f"Found {len(sources)} sources and {len(citations)} citations")
    if search_entry_point:
        print("Google Search Suggestions available")
```

#### Legacy Search (Gemini 1.5)

```python
from llm_studio.tools.web_search.google import GoogleSearchRetrieval

# Legacy search with dynamic threshold
search_tool = GoogleSearchRetrieval(
    mode="MODE_DYNAMIC",
    dynamic_threshold=0.7  # Only search if confidence > 70%
)

response = agent.generate("Research quantum computing", tools=[search_tool.spec()])
```

### Code Execution Tool

```python
from llm_studio.tools.code_execution.google import GoogleCodeExecution

# Python code execution with visualization
tool = GoogleCodeExecution()
response = agent.generate(
    "Calculate the first 50 prime numbers and create a visualization",
    tools=[tool.spec()]
)

# Access execution metadata
if response.grounding_metadata:
    code_executions = response.grounding_metadata.get("code_execution", [])
    inline_media = response.grounding_metadata.get("inline_media", [])
    
    print(f"Executed {len(code_executions)} code blocks")
    print(f"Generated {len(inline_media)} media files")
```

#### Advanced Code Analysis

```python
# Multi-step analysis with iterative refinement
response = agent.generate("""
    Analyze this dataset and create visualizations:
    1. Load sample data about California housing
    2. Calculate statistics and correlations
    3. Create multiple plots showing key insights
    4. Identify any anomalies or interesting patterns
""", tools=["google_code_execution"])

# Code execution supports:
# - Data analysis (pandas, numpy, scipy)
# - Visualization (matplotlib, seaborn, plotly)
# - Machine learning (scikit-learn)
# - Image processing (PIL, opencv)
# - File I/O and data processing
```

### URL Context Tool

```python
from llm_studio.tools.url_context.google import GoogleUrlContext

# Direct URL content processing
tool = GoogleUrlContext()

# Website analysis
response = agent.generate(
    "Based on https://ai.google.dev/gemini-api/docs, summarize the key features",
    tools=[tool.spec()]
)

# PDF analysis
response = agent.generate(
    "Analyze this research paper: https://arxiv.org/pdf/2301.00000.pdf",
    tools=[tool.spec()]
)

# Image analysis
response = agent.generate(
    "Describe the components in this diagram: https://example.com/diagram.png",
    tools=[tool.spec()]
)

# Access URL processing metadata
if response.grounding_metadata:
    processed_urls = response.grounding_metadata.get("processed_urls", [])
    for url_info in processed_urls:
        print(f"URL: {url_info['url']}")
        print(f"Status: {url_info['status']}")
        print(f"Content Type: {url_info['content_type']}")
```

#### Supported Content Types
- **Websites**: HTML pages, documentation, blogs
- **Documents**: PDF files, research papers, reports
- **Images**: PNG, JPEG, diagrams, charts, photos
- **Data**: JSON, CSV, XML, plain text files
- **Code**: Source code files, documentation

### Image Generation Tool

```python
from llm_studio.tools.image_generation.google import GoogleImageGeneration

# Basic image generation (requires gemini-2.5-flash-image-preview)
tool = GoogleImageGeneration()
image_agent = Agent(
    provider="google", 
    model="gemini-2.5-flash-image-preview", 
    api_key=api_key
)

# Single image generation
response = image_agent.generate(
    "Create a photorealistic image of a robot reading in a futuristic library",
    tools=[tool.spec()]
)

# Multi-image storytelling
response = image_agent.generate(
    "Create a 4-part visual story about a space explorer's adventure",
    tools=[tool.spec()]
)

# Image editing and consistency
response1 = image_agent.generate("Create a cartoon fox character")
response2 = image_agent.generate("Show the same fox in a forest setting")
response3 = image_agent.generate("Now show the fox cooking in a kitchen")

# Access image metadata
if response.grounding_metadata:
    images = response.grounding_metadata.get("image_generation", [])
    for img_info in images:
        print(f"Image type: {img_info['type']}")
        print(f"Format: {img_info['format']}")
```

#### Image Generation Features
- **Character Consistency**: Maintain subjects across multiple images
- **Intelligent Editing**: Natural language image modifications
- **Scene Composition**: Combine multiple elements
- **Style Control**: Artistic styles and transformations
- **Iterative Refinement**: Chat-based image improvement

## Combined Tool Workflows

### Research and Analysis Pipeline

```python
# Comprehensive research workflow
response = agent.generate("""
    Research Tesla's recent performance:
    1. Search for latest Tesla news and financial data
    2. Analyze the company documentation from their investor page
    3. Calculate key financial ratios using Python
    4. Create visualization of stock performance trends
    
    Provide a comprehensive analysis with citations.
""", tools=["google_search", "google_url_context", "google_code_execution"])

# Access all metadata types
if response.grounding_metadata:
    metadata_types = list(response.grounding_metadata.keys())
    print(f"Tools used: {metadata_types}")
```

### Creative Content Generation

```python
# Story with images
image_agent = Agent(provider="google", model="gemini-2.5-flash-image-preview", api_key=key)

response = image_agent.generate("""
    Create an educational story about space exploration:
    1. Write a short story about an astronaut's mission
    2. Generate 3 images showing key moments
    3. Include factual information about space travel
    4. Search for recent space mission news to make it current
""", tools=["google_search", "google_image_generation"])
```

### Data Analysis with Context

```python
# Research-backed data analysis
response = agent.generate("""
    Analyze global renewable energy trends:
    1. Search for latest renewable energy statistics
    2. Find and analyze data from energy.gov or similar sources
    3. Calculate growth rates and projections
    4. Create comprehensive visualizations
    5. Identify key insights and recommendations
""", tools=["google_search", "google_url_context", "google_code_execution"])
```

## Response Metadata Structure

```python
response.grounding_metadata = {
    "grounding": {                    # Search grounding from Gemini 2.0+
        "search_entry_point": {...},
        "grounding_chunks": [...],
        "web_search_queries": [...]
    },
    "sources": [...],                 # Extracted source URLs
    "citations": [{                   # Detailed citations with snippets
        "url": "https://source.com",
        "title": "Article Title",
        "snippet": "Relevant excerpt..."
    }],
    "code_execution": [{              # Code execution details
        "code": "python_code_here",
        "language": "python",
        "result": {
            "output": "execution_output",
            "outcome": "OUTCOME_OK"
        }
    }],
    "inline_media": [{                # Generated visualizations
        "mime_type": "image/png",
        "data": "base64_data",
        "size": 21071
    }],
    "url_context": [...],             # URL processing metadata
    "processed_urls": [{              # URL processing status
        "url": "https://processed.com",
        "status": "success",
        "content_type": "text/html",
        "size": 50000
    }],
    "image_generation": [{            # Generated images
        "type": "native_generation",
        "available": True,
        "format": "PIL_Image"
    }]
}
```

## Advanced Configuration

### System Instructions and Context

```python
# Research assistant with specific focus
agent = Agent(
    provider="google",
    model="gemini-2.5-pro",
    api_key=api_key,
    system_prompt="""You are an expert research analyst specializing in technology trends.
    Always provide citations for your claims and use data-driven analysis.
    When generating visualizations, make them publication-ready with proper labels."""
)

# Scientific analysis agent
science_agent = Agent(
    provider="google",
    model="gemini-2.5-flash",
    api_key=api_key,
    system_prompt="""You are a scientific research assistant. Always:
    1. Verify information from multiple sources
    2. Show your calculations and methodology
    3. Include statistical analysis when relevant
    4. Generate clear, informative visualizations"""
)
```

### Temperature and Generation Control

```python
# Precise technical analysis
agent = Agent(
    provider="google",
    model="gemini-2.5-pro",
    api_key=api_key,
    temperature=0.1,  # Low temperature for consistency
    max_tokens=8192   # Extended responses
)

# Creative content generation
creative_agent = Agent(
    provider="google",
    model="gemini-2.5-flash-image-preview",
    api_key=api_key,
    temperature=0.8   # Higher temperature for creativity
)
```

## Production Best Practices

### Error Handling and Resilience

```python
from llm_studio import Agent
import logging

# Enable comprehensive logging
setup_logging(level=LogLevel.DEBUG)

try:
    agent = Agent(provider="google", model="gemini-2.5-flash", api_key=api_key)
    
    response = agent.generate(
        "Analyze market trends with visualizations",
        tools=["google_search", "google_code_execution"]
    )
    
    # Check for successful tool execution
    if response.grounding_metadata:
        if "code_execution" in response.grounding_metadata:
            for exec_info in response.grounding_metadata["code_execution"]:
                if exec_info.get("result", {}).get("outcome") != "OUTCOME_OK":
                    logging.warning("Code execution had issues")
        
        if "sources" in response.grounding_metadata:
            logging.info(f"Found {len(response.grounding_metadata['sources'])} sources")
    
except Exception as e:
    logging.error(f"Analysis failed: {e}")
```

### Performance Optimization

```python
# Use appropriate models for different tasks
def get_optimal_agent(task_type: str):
    if task_type == "simple_qa":
        return Agent(provider="google", model="gemini-2.5-flash", api_key=api_key)
    elif task_type == "complex_analysis":
        return Agent(provider="google", model="gemini-2.5-pro", api_key=api_key)
    elif task_type == "image_generation":
        return Agent(provider="google", model="gemini-2.5-flash-image-preview", api_key=api_key)
    elif task_type == "real_time":
        return Agent(provider="google", model="gemini-2.5-flash-lite", api_key=api_key)

# Batch similar requests for efficiency
research_tasks = [
    "Analyze renewable energy trends",
    "Research electric vehicle adoption",
    "Study carbon capture technologies"
]

results = []
agent = get_optimal_agent("complex_analysis")
for task in research_tasks:
    result = agent.generate(task, tools=["google_search", "google_code_execution"])
    results.append(result)
```

### Content Moderation and Safety

```python
# Google's safety features are built-in, but you can add additional checks
def safe_generate(agent, prompt, tools=None):
    """Generate content with additional safety checks."""
    
    # Pre-filter sensitive content
    sensitive_keywords = ["personal_info", "private_data", "credentials"]
    if any(keyword in prompt.lower() for keyword in sensitive_keywords):
        return "Cannot process requests for sensitive information"
    
    response = agent.generate(prompt, tools=tools)
    
    # Post-process for additional safety
    if response.grounding_metadata and "sources" in response.grounding_metadata:
        # Log sources for audit trail
        logging.info(f"Response used {len(response.grounding_metadata['sources'])} sources")
    
    return response
```

## Batch API for Large-Scale Processing

Google's Batch API provides high-throughput, cost-effective processing for large datasets:

### Benefits
- **50% cost savings** compared to standard API
- **High throughput**: Process millions of requests
- **24-hour completion window** with asynchronous processing
- **File-based and inline** request modes

### Basic Batch Processing

```python
from llm_studio import GoogleBatchProcessor, SystemMessage, UserMessage

processor = GoogleBatchProcessor(api_key=gemini_key)

# Create batch tasks
tasks = []
for i, topic in enumerate(research_topics):
    task = processor.create_task(
        key=f"research-{i}",
        model="gemini-2.5-flash",
        contents=f"Research and analyze: {topic}",
        system_instruction="You are a research analyst. Provide comprehensive analysis.",
        generation_config={"temperature": 0.2}
    )
    tasks.append(task)

# Submit file-based batch (recommended for large jobs)
batch_job = processor.submit_batch(tasks, display_name="research-analysis")
print(f"Batch submitted: {batch_job.name}")

# Wait for completion and get results  
results = processor.get_completed_results(batch_job.name, wait=True)
for result in results:
    print(f"{result.key}: {result.response}")
```

### Inline Batch Processing

```python
# For smaller jobs, use inline processing (no file upload)
inline_tasks = []
for i, text in enumerate(short_texts):
    task = processor.create_task(
        key=f"inline-{i}",
        model="gemini-2.5-flash", 
        contents=text,
        system_instruction="Summarize in one sentence"
    )
    inline_tasks.append(task)

# Submit inline batch
batch_job = processor.submit_inline_batch(inline_tasks, display_name="quick-summaries")
results = processor.get_completed_results(batch_job.name)
```

### Multimodal Batch Processing

```python
# Batch process images with text
content_pairs = [
    [
        {"text": "Describe this image in detail"},
        {"file_data": {"file_uri": image_uri, "mime_type": "image/jpeg"}}
    ]
    for image_uri in image_uris
]

file_path = processor.process_multimodal_batch(
    content_pairs=content_pairs,
    system_instruction="Provide detailed image analysis",
    model="gemini-2.5-flash"
)

# Upload and process
file_name = processor.upload_batch_file(file_path)
batch_job = processor.create_batch_job(model="gemini-2.5-flash", src=file_name)
```

### Batch Embeddings

```python
# High-throughput text embeddings
texts = ["Text 1", "Text 2", "Text 3", ...]

batch_job = processor.create_embeddings_batch(
    texts=texts,
    model="gemini-embedding-001",
    output_dimensionality=768
)

results = processor.get_completed_results(batch_job.name)
for result in results:
    embedding = result.response["embedding"]["values"]
    print(f"Embedding dimension: {len(embedding)}")
```

### Image Generation Batch

```python
# Batch generate images
prompts = [
    "A robot in a futuristic city",
    "A cat in a magical forest", 
    "A spaceship among the stars"
]

file_path = create_image_generation_batch(
    prompts=prompts,
    processor=processor,
    model="gemini-2.5-flash-image-preview"
)

# Submit and process
file_name = processor.upload_batch_file(file_path)
batch_job = processor.create_batch_job(model="gemini-2.5-flash-image-preview", src=file_name)
```

### Job Management

```python
# List all batch jobs
jobs = processor.list_batch_jobs(page_size=10)
for job in jobs:
    print(f"Job: {job.name} - Status: {job.state.value}")

# Monitor specific job
batch_job = processor.get_batch_status("batches/your-job-name")
print(f"Status: {batch_job.state.value}")

# Cancel if needed
cancelled_job = processor.cancel_batch_job("batches/job-to-cancel")
```