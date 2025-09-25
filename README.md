# llm_studio

A modular, provider-agnostic agent framework for working with multiple LLM providers (OpenAI, Anthropic/Claude, Google/Gemini) and tool calling, with a clean structure designed for future packaging and easy maintenance.

## 🎯 Goals
- Unify provider differences behind a normalized interface
- Support tool/function calling in a provider-neutral way
- Keep zero hard-coding in user flows; everything pluggable
- Provide a simple agent runtime that loops over tool calls

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-repo/llm_studio.git
cd llm_studio
```

2. **Create and activate virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux  
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package in development mode:**
```bash
pip install -e .
```

5. **Test the installation:**
```bash
python examples/quickstart.py
python examples/agent_demo.py
```

### Basic Usage

```python
from llm_studio import Agent
from dotenv import load_dotenv
import os

load_dotenv()

# Create a production agent
agent = Agent(
    provider="openai",  # or "anthropic", "google"
    model="gpt-4o-search-preview",
    api_key=os.getenv("OPENAI_API_KEY"),
    system_prompt="You are a helpful assistant."
)

# Simple conversation
response = agent.generate("What is machine learning?")
print(response.content)

# Using built-in tools (just use string names!)
response = agent.generate(
    "Format this as JSON: name=Alice, age=30",
    tools=["json_format"]  # No imports needed!
)

# Use any of 10 built-in tools by string name!
response = agent.generate(
    "Search for AI news and calculate statistics",
    tools=["openai_web_search", "google_code_execution"]
)

# See all available tools
from llm_studio.tools.registry import list_all_tools
print(f"Available tools: {list(list_all_tools().keys())}")
```

## 📁 Project Structure
- `llm_studio/schemas/*` — normalized message and tool specs
- `llm_studio/models/*` — provider adapters and registry
- `llm_studio/tools/*` — tool base, registry, and built-ins
- `llm_studio/agent/runtime.py` — agent orchestration with tool loop
- `examples/` — demo scripts and usage examples
- `tests/` — comprehensive test suite

## 🤖 Supported Providers
- **`openai`** — Chat Completions + Responses API (with web search)
- **`anthropic`** — Claude Messages API (with web search & web fetch)
- **`google`** — Gemini API with 2.0+ search grounding (search, code execution, URL context, batch processing)
- **`mock`** — offline adapter for local development and testing

## 🔧 Built-in Tools (11 Available)

**All tools accessible by simple string names - no imports needed!**

### Local Tools (executed by agent)
- `"json_format"` — Format data as JSON
- `"fetch_url"` — HTTP GET requests

### Provider-Native Tools (executed server-side)
**OpenAI Tools:**
- `"openai_web_search"` — Web search with domain filtering
- `"openai_web_search_preview"` — Preview version of web search

**Anthropic Tools:**
- `"anthropic_web_search"` — Claude web search
- `"anthropic_web_fetch"` — Claude web content fetching

**Google Tools:**
- `"google_search"` — Gemini 2.0+ search with automatic grounding ✅
- `"google_search_retrieval"` — Legacy search (Gemini 1.5)
- `"google_code_execution"` — Python code execution ✅
- `"google_url_context"` — Direct URL content processing (websites, PDFs, images) ✅
- `"google_image_generation"` — Gemini 2.5+ native image generation ✅

### Simple Usage
```python
# Use any of 13 built-in tools by string name!
response = agent.generate(
    "Calculate factorial using Python",
    tools=["google_code_execution"]  # ✅ Works perfectly!
)

response = agent.generate(
    "Format data as JSON: name=Alice, age=30", 
    tools=["json_format"]  # ✅ Always works
)

# Generic tool names (default to Google)
response = agent.generate(
    "Search for AI research",
    tools=["web_search"]  # Defaults to Google Gemini 2.0+ search
)
```

### Tool Compatibility
- **Google tools**: ✅ All working (`google_code_execution`, `google_search_retrieval`, etc.)
- **Local tools**: ✅ Always working (`json_format`, `fetch_url`)
- **OpenAI web search**: ✅ Working with `gpt-4o-search-preview` (recommended)
- **Generic names**: `web_search`, `code_execution`, `url_context` (use Google)

### Recommended Model + Tool Combinations
```python
# ✅ GUARANTEED WORKING COMBINATIONS

# Google code execution (always works)
google_agent = Agent(provider="google", model="gemini-1.5-flash", api_key=google_key)
response = google_agent.generate("Calculate factorial", tools=["google_code_execution"])

# OpenAI web search (works perfectly)  
openai_agent = Agent(provider="openai", model="gpt-4o-search-preview", api_key=openai_key)
response = openai_agent.generate("Search for news", tools=["openai_web_search"])

# Local tools (always work)
response = agent.generate("Format as JSON", tools=["json_format"])
response = agent.generate("Fetch URL data", tools=["fetch_url"])
```

## 🌟 Usage Examples

### Simple Examples
```python
# Basic chat
response = agent.generate("What is artificial intelligence?")

# Use built-in tools (no imports!)
response = agent.generate("Calculate factorial of 8", tools=["google_code_execution"])
response = agent.generate("Search for AI news", tools=["openai_web_search"])
response = agent.generate("Format data as JSON", tools=["json_format"])

# Multiple tools
response = agent.generate(
    "Research and calculate",
    tools=["google_search", "google_code_execution", "json_format"]
)
```

### Provider-Specific Examples
```python
# Google with code execution
google_agent = Agent(provider="google", model="gemini-1.5-flash", api_key=google_key)
response = google_agent.generate(
    "Solve math problems with Python",
    tools=["google_code_execution"]
)

# OpenAI with web search  
openai_agent = Agent(provider="openai", model="gpt-4o-mini", api_key=openai_key)
response = openai_agent.generate(
    "Find latest AI research",
    tools=["openai_web_search"]
)

# Anthropic with web tools
anthropic_agent = Agent(provider="anthropic", model="claude-3-haiku", api_key=anthropic_key)
response = anthropic_agent.generate(
    "Research and fetch papers",
    tools=["anthropic_web_search", "anthropic_web_fetch"]
)
```

## 🧪 Testing & Troubleshooting

```bash
# Test your setup
python test.py

# Run examples
python examples/quickstart.py
python examples/agent_demo.py
```

### Common Issues & Solutions

#### Empty Response with Tools
**Problem**: `response.content` is empty when using tools

**Root Cause**: Some model/tool combinations need specific configurations

**✅ WORKING SOLUTIONS**:
```python
# For tiderock.com news (user's original request):

# Solution 1: Use OpenAI search model (works perfectly!)
agent = Agent(provider="openai", model="gpt-4o-search-preview", api_key=openai_key)
response = agent.generate("give me news on tiderock.com", tools=["openai_web_search"])

# Solution 2: Use Google search (alternative)
google_agent = Agent(provider="google", model="gemini-1.5-flash", api_key=google_key)
response = google_agent.generate("search for tiderock.com", tools=["google_search_retrieval"])

# Solution 3: Use local fetch tool (direct access)
response = agent.generate("get data from https://tiderock.com", tools=["fetch_url"])

# Solution 4: Use Google code execution (for calculations)
response = google_agent.generate("calculate something", tools=["google_code_execution"])
```

#### Tool Not Supported
If you get "not supported" messages, the system is working correctly and telling you model limitations.

#### Check Available Tools
```python
from llm_studio.tools.registry import list_all_tools
print(list_all_tools())  # Shows all 13 built-in tools
```

## 🔌 Production Setup

### 1. Get API Keys
```bash
# Create .env file in your project root
echo "OPENAI_API_KEY=your-openai-key" >> .env
echo "ANTHROPIC_API_KEY=your-anthropic-key" >> .env
echo "GEMINI_API_KEY=your-google-key" >> .env
```

### 2. Install with API Support
```bash
# Install required SDKs
pip install openai anthropic google-genai python-dotenv

# Test your setup
python test.py
```

### 3. Start Building
```python
from llm_studio import Agent
from dotenv import load_dotenv
import os

load_dotenv()

# Create production agent
agent = Agent(
    provider="openai",  # or "anthropic", "google"
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    system_prompt="You are a helpful assistant."
)

# Start chatting!
response = agent.generate("Hello! How can you help me today?")
print(response.content)
```

### ✅ Production Status
**All providers are now fully integrated and production-ready!**
- ✅ OpenAI Chat Completions + Responses API
- ✅ Anthropic Messages API + Server Tools
- ✅ Google Gemini + Code Execution  
- ✅ Local and server-side tools
- ✅ Robust error handling
- ✅ Type-safe interfaces

## 📚 Documentation

For detailed usage instructions, see [DOCUMENTATION.md](DOCUMENTATION.md)

### Quick Reference
- **Installation**: See above Quick Start section
- **Creating Agents**: `Agent(provider, model, api_key)`
- **Using Tools**: Pass tools to `agent.generate()`
- **Provider Features**: Each provider has unique capabilities
- **Production Deployment**: Environment setup and best practices

## 🏗️ Architecture

The framework uses a clean modular architecture:
- **Agent**: Orchestrates conversations and tool calling
- **Providers**: Normalize different LLM APIs (OpenAI, Anthropic, Google)
- **Tools**: Local execution + provider-native server tools
- **Registries**: Dynamic discovery of providers and tools
- **Schemas**: Type-safe message and tool specifications

**Key principle**: Provider-agnostic interface with pluggable components.
