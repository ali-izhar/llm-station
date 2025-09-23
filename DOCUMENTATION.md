# LLM Studio Documentation

Build multi-provider AI applications with ease.

## 🚀 Installation

```bash
git clone https://github.com/your-repo/llm_studio.git
cd llm_studio
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -e .
pip install openai anthropic google-genai python-dotenv
```

## ⚡ Quick Start

### 1. Setup API Keys
```bash
# Create .env file
echo "OPENAI_API_KEY=your-key" >> .env
echo "ANTHROPIC_API_KEY=your-key" >> .env  
echo "GEMINI_API_KEY=your-key" >> .env
```

### 2. Basic Usage
```python
from llm_studio import Agent
from dotenv import load_dotenv
import os

load_dotenv()

# Create agent
agent = Agent(
    provider="openai",  # or "anthropic", "google"
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Chat
response = agent.generate("What is machine learning?")
print(response.content)
```

### 3. Test Setup
```bash
python test.py
```

## 🤖 Creating Agents

```python
# OpenAI
agent = Agent(provider="openai", model="gpt-4o-mini", api_key=openai_key)

# Anthropic  
agent = Agent(provider="anthropic", model="claude-3-haiku-20240307", api_key=anthropic_key)

# Google
agent = Agent(provider="google", model="gemini-1.5-flash", api_key=google_key)

# With options
agent = Agent(
    provider="openai",
    model="gpt-4o-mini", 
    api_key=api_key,
    system_prompt="You are a helpful assistant",
    temperature=0.7,
    max_tokens=1000
)
```

## 🔧 Using Tools

### All Tools by String Name (10 Built-in)
**No imports needed - just use string names!**

```python
# Local tools
response = agent.generate("Format as JSON: name=Alice", tools=["json_format"])
response = agent.generate("Get https://api.github.com", tools=["fetch_url"])

# Provider tools  
response = agent.generate("Search AI news", tools=["openai_web_search"])
response = agent.generate("Calculate factorial", tools=["google_code_execution"])
response = agent.generate("Research papers", tools=["anthropic_web_search"])

# Mix any tools
response = agent.generate(
    "Complex research task", 
    tools=["json_format", "google_code_execution", "google_search"]
)
```

### Available Built-in Tools
**Local Tools:**
- `"json_format"` - Format data as JSON
- `"fetch_url"` - HTTP GET requests

**OpenAI Tools:**  
- `"openai_web_search"` - Web search with domain filtering
- `"openai_web_search_preview"` - Preview version

**Anthropic Tools:**
- `"anthropic_web_search"` - Claude web search  
- `"anthropic_web_fetch"` - Claude web fetch

**Google Tools:**
- `"google_search"` - Search with citations
- `"google_search_retrieval"` - Legacy search (Gemini 1.5)
- `"google_code_execution"` - Python code execution ✅
- `"google_url_context"` - URL content analysis

```python
# Check available tools
from llm_studio.tools.registry import list_all_tools
print(list_all_tools())
```

## 🎯 Examples

### Research Assistant
```python
# Google agent with multiple tools
agent = Agent(provider="google", model="gemini-1.5-flash", api_key=google_key)

response = agent.generate(
    "Research quantum computing and calculate examples",
    tools=["google_search", "google_code_execution", "json_format"]
)
```

### Content Analyzer  
```python
# Anthropic agent with web tools
agent = Agent(provider="anthropic", model="claude-3-haiku", api_key=anthropic_key)

response = agent.generate(
    "Analyze research papers and summarize",
    tools=["anthropic_web_search", "anthropic_web_fetch"]
)
```

### Data Processor
```python
# OpenAI agent with search and formatting
agent = Agent(provider="openai", model="gpt-4o-mini", api_key=openai_key)

response = agent.generate(
    "Search for data and format results",
    tools=["openai_web_search", "json_format", "fetch_url"]
)
```

### Multi-Provider Setup
```python
# Use different providers for different tasks
agents = {
    "chat": Agent(provider="anthropic", model="claude-3-haiku", api_key=anthropic_key),
    "code": Agent(provider="google", model="gemini-1.5-flash", api_key=google_key),
    "search": Agent(provider="openai", model="gpt-4o-mini", api_key=openai_key)
}

# Chat
chat_response = agents["chat"].generate("Hello! How are you?")

# Code execution
code_response = agents["code"].generate(
    "Calculate prime numbers up to 100", 
    tools=["google_code_execution"]
)

# Web search
search_response = agents["search"].generate(
    "Find latest AI research", 
    tools=["openai_web_search"]
)
```

## 📚 API Reference

### Agent
```python
Agent(
    provider: str,           # "openai", "anthropic", "google", "mock"
    model: str,              # Provider-specific model name
    api_key: str = None,     # API key for provider
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None
)
```

### Generate Method
```python
agent.generate(
    prompt: str,                    # User prompt
    tools: List[str] = None,        # Tool names (no imports needed!)
    structured_schema: Dict = None, # JSON schema for output
    max_tool_rounds: int = 4        # Max tool calling rounds
) -> AssistantMessage
```

### Response Object
```python
response.content          # Generated text
response.tool_calls       # List of ToolCall objects
response.grounding_metadata  # Search metadata (Google only)
```

## 🔧 Advanced Usage

### Error Handling
```python
try:
    response = agent.generate("Your query", tools=["google_code_execution"])
    print(response.content)
except Exception as e:
    print(f"Error: {e}")
```

### Multi-Provider Fallback
```python
providers = ["openai", "anthropic", "google"]
api_keys = [openai_key, anthropic_key, google_key]

for provider, key in zip(providers, api_keys):
    if key:
        try:
            agent = Agent(provider=provider, model=f"{provider}-model", api_key=key)
            response = agent.generate("Test query")
            break
        except:
            continue
```

### Custom Tool Creation
```python
from llm_studio.tools.base import Tool
from llm_studio.tools.registry import register_tool

class MyTool(Tool):
    def spec(self):
        return ToolSpec(name="my_tool", description="Custom tool", input_schema={})
    
    def run(self, *, tool_call_id: str, **kwargs):
        return ToolResult(name="my_tool", content="result", tool_call_id=tool_call_id)

register_tool("my_tool", MyTool)

# Now use it
response = agent.generate("Use my custom tool", tools=["my_tool"])
```

## 🚀 Production Ready

### Test Your Setup
```bash
python test.py  # Tests all providers and tools
```

### Environment Setup
```python
# production.py
import os
from dotenv import load_dotenv
from llm_studio import Agent

load_dotenv()

def create_agent(provider="openai"):
    keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "google": os.getenv("GEMINI_API_KEY")
    }
    
    models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307", 
        "google": "gemini-1.5-flash"
    }
    
    return Agent(
        provider=provider,
        model=models[provider],
        api_key=keys[provider],
        system_prompt="You are a helpful AI assistant."
    )

# Usage
agent = create_agent("google")
response = agent.generate("Calculate something", tools=["google_code_execution"])
```

---

**That's it! Your llm_studio framework is production-ready.** 🎉

Start building AI applications with:
1. **Create agent** with your preferred provider
2. **Chat** with `agent.generate()`  
3. **Use tools** by string names
4. **Deploy** to production

For more examples, check `examples/` directory and run `python test.py` to verify everything works!
