# Changelog

All notable changes to LLM Station will be documented in this file.

## [1.0.0] - 2025-09-25

### 🎉 Initial Release

**Major Features:**
- **Smart Tools System**: Provider-agnostic tool interface with simple, memorable names
- **Multi-Provider Support**: OpenAI, Anthropic Claude, and Google Gemini integration
- **Automatic Routing**: Intelligent tool routing to best available provider
- **Batch Processing**: Cost-effective batch operations for all providers
- **Comprehensive Logging**: Detailed session tracking and debugging

**Smart Tools:**
- `search` - Web search and research across all providers
- `code` - Code execution and data analysis
- `image` - Image generation and creation  
- `json` - JSON formatting and parsing
- `fetch` - URL fetching and downloading
- `url` - URL content processing and extraction

**Provider Features:**
- **OpenAI**: Chat Completions + Responses API, web search, code interpreter, image generation
- **Anthropic**: Messages API, web search, code execution, token management
- **Google**: Gemini 2.0+ search grounding, code execution, URL context, image generation

**Developer Experience:**
- Simple API: `agent.generate("prompt", tools=["search", "code"])`
- Tool aliases: `websearch`, `python`, `draw`, `execute`, etc.
- Provider preferences: `{"name": "search", "provider_preference": "google"}`
- Cross-provider compatibility: Same tools work with any provider
- Comprehensive documentation and examples

**Architecture:**
- Clean, maintainable codebase with zero legacy debt
- Comprehensive test suite with 66 tests using mocks
- Production-ready logging and error handling
- Extensible design for future providers and tools

### 🛠️ Technical Details

**Core Components:**
- Agent runtime with smart tool integration
- Provider adapters for API normalization
- Tool registry with intelligent routing
- Batch processors for cost-effective operations
- Logging system with multiple output formats

**Supported Models:**
- OpenAI: GPT-4o, GPT-4o-mini, GPT-5, and more
- Anthropic: Claude Opus 4.1, Claude Sonnet 4, Claude Haiku
- Google: Gemini 2.5 Flash, Gemini 2.5 Pro, Gemini 2.0 Flash

**Installation:**
```bash
pip install llm-station[all]  # All providers
pip install llm-station[openai]  # OpenAI only
pip install llm-station[anthropic]  # Anthropic only
pip install llm-station[google]  # Google only
```

### 📚 Documentation

- Complete setup guides for each provider
- Usage examples and best practices
- API reference documentation
- Migration guides and troubleshooting

---

**Full Changelog**: This is the initial release of LLM Station.
