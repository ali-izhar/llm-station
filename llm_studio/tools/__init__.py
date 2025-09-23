from .registry import register_tool, register_provider_tool
from .fetch_url import FetchUrlTool
from .json_format import JsonFormatTool

# Import provider tool factories
from .web_search.openai import OpenAIWebSearch
from .web_search.anthropic import AnthropicWebSearch
from .web_search.google import GoogleWebSearch, GoogleSearchRetrieval
from .web_fetch.anthropic import AnthropicWebFetch
from .code_execution.google import GoogleCodeExecution
from .url_context.google import GoogleUrlContext

# Register built-in local tools on import
register_tool("fetch_url", FetchUrlTool)
register_tool("json_format", JsonFormatTool)

# Register provider-native tools with default configurations
register_provider_tool("openai_web_search", lambda: OpenAIWebSearch().spec())
register_provider_tool(
    "openai_web_search_preview", lambda: OpenAIWebSearch(preview=True).spec()
)

register_provider_tool("anthropic_web_search", lambda: AnthropicWebSearch().spec())
register_provider_tool("anthropic_web_fetch", lambda: AnthropicWebFetch().spec())

register_provider_tool("google_search", lambda: GoogleWebSearch().spec())
register_provider_tool(
    "google_search_retrieval",
    lambda: GoogleSearchRetrieval(mode="MODE_DYNAMIC", dynamic_threshold=0.7).spec(),
)
register_provider_tool("google_code_execution", lambda: GoogleCodeExecution().spec())
register_provider_tool("google_url_context", lambda: GoogleUrlContext().spec())

# Register alternative web search options with better user experience
register_provider_tool(
    "web_search",
    lambda: GoogleSearchRetrieval(mode="MODE_DYNAMIC", dynamic_threshold=0.7).spec(),
)  # Use working Google search retrieval
register_provider_tool(
    "code_execution", lambda: GoogleCodeExecution().spec()
)  # Default to Google
register_provider_tool(
    "url_context", lambda: GoogleUrlContext().spec()
)  # Default to Google

__all__ = [
    "FetchUrlTool",
    "JsonFormatTool",
    "OpenAIWebSearch",
    "AnthropicWebSearch",
    "AnthropicWebFetch",
    "GoogleWebSearch",
    "GoogleSearchRetrieval",
    "GoogleCodeExecution",
    "GoogleUrlContext",
]
