#!/usr/bin/env python3
from __future__ import annotations

import os
import requests
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

from ..provider import Provider
from ..types import Message, ModelConfig, ModelResponse, ToolCall, ToolSpec


@dataclass
class HuggingFaceConfig:
    """HuggingFace-specific configuration parameters."""

    reasoning_level: Optional[str] = None  # "low", "medium", "high"
    use_web_search: Optional[bool] = None  # Enable/disable web search
    extract_full_text: Optional[bool] = True  # Extract full text from URLs
    max_urls: Optional[int] = 3  # Max URLs to extract full text from
    num_search_results: Optional[int] = 5  # Number of search results


class HuggingFaceProvider(Provider):
    """Adapter for Hugging Face Inference Endpoint with SerpAPI web search support.

    Supports web search integration via SerpAPI when enabled.
    Uses gpt-oss-20b model via Hugging Face Inference Endpoint.
    """

    name = "huggingface"

    # Default configuration constants
    DEFAULT_MAX_URL_LENGTH = 5000
    DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
    DEFAULT_URL_EXTRACTION_TIMEOUT_SECONDS = 10
    DEFAULT_MAX_URLS = 3
    DEFAULT_NUM_SEARCH_RESULTS = 5

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        serpapi_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Get API key from parameter or environment variable
        hf_token = api_key or os.getenv("HUGGING_FACE_API_TOKEN")
        super().__init__(api_key=hf_token, **kwargs)
        self.endpoint_url = endpoint_url or os.getenv("HUGGING_FACE_ENDPOINT_URL")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")

        # Validate credentials
        if not self.api_key:
            raise ValueError(
                "HuggingFace API key not provided. Set api_key parameter or HUGGING_FACE_API_TOKEN environment variable"
            )
        if not self.endpoint_url:
            raise ValueError(
                "HuggingFace endpoint URL not provided. Set endpoint_url parameter or HUGGING_FACE_ENDPOINT_URL environment variable"
            )

    def supports_tools(self) -> bool:
        return True

    def prepare_tools(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        """Map normalized ToolSpec to HuggingFace tools format.

        HuggingFace provider supports web search via SerpAPI integration.
        Web search is handled server-side by augmenting prompts with search results.
        """
        prepared: List[Dict[str, Any]] = []
        for t in tools:
            if t.provider == "huggingface" and t.provider_type == "web_search":
                # Web search is handled via prompt augmentation
                # Store config for use in generate method
                prepared.append(
                    {
                        "type": "web_search",
                        "provider": "huggingface",
                        "config": t.provider_config or {},
                    }
                )
            else:
                # Standard function tools (local execution)
                prepared.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        },
                    }
                )
        return prepared

    @staticmethod
    def _map_messages(messages: List[Message]) -> List[Dict[str, Any]]:
        """Map normalized messages to HuggingFace chat format."""
        mapped: List[Dict[str, Any]] = []
        for m in messages:
            obj: Dict[str, Any] = {"role": m.role, "content": m.content}
            if m.name:
                obj["name"] = m.name
            if m.role == "tool" and m.tool_call_id:
                obj["tool_call_id"] = m.tool_call_id
            mapped.append(obj)
        return mapped

    def _perform_web_search(
        self, query: str, num_results: int = None
    ) -> List[Dict[str, Any]]:
        """Perform web search using SerpAPI."""
        if num_results is None:
            num_results = self.DEFAULT_NUM_SEARCH_RESULTS
        if not self.serpapi_key:
            raise ValueError(
                "SerpAPI key not configured. Set serpapi_key or SERPAPI_API_KEY environment variable"
            )

        try:
            from serpapi import GoogleSearch

            params = {"q": query, "api_key": self.serpapi_key, "num": num_results}

            search = GoogleSearch(params)
            results = search.get_dict()
            return results.get("organic_results", [])
        except (ImportError, AttributeError) as e:
            raise ImportError(
                "SerpAPI package not installed. Install with: pip install google-search-results"
            )
        except (KeyError, ValueError, RuntimeError) as e:
            raise RuntimeError(f"Web search failed: {str(e)}")

    def _extract_text_from_url(self, url: str, max_length: int = None) -> str:
        """Extract clean text content from a URL.

        Args:
            url: URL to extract text from
            max_length: Maximum length of extracted text

        Returns:
            Extracted text content or error message

        Raises:
            ValueError: If URL is invalid or uses a dangerous protocol
        """
        if max_length is None:
            max_length = self.DEFAULT_MAX_URL_LENGTH

        # Validate URL scheme to prevent dangerous protocols
        try:
            parsed = urlparse(url)
            allowed_schemes = {"http", "https"}
            if parsed.scheme not in allowed_schemes:
                raise ValueError(
                    f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed."
                )
            if not parsed.netloc:
                raise ValueError(f"Invalid URL: missing network location")
        except ValueError as e:
            return f"Error: Invalid URL '{url}': {str(e)}"
        except (AttributeError, TypeError) as e:
            return f"Error: Invalid URL format '{url}': {str(e)}"

        try:
            response = requests.get(
                url,
                timeout=self.DEFAULT_URL_EXTRACTION_TIMEOUT_SECONDS,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            response.raise_for_status()

            # Try trafilatura first (clean extraction)
            try:
                import trafilatura

                text = trafilatura.extract(response.text)
                if text:
                    text = " ".join(text.split())
                    if len(text) > max_length:
                        text = text[:max_length] + "..."
                    return text
            except ImportError:
                pass

            # Fallback to BeautifulSoup
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script, style, and other non-content elements
                for element in soup(
                    ["script", "style", "nav", "header", "footer", "aside", "iframe"]
                ):
                    element.decompose()

                # Get text from main content areas
                main_content = (
                    soup.find("main") or soup.find("article") or soup.find("body")
                )
                if main_content:
                    text = main_content.get_text(separator=" ", strip=True)
                else:
                    text = soup.get_text(separator=" ", strip=True)

                text = " ".join(text.split())
                if len(text) > max_length:
                    text = text[:max_length] + "..."
                return text
            except ImportError:
                pass

            # Final fallback
            return (
                response.text[:max_length]
                if len(response.text) > max_length
                else response.text
            )

        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            return f"Error extracting text from {url}: {str(e)}"

    def _format_search_context(
        self,
        search_results: List[Dict[str, Any]],
        extract_full_text: bool = True,
        max_urls: int = None,
    ) -> str:
        """Format search results into context string for the LLM."""
        if max_urls is None:
            max_urls = self.DEFAULT_MAX_URLS
        if not search_results:
            return ""

        context_parts = ["Web Search Results:"]

        if extract_full_text:
            # Extract URLs and get full text content
            urls = [
                result.get("link", "")
                for result in search_results[:max_urls]
                if result.get("link")
            ]
            extracted_texts = []
            for url in urls:
                text = self._extract_text_from_url(url)
                if text:
                    extracted_texts.append(f"Source: {url}\n{text}\n\n")

            if extracted_texts:
                context_parts.append("\n=== Full Article Content ===\n")
                context_parts.extend(extracted_texts)
                context_parts.append("\n=== Search Result Summaries ===\n")

        # Add search result summaries (title + snippet)
        for i, result in enumerate(
            search_results[: self.DEFAULT_NUM_SEARCH_RESULTS], 1
        ):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No snippet")
            link = result.get("link", "")
            context_parts.append(f"\n{i}. {title}\n   {snippet}\n   Source: {link}")

        return "\n".join(context_parts)

    def _generate_with_web_search(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]],
        hf_config: HuggingFaceConfig,
    ) -> ModelResponse:
        """Generate response with web search context."""
        # Check if web search is enabled
        use_web_search = (
            hf_config.use_web_search if hf_config.use_web_search is not None else True
        )

        # Extract user query from messages
        user_query = ""
        for msg in messages:
            if msg.role == "user":
                user_query = msg.content
                break

        # Perform web search if enabled
        search_context = ""
        sources = []
        if use_web_search and user_query:
            try:
                search_results = self._perform_web_search(
                    user_query,
                    num_results=hf_config.num_search_results
                    or self.DEFAULT_NUM_SEARCH_RESULTS,
                )

                if search_results:
                    search_context = self._format_search_context(
                        search_results,
                        extract_full_text=hf_config.extract_full_text or True,
                        max_urls=hf_config.max_urls or self.DEFAULT_MAX_URLS,
                    )

                    # Extract sources for metadata
                    sources = [
                        {
                            "title": result.get("title", ""),
                            "url": result.get("link", ""),
                            "snippet": result.get("snippet", ""),
                        }
                        for result in search_results[: self.DEFAULT_NUM_SEARCH_RESULTS]
                    ]
            except (RuntimeError, ValueError, KeyError) as e:
                # If web search fails, continue without it
                search_context = f"Note: Web search failed: {str(e)}"

        # Prepare messages with search context
        reasoning_level = hf_config.reasoning_level or "medium"
        system_content = f"Reasoning: {reasoning_level}\n\nYou are a helpful assistant"
        if search_context:
            system_content += " with access to web search results. Use the provided context to answer questions accurately."
        else:
            system_content += "."

        # Build messages with search context
        formatted_messages = []
        has_system = False
        for msg in messages:
            if msg.role == "system":
                has_system = True
                formatted_messages.append(
                    {"role": "system", "content": f"{msg.content}\n\n{system_content}"}
                )
            elif msg.role == "user":
                content = msg.content
                if search_context:
                    content = f"{search_context}\n\nQuestion: {content}"
                formatted_messages.append({"role": "user", "content": content})
            else:
                formatted_messages.append(self._map_messages([msg])[0])

        # Add system message if not present
        if not has_system:
            formatted_messages.insert(0, {"role": "system", "content": system_content})

        # Build metadata
        metadata = {}
        if sources:
            metadata["sources"] = sources
            metadata["web_search"] = {
                "enabled": use_web_search,
                "num_results": len(sources),
            }

        response = self._make_api_request(formatted_messages, config)
        if metadata:
            response.grounding_metadata = metadata
        return response

    def _build_api_url(self) -> str:
        """Build the API URL for HuggingFace endpoint."""
        base_url = self.endpoint_url.rstrip("/")
        if base_url.endswith("/v1") or base_url.endswith("/v1/"):
            return f"{base_url.rstrip('/')}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self, messages: List[Dict[str, Any]], config: ModelConfig
    ) -> Dict[str, Any]:
        """Build request payload."""
        payload = {
            "model": config.model or "openai/gpt-oss-20b",
            "messages": messages,
        }
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        return payload

    def _make_api_request(
        self, messages: List[Dict[str, Any]], config: ModelConfig
    ) -> ModelResponse:
        """Make API request to HuggingFace endpoint."""
        api_url = self._build_api_url()
        headers = self._build_headers()
        payload = self._build_payload(messages, config)

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=self.DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()

            # Parse response
            content = ""
            if isinstance(result, dict) and "choices" in result:
                choices = result.get("choices", [])
                if choices:
                    choice = choices[0]
                    if "message" in choice:
                        content = choice["message"].get("content", "")
                    elif "text" in choice:
                        content = choice["text"]

            return ModelResponse(
                content=content or str(result), tool_calls=[], raw=result
            )
        except requests.exceptions.RequestException as e:
            return ModelResponse(
                content=f"Error generating response: {str(e)}",
                tool_calls=[],
                raw={"error": str(e)},
            )

    def generate(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> ModelResponse:
        """Generate response using HuggingFace Inference Endpoint.

        If web search tools are specified, performs web search and augments the prompt.
        """
        # Extract HuggingFace-specific config
        hf_config = HuggingFaceConfig()
        if config.provider_kwargs:
            for key, value in config.provider_kwargs.items():
                if hasattr(hf_config, key):
                    setattr(hf_config, key, value)

        # Check if web search is requested
        has_web_search = False
        if tools:
            for tool in tools:
                if (
                    tool.provider == "huggingface"
                    and tool.provider_type == "web_search"
                ):
                    has_web_search = True
                    # Override web search config from tool config
                    if tool.provider_config:
                        if "use_web_search" in tool.provider_config:
                            hf_config.use_web_search = tool.provider_config[
                                "use_web_search"
                            ]
                        if "extract_full_text" in tool.provider_config:
                            hf_config.extract_full_text = tool.provider_config[
                                "extract_full_text"
                            ]
                        if "max_urls" in tool.provider_config:
                            hf_config.max_urls = tool.provider_config["max_urls"]
                        if "num_search_results" in tool.provider_config:
                            hf_config.num_search_results = tool.provider_config[
                                "num_search_results"
                            ]
                    break

        # If web search is enabled, use web search generation
        if has_web_search or hf_config.use_web_search:
            return self._generate_with_web_search(messages, config, tools, hf_config)

        # Standard generation without web search
        formatted_messages = self._map_messages(messages)

        # Ensure system message exists
        has_system = any(msg.get("role") == "system" for msg in formatted_messages)
        if not has_system:
            reasoning_level = hf_config.reasoning_level or "medium"
            formatted_messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"Reasoning: {reasoning_level}\n\nYou are a helpful assistant.",
                },
            )

        return self._make_api_request(formatted_messages, config)
