from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .base import ModelConfig, ProviderAdapter
from ..schemas.messages import AssistantMessage, Message, ModelResponse, ToolCall
from ..schemas.tooling import ToolSpec


class OpenAIProvider(ProviderAdapter):
    """Adapter for OpenAI APIs (Chat Completions and Responses).

    This adapter only shapes requests and expected outputs. Actual network I/O
    is intentionally omitted in this scaffold. Replace the "_request" method
    with calls to the OpenAI SDK or HTTP client as needed.
    """

    name = "openai"

    def supports_tools(self) -> bool:
        return True

    def prepare_tools(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        # Map normalized ToolSpec to OpenAI "tools" entries for both Chat and Responses APIs.
        prepared: List[Dict[str, Any]] = []
        for t in tools:
            if t.provider == "openai" and t.provider_type in {
                "web_search",
                "web_search_preview",
            }:
                # OpenAI web search tool structure
                entry: Dict[str, Any] = {"type": t.provider_type}

                if t.provider_config:
                    # Domain filtering: {"filters": {"allowed_domains": [...]}}
                    if "filters" in t.provider_config:
                        entry["filters"] = t.provider_config["filters"]

                    # User location: {"user_location": {"type": "approximate", "country": "US", ...}}
                    if "user_location" in t.provider_config:
                        entry["user_location"] = t.provider_config["user_location"]

                prepared.append(entry)
            else:
                # Standard function tool for local tools
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

    # -- Message mapping helpers --
    @staticmethod
    def _map_messages_chat(messages: List[Message]) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        for m in messages:
            obj: Dict[str, Any] = {"role": m.role, "content": m.content}
            if m.name:
                obj["name"] = m.name
            if m.role == "tool":
                # OpenAI expects tool role as: role="tool", name=<tool_name>, tool_call_id
                if m.tool_call_id:
                    obj["tool_call_id"] = m.tool_call_id
            mapped.append(obj)
        return mapped

    @staticmethod
    def _map_messages_responses(messages: List[Message]) -> Tuple[Optional[str], str]:
        # Responses API: use "instructions" for the system prompt and "input" for user text
        system: Optional[str] = None
        user_parts: List[str] = []
        for m in messages:
            if m.role == "system" and system is None:
                system = m.content
            elif m.role == "user":
                user_parts.append(m.content)
        return system, "\n".join(user_parts).strip()

    @staticmethod
    def _parse_response(payload: Dict[str, Any]) -> ModelResponse:
        # Handle both Responses API and Chat Completions API formats

        # Check if this is a Responses API response (comes as array of items)
        if (
            isinstance(payload, list)
            or "output" in payload
            or "text" in payload
            or "status" in payload
        ):
            content = ""

            # Handle array format (multiple response items)
            if isinstance(payload, list):
                for item in payload:
                    if item.get("type") == "message" and "content" in item:
                        # Extract text from message content
                        for content_item in item["content"]:
                            if content_item.get("type") == "output_text":
                                content += content_item.get("text", "")

            # Handle single object format
            elif "output" in payload and payload["output"]:
                content = str(payload["output"])
            elif "text" in payload and payload["text"]:
                content = str(payload["text"])
            elif "output_text" in payload:
                content = payload["output_text"]

            # If still no content, check for errors
            if not content:
                if (
                    isinstance(payload, dict)
                    and "error" in payload
                    and payload["error"]
                ):
                    content = f"API Error: {payload['error']}"
                else:
                    content = "No content returned from Responses API"

            # Responses API handles tools server-side, so no local tool_calls
            return ModelResponse(content=content, tool_calls=[], raw=payload)

        # Chat Completions API format
        choices = payload.get("choices", [])
        if not choices:
            return ModelResponse(content="", tool_calls=[], raw=payload)
        msg = choices[0].get("message", {})
        content = msg.get("content") or ""
        raw_tool_calls = msg.get("tool_calls") or []

        tool_calls: List[ToolCall] = []
        for i, tc in enumerate(raw_tool_calls):
            fn = (tc or {}).get("function", {})
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {"_raw": fn.get("arguments")}
            tool_calls.append(
                ToolCall(id=str(tc.get("id") or i), name=name, arguments=args)
            )

        return ModelResponse(content=content, tool_calls=tool_calls, raw=payload)

    def generate(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> ModelResponse:
        # Check for web search tools and model compatibility
        has_web_search = False
        if tools:
            for t in tools:
                if t.provider == "openai" and (
                    t.provider_type in {"web_search", "web_search_preview"}
                ):
                    has_web_search = True
                    break

        # Decide whether to use Responses API or Chat Completions
        use_responses = False
        if config.api == "responses":
            use_responses = True
        elif has_web_search:
            # Models that support web search via Responses API (expanded list)
            responses_api_models = {
                "gpt-5",
                "o4-mini",
                "o3-deep-research",
                "o4-mini-deep-research",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "gpt-4-1106-preview",
                "gpt-4-0125-preview",
            }

            # Models with built-in search via Chat Completions
            search_models = {"gpt-4o-search-preview", "gpt-4o-mini-search-preview"}

            if config.model in responses_api_models:
                use_responses = True
            elif config.model in search_models:
                # These models have built-in web search capabilities
                # Remove web search tools and let the model handle search automatically
                tools = [
                    t
                    for t in (tools or [])
                    if not (
                        t.provider == "openai"
                        and t.provider_type in {"web_search", "web_search_preview"}
                    )
                ]
                use_responses = False
            else:
                # Try Responses API for other models (more permissive)
                use_responses = True

        if use_responses:
            instructions, user_input = self._map_messages_responses(messages)
            request: Dict[str, Any] = {
                "model": config.model,
                "input": user_input or "",
            }
            if instructions:
                request["instructions"] = instructions
            if tools:
                request["tools"] = self.prepare_tools(tools)
            if config.tool_choice:
                request["tool_choice"] = config.tool_choice
            if config.include:
                request["include"] = list(config.include)
            if config.reasoning:
                request["reasoning"] = dict(config.reasoning)
            if config.temperature is not None:
                request["temperature"] = config.temperature
            if config.max_tokens is not None:
                # Responses API uses max_output_tokens
                request["max_output_tokens"] = config.max_tokens
            if config.response_json_schema:
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": config.response_json_schema,
                }
            # Real call to OpenAI Responses SDK
            try:
                import openai

                client = openai.OpenAI(api_key=self.api_key)

                response = client.responses.create(**request)
                return self._parse_response(response.model_dump())
            except ImportError:
                return ModelResponse(
                    content="OpenAI SDK not installed. Install with: pip install openai",
                    tool_calls=[],
                    raw={"error": "sdk_not_installed"},
                )
            except AttributeError:
                # Responses API not available in this SDK version
                return ModelResponse(
                    content=f"OpenAI Responses API not available in current SDK. "
                    f"For web search with model '{config.model}', use: "
                    f"1) Model 'gpt-4o-search-preview' (built-in search), or "
                    f"2) Upgrade OpenAI SDK for Responses API support.",
                    tool_calls=[],
                    raw={"error": "responses_api_not_available"},
                )
            except Exception as e:
                return ModelResponse(
                    content=f"OpenAI Responses API error: {str(e)}",
                    tool_calls=[],
                    raw={"error": "api_call_failed", "details": str(e)},
                )
        else:
            # Chat Completions shape
            request: Dict[str, Any] = {
                "model": config.model,
                "messages": self._map_messages_chat(messages),
            }
            if config.temperature is not None:
                request["temperature"] = config.temperature
            if config.top_p is not None:
                request["top_p"] = config.top_p
            if config.max_tokens is not None:
                request["max_tokens"] = config.max_tokens

            if tools:
                request["tools"] = self.prepare_tools(tools)

            if config.response_json_schema:
                # Prefer Responses API's json_schema if you wire it; for chat, hint
                request["response_format"] = {"type": "json_object"}

            # Real call to OpenAI Chat Completions SDK
            try:
                import openai

                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(**request)
                return self._parse_response(response.model_dump())
            except ImportError:
                raise RuntimeError(
                    "OpenAI SDK not installed. Install with: pip install openai"
                )
            except Exception as e:
                raise RuntimeError(f"OpenAI Chat Completions API call failed: {str(e)}")
