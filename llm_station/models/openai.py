#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .base import ModelConfig, ProviderAdapter
from ..schemas.messages import Message, ModelResponse, ToolCall
from ..schemas.tooling import ToolSpec


@dataclass
class OpenAIConfig:
    """OpenAI-specific configuration parameters."""

    # API selection
    api: Optional[str] = None  # "responses" or "chat"
    # Chat Completions specific
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    # Responses API specific
    include: Optional[List[str]] = None  # e.g., ["web_search_call.action.sources"]
    reasoning: Optional[Dict[str, Any]] = (
        None  # e.g., {"effort": "low"|"medium"|"high"}
    )
    instructions: Optional[str] = None  # Custom instructions for Responses API


class OpenAIProvider(ProviderAdapter):
    """Adapter for OpenAI APIs (Chat Completions and Responses).

    Automatically selects the appropriate API based on tools:
    - Chat Completions API: Default for standard interactions
    - Responses API: Used for web search, code interpreter, and image generation tools
    """

    name = "openai"

    def supports_tools(self) -> bool:
        return True

    def prepare_tools(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        """Map normalized ToolSpec to OpenAI tools format for Chat/Responses APIs.

        Handles:
        - Web search tools (web_search, web_search_preview)
        - Code Interpreter tool (code_interpreter)
        - Image Generation tool (image_generation)
        - Standard function tools (local tools)
        """
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

            elif t.provider == "openai" and t.provider_type == "code_interpreter":
                # OpenAI Code Interpreter tool structure
                entry: Dict[str, Any] = {"type": "code_interpreter"}

                if t.provider_config:
                    # Container configuration (auto mode or explicit container ID)
                    if "container" in t.provider_config:
                        entry["container"] = t.provider_config["container"]

                    # Optional container name for creation
                    if "name" in t.provider_config:
                        entry["name"] = t.provider_config["name"]

                prepared.append(entry)

            elif t.provider == "openai" and t.provider_type == "image_generation":
                # OpenAI Image Generation tool structure
                entry: Dict[str, Any] = {"type": "image_generation"}

                if t.provider_config:
                    # Image generation parameters
                    for param in [
                        "size",
                        "quality",
                        "format",
                        "compression",
                        "background",
                        "partial_images",
                    ]:
                        if param in t.provider_config:
                            entry[param] = t.provider_config[param]

                    # Multi-turn editing parameters
                    if "previous_response_id" in t.provider_config:
                        entry["previous_response_id"] = t.provider_config[
                            "previous_response_id"
                        ]
                    if "image_generation_call_id" in t.provider_config:
                        entry["image_generation_call_id"] = t.provider_config[
                            "image_generation_call_id"
                        ]

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
        """Parse OpenAI API response from either Responses API or Chat Completions API.

        Handles:
        - Responses API: Array format with web_search_call and message items
        - Chat Completions: Standard choices format with tool calls
        - Citations and annotations from web search
        - Error handling for both APIs
        """
        # Handle OpenAI SDK response format
        # The model_dump() returns a dict with 'output' field containing the actual response array
        if isinstance(payload, dict) and "output" in payload:
            return OpenAIProvider._parse_responses_api(payload["output"])
        # Detect Responses API format (direct array)
        elif isinstance(payload, list):
            return OpenAIProvider._parse_responses_api(payload)
        # Other response formats
        elif isinstance(payload, dict) and ("text" in payload or "status" in payload):
            return OpenAIProvider._parse_responses_api(payload)
        else:
            return OpenAIProvider._parse_chat_completions_api(payload)

    @staticmethod
    def _parse_responses_api(payload: Any) -> ModelResponse:
        """Parse OpenAI Responses API response format.

        Handles:
        - Web search calls and citations
        - Code interpreter calls and file citations
        - Image generation calls and base64 results
        - Message content and annotations
        """
        # Initialize variables
        content_parts = []
        citations = []
        sources = []
        file_citations = []
        web_search_metadata = {}
        code_interpreter_metadata = {}
        image_generation_metadata = []

        # Handle array format (standard Responses API output)
        if isinstance(payload, list):
            for item in payload:
                item_type = item.get("type", "")

                # Web search call metadata
                if item_type == "web_search_call":
                    web_search_metadata = {
                        "id": item.get("id"),
                        "status": item.get("status"),
                        "action": item.get("action", {}),
                        "query": item.get("action", {}).get("query", ""),
                        "search_type": item.get("action", {}).get("type", ""),
                    }
                    # Extract sources if available
                    action = item.get("action", {})
                    if "sources" in action:
                        sources = action["sources"]

                # Code interpreter call metadata
                elif item_type == "code_interpreter_call":
                    code_interpreter_metadata = {
                        "id": item.get("id"),
                        "status": item.get("status"),
                        "container_id": item.get("container_id"),
                        "code": item.get("code"),
                        "output": item.get("output"),
                    }

                # Image generation call metadata and results
                elif item_type == "image_generation_call":
                    image_call = {
                        "id": item.get("id"),
                        "status": item.get("status"),
                        "revised_prompt": item.get("revised_prompt"),
                        "result": item.get("result"),
                        "partial_results": item.get("partial_results", []),
                    }
                    for param in [
                        "size",
                        "quality",
                        "format",
                        "compression",
                        "background",
                    ]:
                        if param in item:
                            image_call[param] = item[param]
                    image_generation_metadata.append(image_call)

                # Message content with text and citations
                elif item_type == "message" and "content" in item:
                    message_content = item.get("content", [])
                    if isinstance(message_content, list):
                        for content_item in message_content:
                            if (
                                isinstance(content_item, dict)
                                and content_item.get("type") == "output_text"
                            ):
                                text = content_item.get("text", "")
                                if text:
                                    content_parts.append(text)

                                # Extract annotations
                                annotations = content_item.get("annotations", [])
                                for annotation in annotations:
                                    annotation_type = annotation.get("type")

                                    if annotation_type == "url_citation":
                                        citations.append(
                                            {
                                                "type": "url_citation",
                                                "url": annotation.get("url"),
                                                "title": annotation.get("title"),
                                                "start_index": annotation.get(
                                                    "start_index"
                                                ),
                                                "end_index": annotation.get(
                                                    "end_index"
                                                ),
                                            }
                                        )

                                    elif annotation_type == "container_file_citation":
                                        file_citations.append(
                                            {
                                                "type": "container_file_citation",
                                                "file_id": annotation.get("file_id"),
                                                "filename": annotation.get("filename"),
                                                "container_id": annotation.get(
                                                    "container_id"
                                                ),
                                                "start_index": annotation.get(
                                                    "start_index"
                                                ),
                                                "end_index": annotation.get(
                                                    "end_index"
                                                ),
                                                "index": annotation.get("index"),
                                            }
                                        )

        # Handle single object format (alternative Responses API format)
        elif isinstance(payload, dict):
            if "output" in payload:
                content = str(payload["output"])
            elif "text" in payload:
                content = str(payload["text"])
            elif "output_text" in payload:
                content = payload["output_text"]

        # Assemble final content from parts
        final_content = "\n".join(content_parts).strip() if content_parts else ""

        # Handle single object format (alternative Responses API format)
        if not final_content and isinstance(payload, dict):
            if "output" in payload:
                final_content = str(payload["output"])
            elif "text" in payload:
                final_content = str(payload["text"])
            elif "output_text" in payload:
                final_content = payload["output_text"]

        # Final fallback: try extraction function
        if not final_content:
            final_content = OpenAIProvider._extract_text_from_raw_payload(payload)

        # Error handling
        if not final_content:
            if isinstance(payload, dict) and "error" in payload:
                final_content = f"API Error: {payload['error']}"
            else:
                final_content = "No content extracted from Responses API"

        # Build comprehensive metadata
        metadata = {}
        if web_search_metadata:
            metadata["web_search"] = web_search_metadata
        if code_interpreter_metadata:
            metadata["code_interpreter"] = code_interpreter_metadata
        if image_generation_metadata:
            metadata["image_generation"] = image_generation_metadata
        if citations:
            metadata["citations"] = citations
        if file_citations:
            metadata["file_citations"] = file_citations
        if sources:
            metadata["sources"] = sources

        # Responses API handles tools server-side, so no local tool_calls
        return ModelResponse(
            content=final_content,
            tool_calls=[],
            raw=payload,
            grounding_metadata=metadata if metadata else None,
        )

    @staticmethod
    def _extract_text_from_raw_payload(payload: Any) -> str:
        """Extract readable text content from raw OpenAI response payload."""
        if isinstance(payload, list):
            # Look for message items in the response array
            for item in payload:
                if item.get("type") == "message":
                    content_items = item.get("content", [])
                    if isinstance(content_items, list):
                        for content_item in content_items:
                            if (
                                isinstance(content_item, dict)
                                and content_item.get("type") == "output_text"
                            ):
                                text = content_item.get("text", "")
                                if text and text.strip():
                                    return text.strip()
        elif isinstance(payload, dict):
            # Handle single object format
            if "output_text" in payload:
                return str(payload["output_text"])
            elif "text" in payload:
                return str(payload["text"])

        return "Content extraction failed - unable to parse OpenAI response"

    @staticmethod
    def _parse_chat_completions_api(payload: Dict[str, Any]) -> ModelResponse:
        """Parse OpenAI Chat Completions API response format."""
        choices = payload.get("choices", [])
        if not choices:
            return ModelResponse(content="", tool_calls=[], raw=payload)

        msg = choices[0].get("message", {})
        content = msg.get("content") or ""
        raw_tool_calls = msg.get("tool_calls") or []

        # Parse tool calls
        tool_calls: List[ToolCall] = []
        for i, tc in enumerate(raw_tool_calls):
            fn = (tc or {}).get("function", {})
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments", "")}

            tool_calls.append(
                ToolCall(id=str(tc.get("id") or f"call_{i}"), name=name, arguments=args)
            )

        return ModelResponse(content=content, tool_calls=tool_calls, raw=payload)

    def generate(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> ModelResponse:
        """Generate response using appropriate OpenAI API.

        Args:
            messages: Conversation history
            config: Model configuration
            tools: Optional tools to make available

        Returns:
            ModelResponse with content, tool calls, and metadata
        """
        # Use the same logic as get_api_type
        api_type = self.get_api_type(config, tools)

        if api_type == "responses_api":
            return self._generate_responses_api(messages, config, tools)
        else:
            return self._generate_chat_api(messages, config, tools)

    def get_api_type(self, config: ModelConfig, tools: Optional[List[ToolSpec]]) -> str:
        """Get the API type that will be used for this request."""
        needs_responses_api = tools and any(
            t.provider == "openai"
            and t.provider_type
            in {
                "web_search",
                "web_search_preview",
                "code_interpreter",
                "image_generation",
            }
            for t in tools
        )

        return "responses_api" if needs_responses_api else "chat_completions"

    def _generate_responses_api(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]],
    ) -> ModelResponse:
        """Generate response using OpenAI Responses API."""
        instructions, user_input = self._map_messages_responses(messages)

        # Extract OpenAI-specific parameters from provider_kwargs
        openai_config = OpenAIConfig()
        if config.provider_kwargs:
            for key, value in config.provider_kwargs.items():
                if hasattr(openai_config, key):
                    setattr(openai_config, key, value)

        # Build request with all Responses API parameters
        request: Dict[str, Any] = {
            "model": config.model,
            "input": user_input or "",
        }

        # Add instructions (system prompt for Responses API)
        if instructions:
            request["instructions"] = instructions
        elif openai_config.instructions:
            request["instructions"] = openai_config.instructions

        # Add tools if provided
        if tools:
            request["tools"] = self.prepare_tools(tools)

        # Add Responses API specific parameters (model-dependent)
        if openai_config.tool_choice:
            request["tool_choice"] = openai_config.tool_choice
        if openai_config.include:
            request["include"] = list(openai_config.include)

        # Add reasoning parameter if provided
        if openai_config.reasoning:
            request["reasoning"] = dict(openai_config.reasoning)

        # Add temperature parameter
        if config.temperature is not None:
            request["temperature"] = config.temperature

        if config.max_tokens is not None:
            request["max_output_tokens"] = (
                config.max_tokens
            )  # Responses API uses max_output_tokens
        if config.response_json_schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": config.response_json_schema,
            }

        # Make API call
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            response = client.responses.create(**request)

            # Use model_dump() to get the proper dictionary structure
            response_data = response.model_dump()
            return self._parse_response(response_data)

        except ImportError:
            return ModelResponse(
                content="OpenAI SDK not installed. Install with: pip install openai",
                tool_calls=[],
                raw={"error": "sdk_not_installed"},
            )
        except AttributeError:
            return ModelResponse(
                content=f"OpenAI Responses API not available in current SDK version. "
                f"Upgrade OpenAI SDK or use Chat Completions API instead.",
                tool_calls=[],
                raw={"error": "responses_api_not_available"},
            )
        except Exception as e:
            return ModelResponse(
                content=f"OpenAI Responses API error: {str(e)}",
                tool_calls=[],
                raw={"error": "api_call_failed", "details": str(e)},
            )

    def _generate_chat_api(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]],
    ) -> ModelResponse:
        """Generate response using OpenAI Chat Completions API."""
        request: Dict[str, Any] = {
            "model": config.model,
            "messages": self._map_messages_chat(messages),
        }

        # Add standard Chat Completions parameters
        if config.temperature is not None:
            request["temperature"] = config.temperature
        if config.top_p is not None:
            request["top_p"] = config.top_p
        if config.max_tokens is not None:
            request["max_tokens"] = config.max_tokens
        if config.frequency_penalty is not None:
            request["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            request["presence_penalty"] = config.presence_penalty
        if config.stop is not None:
            request["stop"] = config.stop

        # Extract OpenAI-specific parameters from provider_kwargs
        openai_config = OpenAIConfig()
        if config.provider_kwargs:
            for key, value in config.provider_kwargs.items():
                if hasattr(openai_config, key):
                    setattr(openai_config, key, value)

        # Add OpenAI-specific Chat Completions parameters
        if openai_config.tool_choice is not None:
            request["tool_choice"] = openai_config.tool_choice
        if openai_config.logit_bias is not None:
            request["logit_bias"] = openai_config.logit_bias
        if openai_config.user is not None:
            request["user"] = openai_config.user
        if openai_config.seed is not None:
            request["seed"] = openai_config.seed

        # Add tools if provided
        if tools:
            request["tools"] = self.prepare_tools(tools)

        # JSON response format (basic hint for Chat Completions)
        if config.response_json_schema:
            request["response_format"] = {"type": "json_object"}

        # Make API call
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
