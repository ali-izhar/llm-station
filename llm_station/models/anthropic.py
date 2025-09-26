#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .base import ModelConfig, ProviderAdapter
from ..schemas.messages import Message, ModelResponse, ToolCall
from ..schemas.tooling import ToolSpec


class AnthropicProvider(ProviderAdapter):
    """Adapter for Anthropic Claude models via Messages API.

    Supports Claude models with server-side tool execution including web search
    and web content fetching. Uses the latest anthropic SDK for Messages API.

    Features token counting and rate limit management to stay under 10,000 tokens/minute.
    """

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self._token_usage = {"input_tokens": 0, "output_tokens": 0, "last_reset": 0.0}
        self._request_count = 0

    def supports_tools(self) -> bool:
        return True

    def _reset_token_usage_if_needed(self):
        """Reset token usage if more than 1 minute has passed."""
        import time

        current_time = time.time()
        if current_time - self._token_usage["last_reset"] > 60:  # 1 minute
            self._token_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "last_reset": current_time,
            }
            self._request_count = 0

    def _can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """Check if we can make request without exceeding 10K token/minute limit."""
        self._reset_token_usage_if_needed()
        # Stay under 9500 tokens per minute
        return (self._token_usage["input_tokens"] + estimated_tokens) < 9500

    def _add_token_usage(self, input_tokens: int, output_tokens: int):
        """Track token usage for rate limiting."""
        self._token_usage["input_tokens"] += input_tokens
        self._token_usage["output_tokens"] += output_tokens
        self._request_count += 1

    def count_tokens(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> int:
        """Count tokens before making request using Anthropic's token counting API."""
        shaped = self._map_messages(messages)
        request = {
            "model": config.model,
            "messages": shaped["messages"],
        }

        if shaped.get("system"):
            request["system"] = shaped["system"]
        if tools:
            request["tools"] = self.prepare_tools(tools)

        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.api_key,
                default_headers={"anthropic-version": "2023-06-01"},
            )

            count_response = client.messages.count_tokens(**request)
            return count_response.input_tokens

        except Exception:
            # Fallback estimation if token counting fails
            total_chars = sum(len(m.content) for m in messages)
            return total_chars // 4  # Rough estimation: 4 chars per token

    def prepare_tools(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        """Map normalized ToolSpec to Anthropic Messages API tool definitions.

        Supports:
          - Custom tools: {name, description, input_schema}
          - Server tools: e.g., web_search_20250305, web_fetch_20250910 via
            provider-native ToolSpec with provider="anthropic".
        """
        prepared: List[Dict[str, Any]] = []
        for t in tools:
            if t.provider == "anthropic" and t.provider_type:
                # Server tools pass through specific fields by type
                pt = t.provider_type
                cfg = t.provider_config or {}
                if pt == "web_search_20250305":
                    entry: Dict[str, Any] = {
                        "type": pt,
                        "name": "web_search",
                    }
                    for key in (
                        "allowed_domains",
                        "blocked_domains",
                        "user_location",
                        "max_uses",
                        "cache_control",
                    ):
                        if key in cfg and cfg[key] is not None:
                            entry[key] = cfg[key]
                    prepared.append(entry)
                elif pt in ["web_fetch_20250910", "web_fetch"]:
                    entry = {
                        "type": pt,
                        "name": "web_fetch",
                    }
                    for key in (
                        "allowed_domains",
                        "blocked_domains",
                        "citations",
                        "max_content_tokens",
                        "max_uses",
                        "cache_control",
                    ):
                        if key in cfg and cfg[key] is not None:
                            entry[key] = cfg[key]
                    prepared.append(entry)
                elif pt == "code_execution_20250825":
                    entry = {
                        "type": pt,
                        "name": "code_execution",
                    }
                    # Add container configuration if specified
                    for key in ("container_id", "max_execution_time"):
                        if key in cfg and cfg[key] is not None:
                            entry[key] = cfg[key]
                    prepared.append(entry)
                else:
                    # Unknown server tool type; pass through minimal shape
                    prepared.append(
                        {"type": pt, **({"name": t.name} if t.name else {})}
                    )
            else:
                # Custom (client) tools
                prepared.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    }
                )
        return prepared

    @staticmethod
    def _map_messages(messages: List[Message]) -> Dict[str, Any]:
        # Anthropic expects system as top-level, messages with role user/assistant
        # Tool results are provided as user messages with content blocks of type tool_result.
        from ..schemas.messages import ToolMessage

        system: Optional[str] = None
        msgs: List[Dict[str, Any]] = []
        tool_blocks_buffer: List[Dict[str, Any]] = []

        def flush_tool_blocks():
            nonlocal tool_blocks_buffer
            if tool_blocks_buffer:
                msgs.append({"role": "user", "content": tool_blocks_buffer})
                tool_blocks_buffer = []

        for m in messages:
            if m.role == "system":
                if system is None:
                    system = m.content
                continue
            if m.role == "tool":
                # Accumulate tool_result blocks to send in a single user message
                # Ensure tool_call_id exists and is valid
                if m.tool_call_id:
                    block = {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }
                    tool_blocks_buffer.append(block)
                continue
            # Flush any pending tool blocks before appending a normal message
            flush_tool_blocks()
            if m.role in ("user", "assistant"):
                # Allow simple string content; SDK accepts string or blocks
                msgs.append({"role": m.role, "content": m.content})
            else:
                # Unknown role; skip
                continue
        # Flush at end
        flush_tool_blocks()

        return {"system": system, "messages": msgs}

    def _parse_response(
        self, payload: Dict[str, Any], estimated_tokens: int = 0
    ) -> ModelResponse:
        """Parse Anthropic Messages API response with enhanced metadata extraction."""
        content_blocks = payload.get("content", [])
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        # Enhanced metadata extraction for server tools
        grounding_metadata = {}
        web_search_data = []
        web_fetch_data = []
        code_execution_data = []

        for i, block in enumerate(content_blocks):
            if block.get("type") == "tool_use":
                # Local tool calls
                name = block.get("name") or ""
                args = block.get("input") or {}
                tool_calls.append(
                    ToolCall(id=str(block.get("id") or i), name=name, arguments=args)
                )
            elif block.get("type") == "server_tool_use":
                # Server tools executed by Anthropic - extract metadata
                tool_name = block.get("name", "")
                tool_id = block.get("id", "")
                tool_input = block.get("input", {})

                if tool_name == "web_search":
                    web_search_data.append(
                        {
                            "id": tool_id,
                            "name": tool_name,
                            "query": tool_input.get("query", ""),
                            "status": "completed",
                            "type": "server_tool",
                        }
                    )
                elif tool_name == "web_fetch":
                    web_fetch_data.append(
                        {
                            "id": tool_id,
                            "name": tool_name,
                            "status": "completed",
                            "type": "server_tool",
                        }
                    )
                elif tool_name in [
                    "bash_code_execution",
                    "text_editor_code_execution",
                    "code_execution",
                ]:
                    # Code execution tools
                    execution_info = {
                        "id": tool_id,
                        "name": tool_name,
                        "type": "server_tool",
                        "status": "completed",
                    }

                    # Add specific execution details based on tool type
                    if tool_name == "bash_code_execution":
                        execution_info["command"] = tool_input.get("command", "")
                        execution_info["execution_type"] = "bash"
                    elif tool_name == "text_editor_code_execution":
                        execution_info["command"] = tool_input.get("command", "")
                        execution_info["path"] = tool_input.get("path", "")
                        execution_info["execution_type"] = "file_operation"

                    code_execution_data.append(execution_info)
                continue
            elif block.get("type") == "web_search_tool_result":
                # Web search results with citations and sources
                tool_use_id = block.get("tool_use_id", "")
                search_content = block.get("content", [])

                # Extract search results and sources
                sources = []
                search_results = []

                if isinstance(search_content, list):
                    for result in search_content:
                        if result.get("type") == "web_search_result":
                            url = result.get("url", "")
                            title = result.get("title", "")
                            page_age = result.get("page_age", "")

                            sources.append(url)
                            search_results.append(
                                {"url": url, "title": title, "page_age": page_age}
                            )
                elif (
                    isinstance(search_content, dict)
                    and search_content.get("type") == "web_search_tool_result_error"
                ):
                    # Handle search errors
                    error_code = search_content.get("error_code", "unknown")
                    grounding_metadata["search_error"] = {
                        "tool_use_id": tool_use_id,
                        "error_code": error_code,
                    }

                if sources:
                    grounding_metadata["sources"] = (
                        grounding_metadata.get("sources", []) + sources
                    )
                if search_results:
                    grounding_metadata["search_results"] = (
                        grounding_metadata.get("search_results", []) + search_results
                    )
                continue
            elif block.get("type") in [
                "bash_code_execution_tool_result",
                "text_editor_code_execution_tool_result",
            ]:
                # Code execution results
                tool_use_id = block.get("tool_use_id", "")
                result_content = block.get("content", {})
                result_type = result_content.get("type", "")

                execution_result = {
                    "tool_use_id": tool_use_id,
                    "result_type": result_type,
                }

                if result_type == "bash_code_execution_result":
                    execution_result.update(
                        {
                            "stdout": result_content.get("stdout", ""),
                            "stderr": result_content.get("stderr", ""),
                            "return_code": result_content.get("return_code", 0),
                            "execution_type": "bash",
                        }
                    )
                elif result_type == "text_editor_code_execution_result":
                    execution_result.update(
                        {
                            "file_type": result_content.get("file_type", ""),
                            "content": result_content.get("content", ""),
                            "num_lines": result_content.get("numLines", 0),
                            "is_file_update": result_content.get(
                                "is_file_update", False
                            ),
                            "execution_type": "file_operation",
                        }
                    )

                    # Add edit details if present
                    if "oldStart" in result_content:
                        execution_result["edit_details"] = {
                            "old_start": result_content.get("oldStart"),
                            "old_lines": result_content.get("oldLines"),
                            "new_start": result_content.get("newStart"),
                            "new_lines": result_content.get("newLines"),
                            "diff_lines": result_content.get("lines", []),
                        }
                elif result_type in [
                    "bash_code_execution_tool_result_error",
                    "text_editor_code_execution_tool_result_error",
                ]:
                    # Handle execution errors
                    execution_result.update(
                        {
                            "error_code": result_content.get("error_code", "unknown"),
                            "execution_type": "error",
                        }
                    )

                # Find matching execution data and add result
                for exec_data in code_execution_data:
                    if exec_data["id"] == tool_use_id:
                        exec_data["result"] = execution_result
                        break

                continue
            elif block.get("type") == "text":
                text_content = block.get("text") or ""
                text_parts.append(text_content)

                # Extract citations from text blocks
                citations = block.get("citations", [])
                if citations:
                    extracted_citations = []
                    for citation in citations:
                        if citation.get("type") == "web_search_result_location":
                            extracted_citations.append(
                                {
                                    "url": citation.get("url", ""),
                                    "title": citation.get("title", ""),
                                    "cited_text": citation.get("cited_text", ""),
                                    "encrypted_index": citation.get(
                                        "encrypted_index", ""
                                    ),
                                }
                            )

                    if extracted_citations:
                        if "citations" not in grounding_metadata:
                            grounding_metadata["citations"] = []
                        grounding_metadata["citations"].extend(extracted_citations)

        # Build metadata if server tools were used
        if web_search_data:
            grounding_metadata["web_search"] = web_search_data
        if web_fetch_data:
            grounding_metadata["web_fetch"] = web_fetch_data
        if code_execution_data:
            grounding_metadata["code_execution"] = code_execution_data

        # Extract usage metadata including server tool usage
        if "usage" in payload:
            usage = payload["usage"]
            usage_metadata = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }

            # Add cache usage if present
            if "cache_read_input_tokens" in usage:
                usage_metadata["cache_read_input_tokens"] = usage[
                    "cache_read_input_tokens"
                ]
            if "cache_creation_input_tokens" in usage:
                usage_metadata["cache_creation_input_tokens"] = usage[
                    "cache_creation_input_tokens"
                ]

            # Add server tool usage
            if "server_tool_use" in usage:
                server_usage = usage["server_tool_use"]
                if server_usage:  # Check if not None
                    usage_metadata["server_tool_use"] = server_usage

                    # Extract specific tool usage counts
                    if "web_search_requests" in server_usage:
                        usage_metadata["web_search_requests"] = server_usage[
                            "web_search_requests"
                        ]

            # Add session usage tracking
            usage_metadata["session_usage"] = {
                "session_input_tokens": self._token_usage["input_tokens"],
                "session_output_tokens": self._token_usage["output_tokens"],
                "session_requests": self._request_count,
                "estimated_tokens_before_request": estimated_tokens,
            }

            grounding_metadata["usage"] = usage_metadata

        # Extract response metadata
        response_metadata = {
            "id": payload.get("id"),
            "model": payload.get("model"),
            "stop_reason": payload.get("stop_reason"),
        }

        # Extract container information if present (for code execution)
        if "container" in payload:
            container_info = payload["container"]
            response_metadata["container"] = {
                "id": container_info.get("id"),
                "type": container_info.get("type", "unknown"),
            }

        grounding_metadata["response_info"] = response_metadata

        return ModelResponse(
            content="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw=payload,
            grounding_metadata=grounding_metadata if grounding_metadata else None,
        )

    def generate(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> ModelResponse:
        # Count tokens and check rate limits before making request
        estimated_tokens = self.count_tokens(messages, config, tools)
        self._last_estimated_tokens = estimated_tokens

        # Check if we can make the request
        if not self._can_make_request(estimated_tokens):
            import time

            wait_time = 60 - (time.time() - self._token_usage["last_reset"])
            if wait_time > 0:
                print(f"⏳ Rate limit protection: waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self._reset_token_usage_if_needed()

        shaped = self._map_messages(messages)
        request: Dict[str, Any] = {
            "model": config.model,
            "messages": shaped["messages"],
        }
        if shaped.get("system"):
            request["system"] = shaped["system"]
        if config.max_tokens is not None:
            request["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            request["temperature"] = config.temperature
        if tools:
            request["tools"] = self.prepare_tools(tools)
        # Handle tool_choice if specified via provider_kwargs
        tool_choice = None
        if config.provider_kwargs:
            tool_choice = config.provider_kwargs.get("tool_choice")

        if tool_choice is not None:
            # Handle tool_choice according to Anthropic API documentation
            if isinstance(tool_choice, str):
                # Simple string values: "auto", "any", "none"
                if tool_choice in {"auto", "any", "none"}:
                    request["tool_choice"] = {"type": tool_choice}
                else:
                    # Assume it's a tool name for "tool" type
                    request["tool_choice"] = {
                        "type": "tool",
                        "name": tool_choice,
                    }
            elif isinstance(tool_choice, dict):
                request["tool_choice"] = tool_choice
            else:
                raise ValueError(f"Invalid tool_choice type: {type(tool_choice)}")
        if config.response_json_schema:
            # Anthropic lacks first-class JSON schema constrain today; add hinting
            request["extra_prompt"] = "Respond with valid JSON per provided schema."

        # Real call to Anthropic SDK with proper configuration
        try:
            import anthropic

            # Determine if we need beta features
            has_code_execution = any(
                t.provider == "anthropic"
                and t.provider_type == "code_execution_20250825"
                for t in (tools or [])
            )

            # Create client with proper headers
            headers = {"anthropic-version": "2023-06-01"}
            if has_code_execution:
                headers["anthropic-beta"] = "code-execution-2025-08-25"

            client = anthropic.Anthropic(api_key=self.api_key, default_headers=headers)

            # Use beta client for code execution
            if has_code_execution:
                response = client.beta.messages.create(**request)
            else:
                response = client.messages.create(**request)

            # Convert response to dict format for parsing
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )

            # Track token usage from response
            if "usage" in response_dict:
                usage = response_dict["usage"]
                self._add_token_usage(
                    usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                )

            return self._parse_response(response_dict, estimated_tokens)

        except ImportError:
            raise RuntimeError(
                "Anthropic SDK not installed. Install with: pip install anthropic"
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic Messages API call failed: {str(e)}")
