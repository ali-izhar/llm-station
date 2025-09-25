#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .base import ModelConfig, ProviderAdapter
from ..schemas.messages import Message, ModelResponse, ToolCall
from ..schemas.tooling import ToolSpec


class AnthropicProvider(ProviderAdapter):
    """Adapter for Anthropic Claude models via Messages API.

    This is a request/response shaping scaffold with no network calls.
    """

    name = "anthropic"

    def supports_tools(self) -> bool:
        return True

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
                elif pt == "web_fetch_20250910":
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
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
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

    @staticmethod
    def _parse_response(payload: Dict[str, Any]) -> ModelResponse:
        # Anthropic returns content blocks; tool-use are blocks with type tool_use
        content_blocks = payload.get("content", [])
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for i, block in enumerate(content_blocks):
            if block.get("type") == "tool_use":
                name = block.get("name") or ""
                args = block.get("input") or {}
                tool_calls.append(
                    ToolCall(id=str(block.get("id") or i), name=name, arguments=args)
                )
            elif block.get("type") == "server_tool_use":
                # Server tools are executed by Anthropic; no local ToolCall emitted
                continue
            elif block.get("type") == "text":
                text_parts.append(block.get("text") or "")
        return ModelResponse(
            content="\n".join(text_parts).strip(), tool_calls=tool_calls, raw=payload
        )

    def generate(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> ModelResponse:
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
        if config.tool_choice is not None:
            # Handle tool_choice according to Anthropic API documentation
            if isinstance(config.tool_choice, str):
                # Simple string values: "auto", "any", "none"
                if config.tool_choice in {"auto", "any", "none"}:
                    request["tool_choice"] = {"type": config.tool_choice}
                else:
                    # Assume it's a tool name for "tool" type
                    request["tool_choice"] = {
                        "type": "tool",
                        "name": config.tool_choice,
                    }
            elif isinstance(config.tool_choice, dict):
                request["tool_choice"] = config.tool_choice
            else:
                raise ValueError(
                    f"Invalid tool_choice type: {type(config.tool_choice)}"
                )
        if config.response_json_schema:
            # Anthropic lacks first-class JSON schema constrain today; add hinting
            request["extra_prompt"] = "Respond with valid JSON per provided schema."

        # Real call to Anthropic SDK
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(**request)
            return self._parse_response(response.model_dump())
        except ImportError:
            raise RuntimeError(
                "Anthropic SDK not installed. Install with: pip install anthropic"
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic Messages API call failed: {str(e)}")
