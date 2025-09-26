#!/usr/bin/env python3
"""Run agents with tools"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from ..models.base import ModelConfig
from ..models.registry import get_provider
from ..schemas.messages import (
    AssistantMessage,
    Message,
    ModelResponse,
    SystemMessage,
    ToolMessage,
    ToolCall,
    UserMessage,
)
from ..schemas.tooling import ToolResult, ToolSpec
from ..tools.registry import get_tool, get_tool_spec
from ..logging.agent_logger import (
    get_logger,
    log_step,
    log_tool_call,
    log_provider_call,
    LogLevel,
)


class Agent:
    def __init__(
        self,
        provider: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **provider_kwargs: Any,
    ) -> None:
        self.provider_name = provider
        self._provider = get_provider(provider, api_key=api_key, **provider_kwargs)
        self._base_config = ModelConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            provider_kwargs=provider_kwargs if provider_kwargs else None,
        )
        self._system_prompt = system_prompt

    def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with logging."""
        start_time = time.time()

        # Log tool call start
        log_step(
            "tool_execution",
            {"tool_name": call.name, "tool_call_id": call.id, "status": "starting"},
        )

        try:
            tool = get_tool(call.name)
            result = tool.run(tool_call_id=call.id, **(call.arguments or {}))

            execution_time = (time.time() - start_time) * 1000

            # Log successful tool call
            log_tool_call(
                tool_name=call.name,
                tool_call_id=call.id,
                inputs=call.arguments or {},
                outputs=result.content,
                execution_time_ms=execution_time,
            )

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = str(e)

            # Log failed tool call
            log_tool_call(
                tool_name=call.name,
                tool_call_id=call.id,
                inputs=call.arguments or {},
                error=error_msg,
                execution_time_ms=execution_time,
            )

            raise

    def _append_system(self, messages: List[Message]) -> List[Message]:
        if self._system_prompt:
            return [SystemMessage(self._system_prompt), *messages]
        return messages

    def _call_provider(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]],
        config: ModelConfig,
    ) -> ModelResponse:
        """Call provider with logging."""
        # Prepare request data for logging
        request_data = {
            "model": config.model,
            "provider": config.provider,
            "message_count": len(messages),
            "tools": [tool.name for tool in (tools or [])],
            "has_tools": bool(tools),
        }

        # Add config details for debug logging
        logger = get_logger()
        if logger and logger.level == LogLevel.DEBUG:
            request_data.update(
                {
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "provider_kwargs": config.provider_kwargs,
                }
            )

        # Determine API type based on provider
        api_type = "default"
        if hasattr(self._provider, "get_api_type"):
            api_type = self._provider.get_api_type(config, tools)

        try:
            # Log provider call start
            log_step(
                "provider_api_call",
                {
                    "action": "starting",
                    "api_type": api_type,
                    "model": config.model,
                    "tools_count": len(tools or []),
                },
            )

            # Make the actual provider call
            response = self._provider.generate(
                messages=messages, config=config, tools=tools
            )

            # Log successful provider call with tool usage info
            provider_tool_usage = []
            if tools:
                for tool in tools:
                    if tool.provider:
                        provider_tool_usage.append(f"{tool.provider}:{tool.name}")

            response_data = {
                "content_length": len(response.content),
                "tool_calls_count": len(response.tool_calls),
                "provider_tools_requested": provider_tool_usage,
                "has_metadata": bool(response.grounding_metadata),
            }

            # Add metadata summary for provider tools
            if response.grounding_metadata:
                metadata_summary = {}
                for key, value in response.grounding_metadata.items():
                    if isinstance(value, list):
                        metadata_summary[key] = f"{len(value)} items"
                    elif isinstance(value, dict):
                        metadata_summary[key] = f"dict with {len(value)} keys"
                    else:
                        metadata_summary[key] = str(type(value).__name__)
                response_data["metadata_summary"] = metadata_summary

                # Log provider tool execution detected
                log_step(
                    "provider_tool_execution",
                    {
                        "tools_executed": provider_tool_usage,
                        "metadata_types": list(response.grounding_metadata.keys()),
                        "execution_detected": True,
                    },
                )

            log_provider_call(
                api_type=api_type,
                request_data=request_data,
                response_data=response_data,
                error=None,
            )

            return response

        except Exception as e:
            # Log failed provider call
            log_provider_call(
                api_type=api_type,
                request_data=request_data,
                response_data=None,
                error=str(e),
            )
            raise

    def generate(
        self,
        prompt: str,
        tools: Optional[Sequence[Any]] = None,  # tool names, specs, or config dicts
        structured_schema: Optional[Dict[str, Any]] = None,
    ) -> AssistantMessage:
        """Generate response with optional smart tools.

        Returns the final assistant message with any tool results integrated.
        """
        # Start logging session
        logger = get_logger()
        if logger:
            tool_names = []
            if tools:
                for tool in tools:
                    if isinstance(tool, str):
                        tool_names.append(tool)
                    elif hasattr(tool, "name"):
                        tool_names.append(tool.name)
                    else:
                        tool_names.append(str(tool))

            logger.start_session(
                provider=self.provider_name,
                model=self._base_config.model,
                input_query=prompt,
                tools_requested=tool_names,
                system_prompt=self._system_prompt,
            )

        try:
            # If no tools requested, do simple generation
            if not tools:
                result = self._generate_simple(prompt, structured_schema)
            else:
                # Generate with smart tools
                result = self._generate_with_tools(prompt, tools, structured_schema)

            # End logging session
            if logger:
                logger.end_session(result.content, result.grounding_metadata)

            return result

        except Exception as e:
            # Log error and end session
            if logger:
                logger.log_step(
                    "error_handling", {"error": str(e), "error_type": type(e).__name__}
                )
                logger.end_session(f"Error: {str(e)}")
            raise

    def _generate_simple(
        self, prompt: str, structured_schema: Optional[Dict[str, Any]] = None
    ) -> AssistantMessage:
        """Simple generation without tools."""
        msgs: List[Message] = [UserMessage(prompt)]
        msgs = self._append_system(msgs)

        config_dict = {**self._base_config.__dict__}
        config_dict["response_json_schema"] = structured_schema
        config = ModelConfig(**config_dict)

        try:
            resp = self._call_provider(messages=msgs, tools=None, config=config)
            return AssistantMessage(
                content=resp.content,
                tool_calls=resp.tool_calls,
                grounding_metadata=resp.grounding_metadata,
            )
        except Exception as e:
            return AssistantMessage(content=f"Error: {str(e)}", tool_calls=[])

    def _generate_with_tools(
        self,
        prompt: str,
        tools: Optional[Sequence[Any]] = None,
        structured_schema: Optional[Dict[str, Any]] = None,
    ) -> AssistantMessage:
        """Tool calling implementation using smart tool routing."""

        msgs: List[Message] = [UserMessage(prompt)]
        msgs = self._append_system(msgs)

        # Process tool inputs using smart routing
        tool_specs: Optional[List[ToolSpec]] = None
        if tools:
            tool_specs = []
            tool_names = []
            for t in tools:
                if isinstance(t, ToolSpec):
                    tool_specs.append(t)
                    tool_names.append(t.name)
                elif isinstance(t, str):
                    # Get tool spec using smart routing
                    provider_preference = (
                        self.provider_name if hasattr(self, "provider_name") else None
                    )
                    try:
                        spec = get_tool_spec(t, provider_preference=provider_preference)
                        tool_specs.append(spec)
                        tool_names.append(spec.name)
                    except KeyError as e:
                        # Provide helpful error message with smart tool suggestions
                        primary_tools = [
                            "search",
                            "code",
                            "image",
                            "json",
                            "fetch",
                            "url",
                        ]
                        aliases = [
                            "websearch",
                            "python",
                            "draw",
                            "execute",
                            "compute",
                            "run",
                        ]
                        suggestions = ", ".join(primary_tools)
                        raise KeyError(
                            f"Unknown tool: '{t}'. Available smart tools: {suggestions}. "
                            f"Aliases: {', '.join(aliases[:4])}..."
                        ) from e
                elif isinstance(t, dict):
                    # Support tool configuration: {"name": "search", "provider_preference": "google"}
                    tool_name = t.get("name")
                    if not tool_name:
                        raise TypeError("Tool dict must have 'name' key")

                    tool_config = {k: v for k, v in t.items() if k != "name"}
                    spec = get_tool_spec(tool_name, **tool_config)
                    tool_specs.append(spec)
                    tool_names.append(spec.name)
                else:
                    raise TypeError(
                        "tools entries must be tool names, ToolSpec instances, or configuration dicts"
                    )

            # Log tool selection
            log_step(
                "tool_selection",
                {
                    "selected_tools": tool_names,
                    "tool_count": len(tool_specs),
                    "local_tools": [
                        spec.name for spec in tool_specs if not spec.provider
                    ],
                    "provider_tools": [
                        f"{spec.provider}:{spec.name}"
                        for spec in tool_specs
                        if spec.provider
                    ],
                },
            )

        # Create config
        config_dict = {**self._base_config.__dict__}
        config_dict["response_json_schema"] = structured_schema
        config = ModelConfig(**config_dict)

        # Execute tools with smart routing
        try:
            resp = self._call_provider(messages=msgs, tools=tool_specs, config=config)
            assistant = AssistantMessage(
                content=resp.content,
                tool_calls=resp.tool_calls,
                grounding_metadata=resp.grounding_metadata,
            )

            # Execute any local tools if requested by the model
            if resp.tool_calls:
                for call in resp.tool_calls:
                    try:
                        # Try to execute local tool
                        tool_result = self._execute_tool(call)
                        # Append tool output to content for simple integration
                        assistant.content += (
                            f"\n\n[{call.name} result: {tool_result.content}]"
                        )
                    except KeyError:
                        # Tool doesn't exist locally (it's a server-side tool) - skip
                        continue
                    except Exception as e:
                        # Tool execution failed - append error
                        assistant.content += f"\n\n[{call.name} error: {str(e)}]"

            # Return final response
            return assistant

        except Exception as e:
            # If provider call fails completely, return error message
            return AssistantMessage(content=f"Provider error: {str(e)}", tool_calls=[])
