from __future__ import annotations

import json
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
        )
        self._system_prompt = system_prompt

    def _tool_specs_from_names(self, tool_names: Sequence[str]) -> List[ToolSpec]:
        specs: List[ToolSpec] = []
        for name in tool_names:
            t = get_tool(name)
            specs.append(t.spec())
        return specs

    def _execute_tool(self, call: ToolCall) -> ToolResult:
        tool = get_tool(call.name)
        return tool.run(tool_call_id=call.id, **(call.arguments or {}))

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
        return self._provider.generate(messages=messages, config=config, tools=tools)

    def _should_force_tool_use(
        self, prompt: str, tools: Optional[List[ToolSpec]]
    ) -> bool:
        """Determine if we should try to force tool usage based on prompt content."""
        if not tools:
            return False

        # Check for explicit tool requests in prompt
        force_keywords = [
            "use the",
            "call the",
            "invoke the",
            "execute the",
            "format as json",
            "format this",
            "calculate using",
            "search for",
            "fetch from",
            "analyze the",
        ]

        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in force_keywords)

    def generate(
        self,
        prompt: str,
        tools: Optional[Sequence[Any]] = None,  # names or ToolSpec instances
        structured_schema: Optional[Dict[str, Any]] = None,
        max_tool_rounds: int = 4,
    ) -> AssistantMessage:
        """Run a single agent turn with optional tool-calling loop.

        Returns the final assistant message (with any tool_calls it last produced).
        """
        # If no tools requested, do simple generation
        if not tools:
            return self._generate_simple(prompt, structured_schema)

        # Otherwise try the full tool calling flow
        return self._generate_with_fallback(
            prompt, tools, structured_schema, max_tool_rounds
        )

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

    def _generate_with_fallback(
        self,
        prompt: str,
        tools: Optional[Sequence[Any]] = None,
        structured_schema: Optional[Dict[str, Any]] = None,
        max_tool_rounds: int = 4,
    ) -> AssistantMessage:
        """Production-ready tool calling implementation with fallback handling."""

        msgs: List[Message] = [UserMessage(prompt)]
        msgs = self._append_system(msgs)

        # Normalize tool inputs - now supports both local and provider tools by name
        tool_specs: Optional[List[ToolSpec]] = None
        if tools:
            tool_specs = []
            for t in tools:
                if isinstance(t, ToolSpec):
                    tool_specs.append(t)
                elif isinstance(t, str):
                    # Use new unified tool spec getter
                    tool_specs.append(get_tool_spec(t))
                else:
                    raise TypeError(
                        "tools entries must be tool names or ToolSpec instances"
                    )

        # Create config
        config_dict = {**self._base_config.__dict__}
        config_dict["response_json_schema"] = structured_schema
        config = ModelConfig(**config_dict)

        # Simple approach: Try one call, if it works great, if not return basic response
        try:
            resp = self._call_provider(messages=msgs, tools=tool_specs, config=config)
            assistant = AssistantMessage(
                content=resp.content,
                tool_calls=resp.tool_calls,
                grounding_metadata=resp.grounding_metadata,
            )

            # If the model made tool calls and we have local tools, try to execute them
            if resp.tool_calls and tool_specs:
                local_tool_calls = []

                # Filter for local tools only
                for call in resp.tool_calls:
                    try:
                        get_tool(call.name)  # Check if local tool exists
                        local_tool_calls.append(call)
                    except KeyError:
                        continue  # Skip server-side tools

                # If we have local tools to execute, do a single round
                if local_tool_calls:
                    try:
                        # Add assistant message first
                        msgs.append(assistant)

                        # Execute tools
                        tool_results = []
                        for call in local_tool_calls:
                            try:
                                result = self._execute_tool(call)
                                tool_results.append(
                                    ToolMessage(
                                        content=result.content,
                                        tool_call_id=call.id,
                                        name=call.name,
                                    )
                                )
                            except Exception as e:
                                tool_results.append(
                                    ToolMessage(
                                        content=f"Tool error: {str(e)}",
                                        tool_call_id=call.id,
                                        name=call.name,
                                    )
                                )

                        # Add tool results and try one more call
                        msgs.extend(tool_results)

                        # Final call with tool results
                        final_resp = self._call_provider(
                            messages=msgs, tools=None, config=config
                        )
                        return AssistantMessage(
                            content=final_resp.content,
                            tool_calls=final_resp.tool_calls,
                            grounding_metadata=final_resp.grounding_metadata,
                        )

                    except Exception as e:
                        # If tool execution fails, return original response
                        return AssistantMessage(
                            content=assistant.content
                            + f"\n[Tool execution error: {str(e)}]",
                            tool_calls=assistant.tool_calls,
                            grounding_metadata=assistant.grounding_metadata,
                        )

            # Return assistant response (either no tool calls or server-side tools)
            return assistant

        except Exception as e:
            # If provider call fails completely, return error message
            return AssistantMessage(content=f"Provider error: {str(e)}", tool_calls=[])
