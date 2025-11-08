"""Agent runtime for LLM interactions with tools."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .provider import get as get_provider
from .tools import get as get_tool, get_spec
from .types import (
    AssistantMessage,
    Message,
    ModelConfig,
    ModelResponse,
    SystemMessage,
    ToolCall,
    ToolResult,
    ToolSpec,
    UserMessage,
)

# Constants
PRIMARY_TOOLS = ["search", "code", "image", "json", "fetch", "url"]


class Agent:
    """Agent for interacting with LLM providers with tool support."""

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
        """Execute a tool call."""
        try:
            tool = get_tool(call.name)
            result = tool.run(tool_call_id=call.id, **(call.arguments or {}))
            return result
        except KeyError as e:
            # Tool not found - return error result
            return ToolResult(
                name=call.name,
                content=f"Tool '{call.name}' not found: {str(e)}",
                tool_call_id=call.id,
                is_error=True,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            # Tool execution errors - return error result
            return ToolResult(
                name=call.name,
                content=f"Tool execution error: {str(e)}",
                tool_call_id=call.id,
                is_error=True,
            )

    def _append_system(self, messages: List[Message]) -> List[Message]:
        """Append system prompt if set."""
        if self._system_prompt:
            return [SystemMessage(self._system_prompt), *messages]
        return messages

    def _call_provider(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]],
        config: ModelConfig,
    ) -> ModelResponse:
        """Call provider to generate response."""
        return self._provider.generate(messages=messages, config=config, tools=tools)

    def generate(
        self,
        prompt: str,
        tools: Optional[Sequence[Any]] = None,
        structured_schema: Optional[Dict[str, Any]] = None,
    ) -> AssistantMessage:
        """Generate response with optional tools.

        Args:
            prompt: User prompt
            tools: Optional list of tool names, ToolSpec instances, or config dicts
            structured_schema: Optional JSON schema for structured output

        Returns:
            AssistantMessage with content and optional tool calls
        """
        if not tools:
            return self._generate_simple(prompt, structured_schema)
        else:
            return self._generate_with_tools(prompt, tools, structured_schema)

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
        except (RuntimeError, ValueError, KeyError, AttributeError) as e:
            return AssistantMessage(content=f"Provider error: {str(e)}", tool_calls=[])
        except Exception as e:
            # Fallback for unexpected exceptions
            return AssistantMessage(content=f"Provider error: {str(e)}", tool_calls=[])

    def _generate_with_tools(
        self,
        prompt: str,
        tools: Optional[Sequence[Any]] = None,
        structured_schema: Optional[Dict[str, Any]] = None,
    ) -> AssistantMessage:
        """Generate with tools using smart routing."""
        msgs: List[Message] = [UserMessage(prompt)]
        msgs = self._append_system(msgs)

        # Process tool inputs using smart routing
        tool_specs: Optional[List[ToolSpec]] = None
        if tools:
            tool_specs = []
            for t in tools:
                if isinstance(t, ToolSpec):
                    tool_specs.append(t)
                elif isinstance(t, str):
                    # Get tool spec using smart routing
                    provider_preference = self.provider_name
                    try:
                        spec = get_spec(t, provider_preference=provider_preference)
                        tool_specs.append(spec)
                    except KeyError as e:
                        suggestions = ", ".join(PRIMARY_TOOLS)
                        raise KeyError(
                            f"Unknown tool: '{t}'. Available smart tools: {suggestions}"
                        ) from e
                elif isinstance(t, dict):
                    # Support tool configuration: {"name": "search", "provider_preference": "google"}
                    tool_name = t.get("name")
                    if not tool_name:
                        raise TypeError("Tool dict must have 'name' key")
                    tool_config = {k: v for k, v in t.items() if k != "name"}
                    provider_pref = tool_config.pop(
                        "provider_preference", self.provider_name
                    )
                    spec = get_spec(tool_name, provider_preference=provider_pref)
                    tool_specs.append(spec)
                else:
                    raise TypeError(
                        "tools entries must be tool names, ToolSpec instances, or configuration dicts"
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
                        tool_result = self._execute_tool(call)
                        assistant.content += (
                            f"\n\n[{call.name} result: {tool_result.content}]"
                        )
                    except KeyError:
                        # Tool doesn't exist locally (it's a server-side tool) - skip
                        continue
                    except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                        assistant.content += f"\n\n[{call.name} error: {str(e)}]"

            return assistant

        except (RuntimeError, ValueError, KeyError, AttributeError) as e:
            return AssistantMessage(content=f"Provider error: {str(e)}", tool_calls=[])
        except Exception as e:
            # Fallback for unexpected exceptions
            return AssistantMessage(content=f"Provider error: {str(e)}", tool_calls=[])
