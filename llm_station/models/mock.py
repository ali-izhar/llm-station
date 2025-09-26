#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .base import ModelConfig, ProviderAdapter
from ..schemas.messages import Message, ModelResponse, ToolCall
from ..schemas.tooling import ToolSpec


class MockProvider(ProviderAdapter):
    """A simple offline provider for development/testing the agent loop.

    Behavior:
    - Echoes the last user message as assistant content
    - If the last user message matches a simple pattern like: call <tool>({"k":"v"})
      it returns a tool_call with that name and parsed JSON args.
    """

    name = "mock"

    def supports_tools(self) -> bool:
        return True

    def prepare_tools(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def generate(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> ModelResponse:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        content = last_user.content if last_user else ""

        # Look for naive tool-call pattern: call toolName({...})
        tool_calls: List[ToolCall] = []
        if last_user:
            m = re.search(r"call\s+([a-zA-Z0-9_\-]+)\s*\((\{.*\})\)", last_user.content)
            if m:
                name = m.group(1)
                try:
                    args = json.loads(m.group(2))
                except Exception:
                    args = {"_raw": m.group(2)}
                tool_calls.append(ToolCall(id="tc_1", name=name, arguments=args))

        return ModelResponse(content=content, tool_calls=tool_calls)
