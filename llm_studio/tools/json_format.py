#!/usr/bin/env python3
from __future__ import annotations
from typing import Any

from .base import Tool, json_dumps
from ..schemas.tooling import ToolResult, ToolSpec


class JsonFormatTool(Tool):
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="json_format",
            description="Return a minified JSON string for the given object.",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "description": "Arbitrary JSON-serializable object",
                    }
                },
                "required": ["data"],
                "additionalProperties": False,
            },
        )

    def run(self, *, tool_call_id: str, **kwargs: Any) -> ToolResult:
        self.validate_args(kwargs)
        try:
            content = json_dumps(kwargs.get("data"))
        except TypeError as e:
            return ToolResult(
                name="json_format",
                content=f"Serialization error: {e}",
                tool_call_id=tool_call_id,
                is_error=True,
            )
        return ToolResult(
            name="json_format", content=content, tool_call_id=tool_call_id
        )
