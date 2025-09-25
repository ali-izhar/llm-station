#!/usr/bin/env python3
from __future__ import annotations

import urllib.request
from typing import Any

from .base import Tool, json_dumps
from ..schemas.tooling import ToolResult, ToolSpec


class FetchUrlTool(Tool):
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fetch_url",
            description="Fetch the content at a URL via HTTP GET.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The absolute URL"},
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default 10)",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            requires_network=True,
        )

    def run(self, *, tool_call_id: str, **kwargs: Any) -> ToolResult:
        self.validate_args(kwargs)
        url = kwargs.get("url")
        timeout = float(kwargs.get("timeout", 10))
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                body = resp.read().decode(charset, errors="replace")
                return ToolResult(
                    name="fetch_url",
                    content=json_dumps(
                        {
                            "url": url,
                            "status": resp.status,
                            "content": body[:20000],  # cap to protect prompt budget
                        }
                    ),
                    tool_call_id=tool_call_id,
                )
        except Exception as e:
            return ToolResult(
                name="fetch_url",
                content=f"Error fetching {url}: {e}",
                tool_call_id=tool_call_id,
                is_error=True,
            )
