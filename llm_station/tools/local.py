"""Local tools that execute in the agent."""

from __future__ import annotations

import json
import gzip
import urllib.request
from typing import Any

from ..types import ToolResult, ToolSpec
from . import Tool


def json_dumps(obj: Any) -> str:
    """Minified JSON dumps."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class FetchUrlTool(Tool):
    """Fetch content from a URL."""

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
            },
            requires_network=True,
        )

    def run(self, *, tool_call_id: str, **kwargs: Any) -> ToolResult:
        self.validate_args(kwargs)
        url = kwargs.get("url")
        timeout = float(kwargs.get("timeout", 10))
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept-Encoding", "gzip, deflate")
            req.add_header("User-Agent", "Mozilla/5.0 (compatible; LLM-Station/1.0)")

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"

                # Handle gzip/deflate compression
                content_encoding = resp.headers.get("Content-Encoding", "").lower()
                body_bytes = resp.read()

                if content_encoding == "gzip":
                    body_bytes = gzip.decompress(body_bytes)
                elif content_encoding == "deflate":
                    import zlib

                    body_bytes = zlib.decompress(body_bytes)

                body = body_bytes.decode(charset, errors="replace")

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


class JsonFormatTool(Tool):
    """Format data as minified JSON."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="json_format",
            description="Converts structured data into a minified JSON string. ALWAYS provide the 'data' parameter with the value to format. Examples: json_format(data={'name': 'Alice', 'age': 30}) or json_format(data=[1, 2, 3]) or json_format(data='hello'). The 'data' parameter can be any JSON-serializable value: object/dict, array/list, string, number, boolean, or null.",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "description": "REQUIRED: The data to format as JSON. Can be any JSON-serializable value. Examples: {'name': 'Alice'} for an object, [1, 2, 3] for an array, 'text' for a string, 42 for a number, true for boolean, null for null value.",
                    }
                },
                "required": ["data"],
            },
        )

    def run(self, *, tool_call_id: str, **kwargs: Any) -> ToolResult:
        # Handle case where data might be passed directly or as a dict
        data = kwargs.get("data")

        # If data is not provided, check if there are other arguments that could be data
        if data is None:
            # Filter out tool_call_id and check if we have any other arguments
            other_args = {k: v for k, v in kwargs.items() if k != "tool_call_id"}
            if other_args:
                # If there are other arguments, use them as the data object
                data = other_args
            else:
                # No arguments provided at all
                return ToolResult(
                    name="json_format",
                    content="Error: Missing required argument 'data'. Please provide a 'data' parameter with the object to format as JSON. Example: {'data': {'name': 'Alice', 'age': 30}}",
                    tool_call_id=tool_call_id,
                    is_error=True,
                )

        try:
            content = json_dumps(data)
        except (TypeError, ValueError) as e:
            return ToolResult(
                name="json_format",
                content=f"Serialization error: {e}. The 'data' parameter must be JSON-serializable.",
                tool_call_id=tool_call_id,
                is_error=True,
            )
        return ToolResult(
            name="json_format", content=content, tool_call_id=tool_call_id
        )
