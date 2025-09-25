#!/usr/bin/env python3
from __future__ import annotations

import abc
import json
from typing import Any, Dict

from ..schemas.tooling import ToolResult, ToolSpec


class Tool(abc.ABC):
    """Abstract tool with a normalized spec and execute method."""

    @abc.abstractmethod
    def spec(self) -> ToolSpec:
        raise NotImplementedError

    def validate_args(self, args: Dict[str, Any]) -> None:
        # Minimal validation: ensure required properties if declared
        schema = self.spec().input_schema or {}
        required = schema.get("required", [])
        for r in required:
            if r not in args:
                raise ValueError(f"Missing required argument: {r}")

    @abc.abstractmethod
    def run(self, *, tool_call_id: str, **kwargs: Any) -> ToolResult:
        raise NotImplementedError


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
