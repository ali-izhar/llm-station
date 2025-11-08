#!/usr/bin/env python3
"""Agent Logging System - Clean, Pythonic Implementation"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TextIO, Callable
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Standard logging levels."""

    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    DEBUG = "debug"


class LogFormat(Enum):
    """Log output formats."""

    CONSOLE = "console"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class LogEntry:
    """Single log entry for agent interactions."""

    timestamp: str
    step: int
    action: str
    details: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolCallLog:
    """Log entry for tool call execution."""

    tool_name: str
    tool_call_id: str
    inputs: Dict[str, Any]
    outputs: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


@dataclass
class AgentSessionLog:
    """Complete log of an agent session."""

    session_id: str
    start_time: str
    provider: str
    model: str
    system_prompt: Optional[str]
    input_query: str
    tools_requested: List[str]
    steps: List[LogEntry]
    tool_calls: List[ToolCallLog]
    final_result: str
    total_execution_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class AgentLogger:
    """Comprehensive agent interaction logger."""

    # Formatting constants
    _ICONS = {
        "tool_selection": "🔧",
        "tool_execution": "⚙️",
        "provider_api_call": "🌐",
        "provider_tool_execution": "🛠️",
        "response_parsing": "📖",
        "error_handling": "❌",
    }

    _COLORS = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        format: LogFormat = LogFormat.CONSOLE,
        enabled: bool = True,
        session_id: Optional[str] = None,
        log_file: Optional[TextIO] = None,
    ):
        self.level = level
        self.format = format
        self.enabled = enabled
        self.session_id = session_id or f"session_{int(time.time())}"
        self.log_file = log_file
        self.current_session: Optional[AgentSessionLog] = None
        self.step_counter = 0
        self.start_time = 0.0

    def start_session(
        self,
        provider: str,
        model: str,
        input_query: str,
        tools_requested: List[str],
        system_prompt: Optional[str] = None,
    ) -> None:
        """Start a new agent session."""
        if not self.enabled:
            return

        self.start_time = time.time()
        self.step_counter = 0

        self.current_session = AgentSessionLog(
            session_id=self.session_id,
            start_time=datetime.now().isoformat(),
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            input_query=input_query,
            tools_requested=tools_requested,
            steps=[],
            tool_calls=[],
            final_result="",
            total_execution_time_ms=0.0,
        )

        self._write("AGENT SESSION STARTED", bold=True, color="blue")
        self._write_field("Session ID", self.session_id)
        self._write_field("Provider", provider)
        self._write_field("Model", model)
        if system_prompt:
            self._write_field("System", system_prompt)
        self._write_field("Query", input_query)
        if tools_requested:
            self._write_field("Tools", ", ".join(tools_requested))
        self._write("=" * 80, color="blue")

    def log_step(
        self,
        action: str,
        details: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a step in the agent process."""
        if not self.enabled or not self.current_session:
            return

        self.step_counter += 1
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            step=self.step_counter,
            action=action,
            details=details,
            metadata=metadata,
        )

        self.current_session.steps.append(entry)
        self._format_step(entry)

    def log_tool_call(
        self,
        tool_name: str,
        tool_call_id: str,
        inputs: Dict[str, Any],
        outputs: Optional[str] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
    ) -> None:
        """Log a tool call execution."""
        if not self.enabled or not self.current_session:
            return

        tool_log = ToolCallLog(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            inputs=inputs,
            outputs=outputs,
            error=error,
            execution_time_ms=execution_time_ms,
        )

        self.current_session.tool_calls.append(tool_log)
        self._format_tool_call(tool_log)

    def log_provider_call(
        self,
        api_type: str,
        request_data: Dict[str, Any],
        response_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log provider API calls."""
        if not self.enabled:
            return

        details = {
            "api_type": api_type,
            "model": request_data.get("model"),
            "tools_count": len(request_data.get("tools", [])),
            "has_error": error is not None,
        }

        if self.level == LogLevel.DEBUG:
            details["request"] = request_data
            if response_data:
                details["response"] = response_data

        metadata = {"error": error} if error else None
        self.log_step("provider_api_call", details, metadata)

    def end_session(
        self, final_result: str, metadata: Optional[Dict[str, Any]] = None
    ) -> AgentSessionLog:
        """End the current session and return the complete log."""
        if not self.enabled or not self.current_session:
            return AgentSessionLog(
                session_id="",
                start_time="",
                provider="",
                model="",
                system_prompt=None,
                input_query="",
                tools_requested=[],
                steps=[],
                tool_calls=[],
                final_result="",
                total_execution_time_ms=0.0,
            )

        self.current_session.final_result = final_result
        self.current_session.total_execution_time_ms = (
            time.time() - self.start_time
        ) * 1000
        self.current_session.metadata = metadata

        self._write("✅ SESSION COMPLETED", bold=True, color="green")
        self._write_field("Final Result", final_result)
        self._write("Session Summary", bold=True)

        session = self.current_session
        self._write_field("Total Time", f"{session.total_execution_time_ms:.1f}ms")
        self._write_field("Steps", str(len(session.steps)))

        local_tool_calls = len(session.tool_calls)
        provider_tool_executions = len(
            [step for step in session.steps if step.action == "provider_tool_execution"]
        )
        self._write_field("Local Tool Calls", str(local_tool_calls))
        self._write_field("Provider Tool Executions", str(provider_tool_executions))
        self._write_field(
            "Total Tool Usage", str(local_tool_calls + provider_tool_executions)
        )

        if metadata:
            self._write_field("Metadata", str(list(metadata.keys())))

        self._write("=" * 80, color="blue")

        session = self.current_session
        self.current_session = None
        return session

    def _write(
        self, text: str, bold: bool = False, color: Optional[str] = None
    ) -> None:
        """Unified write method for console and file."""
        if self.format == LogFormat.JSON:
            return  # JSON handled separately

        formatted = self._format_text(text, bold, color)
        print(formatted)

        if self.log_file:
            self.log_file.write(self._strip_colors(formatted) + "\n")
            self.log_file.flush()

    def _write_field(self, label: str, value: str) -> None:
        """Write a labeled field."""
        if self.format == LogFormat.CONSOLE:
            self._write(f"{label}: {value}", color="cyan")
        else:
            self._write(f"{label}: {value}")

    def _format_text(
        self, text: str, bold: bool = False, color: Optional[str] = None
    ) -> str:
        """Format text with colors for console output."""
        if self.format != LogFormat.CONSOLE:
            return text

        parts = []
        if bold:
            parts.append(self._COLORS["bold"])
        if color:
            parts.append(self._COLORS.get(color, ""))
        parts.append(text)
        parts.append(self._COLORS["end"])
        return "".join(parts)

    def _strip_colors(self, text: str) -> str:
        """Remove ANSI color codes."""
        import re

        return re.sub(r"\033\[[0-9;]*m", "", text)

    def _format_step(self, entry: LogEntry) -> None:
        """Format and output a step entry."""
        if self.format == LogFormat.JSON:
            print(json.dumps(asdict(entry), indent=2))
            return

        icon = self._ICONS.get(entry.action, "📝")
        timestamp = entry.timestamp.split("T")[1][:8]  # HH:MM:SS
        action_title = entry.action.replace("_", " ").title()

        self._write(
            f"\n[{timestamp}] Step {entry.step}: {icon} {action_title}", color="yellow"
        )

        # Format action-specific details
        self._format_step_details(entry)

        # Debug level shows all details
        if self.level == LogLevel.DEBUG:
            for key, value in entry.details.items():
                if isinstance(value, dict) and len(str(value)) > 100:
                    self._write(f"  {key}: {type(value).__name__} ({len(value)} items)")
                else:
                    self._write(f"  {key}: {value}")

    def _format_step_details(self, entry: LogEntry) -> None:
        """Format details based on action type."""
        action = entry.action
        details = entry.details

        if action == "tool_selection":
            tools = details.get("selected_tools", [])
            self._write(f"  Selected tools: {', '.join(tools)}", color="purple")
        elif action == "tool_execution":
            self._write(f"  Tool: {details.get('tool_name')}", color="purple")
            self._write(f"  Status: {details.get('status', 'unknown')}", color="purple")
        elif action == "provider_api_call":
            self._write(f"  API: {details.get('api_type')}", color="purple")
            self._write(f"  Model: {details.get('model')}", color="purple")
            if details.get("tools_count", 0) > 0:
                self._write(
                    f"  Tools: {details['tools_count']} tools attached", color="purple"
                )
        elif action == "provider_tool_execution":
            tools_executed = details.get("tools_executed", [])
            self._write(
                f"  Tools Executed: {', '.join(tools_executed)}", color="purple"
            )
            metadata_types = details.get("metadata_types", [])
            if metadata_types:
                self._write(
                    f"  Metadata Generated: {', '.join(metadata_types)}", color="purple"
                )

    def _format_tool_call(self, tool_log: ToolCallLog) -> None:
        """Format and output a tool call."""
        status_icon = "✅" if not tool_log.error else "❌"
        exec_time = (
            f" ({tool_log.execution_time_ms:.1f}ms)"
            if tool_log.execution_time_ms
            else ""
        )

        self._write(
            f"\n  🔨 TOOL CALL: {tool_log.tool_name} {status_icon}{exec_time}",
            color="green",
        )
        self._write(f"    ID: {tool_log.tool_call_id}", color="cyan")

        if tool_log.inputs:
            self._write("    Inputs:", color="cyan")
            for key, value in tool_log.inputs.items():
                self._write(f"      {key}: {value}")

        if tool_log.outputs:
            self._write("    Output:", color="cyan")
            self._write(f"      {tool_log.outputs}")

        if tool_log.error:
            self._write(f"    Error: {tool_log.error}", color="red")

    def export_session(self, format: LogFormat = LogFormat.JSON) -> str:
        """Export current session in specified format."""
        if not self.current_session:
            return ""

        if format == LogFormat.JSON:
            return json.dumps(asdict(self.current_session), indent=2)
        elif format == LogFormat.MARKDOWN:
            return self._export_markdown()
        return str(self.current_session)

    def _export_markdown(self) -> str:
        """Export session as markdown documentation."""
        if not self.current_session:
            return ""

        s = self.current_session
        md = [
            "# Agent Session Report",
            f"**Session ID:** {s.session_id}",
            f"**Provider:** {s.provider}",
            f"**Model:** {s.model}",
            f"**Start Time:** {s.start_time}",
            f"**Duration:** {s.total_execution_time_ms:.1f}ms",
            "",
            "## Input Query",
            f"```\n{s.input_query}\n```",
        ]

        if s.system_prompt:
            md.extend(["", "## System Prompt", f"```\n{s.system_prompt}\n```"])

        if s.tools_requested:
            md.extend(
                ["", "## Tools Requested"] + [f"- {tool}" for tool in s.tools_requested]
            )

        md.extend(["", "## Execution Steps"])
        for step in s.steps:
            md.append(f"### {step.step}. {step.action.replace('_', ' ').title()}")
            md.extend(
                [
                    f"**{k.replace('_', ' ').title()}:** {v}"
                    for k, v in step.details.items()
                ]
            )
            md.append("")

        if s.tool_calls:
            md.extend(["", "## Tool Calls"])
            for i, call in enumerate(s.tool_calls, 1):
                md.extend(
                    [
                        f"### {i}. {call.tool_name}",
                        f"**ID:** {call.tool_call_id}",
                    ]
                )
                if call.inputs:
                    md.append(
                        f"**Inputs:**\n```json\n{json.dumps(call.inputs, indent=2)}\n```"
                    )
                if call.outputs:
                    md.append(f"**Output:**\n```\n{call.outputs}\n```")
                if call.error:
                    md.append(f"**Error:** {call.error}")
                if call.execution_time_ms:
                    md.append(f"**Execution Time:** {call.execution_time_ms:.1f}ms")
                md.append("")

        md.extend(["", "## Final Result", f"```\n{s.final_result}\n```"])
        return "\n".join(md)


class AgentLoggerContext:
    """Context manager for agent logging."""

    def __init__(
        self,
        logger: AgentLogger,
        provider: str,
        model: str,
        input_query: str,
        tools_requested: List[str],
        system_prompt: Optional[str] = None,
    ):
        self.logger = logger
        self.provider = provider
        self.model = model
        self.input_query = input_query
        self.tools_requested = tools_requested
        self.system_prompt = system_prompt

    def __enter__(self) -> AgentLogger:
        """Start logging session."""
        self.logger.start_session(
            self.provider,
            self.model,
            self.input_query,
            self.tools_requested,
            self.system_prompt,
        )
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End logging session."""
        error_msg = (
            f"Session failed: {exc_type.__name__}: {exc_val}"
            if exc_type
            else "Session completed successfully"
        )
        self.logger.end_session(error_msg)


# Global logger instance
_global_logger: Optional[AgentLogger] = None


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    format: LogFormat = LogFormat.CONSOLE,
    enabled: bool = True,
) -> AgentLogger:
    """Setup global agent logger."""
    global _global_logger
    _global_logger = AgentLogger(level=level, format=format, enabled=enabled)
    return _global_logger


def get_logger() -> Optional[AgentLogger]:
    """Get the global logger instance."""
    return _global_logger


def log_step(
    action: str, details: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log a step using the global logger."""
    if _global_logger:
        _global_logger.log_step(action, details, metadata)


def log_tool_call(
    tool_name: str,
    tool_call_id: str,
    inputs: Dict[str, Any],
    outputs: Optional[str] = None,
    error: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
) -> None:
    """Log a tool call using the global logger."""
    if _global_logger:
        _global_logger.log_tool_call(
            tool_name, tool_call_id, inputs, outputs, error, execution_time_ms
        )


def log_provider_call(
    api_type: str,
    request_data: Dict[str, Any],
    response_data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Log a provider API call using the global logger."""
    if _global_logger:
        _global_logger.log_provider_call(api_type, request_data, response_data, error)


def log_agent_start(
    provider: str,
    model: str,
    query: str,
    tools: List[str],
    system_prompt: Optional[str] = None,
) -> None:
    """Log agent session start."""
    if _global_logger:
        _global_logger.start_session(provider, model, query, tools, system_prompt)


def log_agent_end(
    result: str, metadata: Optional[Dict[str, Any]] = None
) -> Optional[AgentSessionLog]:
    """Log agent session end."""
    if _global_logger:
        return _global_logger.end_session(result, metadata)
    return None
