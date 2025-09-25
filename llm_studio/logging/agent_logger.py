"""
Agent Logging System

Provides comprehensive logging of agent interactions including:
- Input queries and system prompts
- Tool selection and execution
- Tool call inputs and outputs
- Provider API calls and responses
- Final results and metadata

Supports multiple output formats and verbosity levels.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TextIO
from dataclasses import dataclass, asdict
from enum import Enum

from ..schemas.messages import Message, ToolCall
from ..schemas.tooling import ToolResult, ToolSpec


class LogLevel(Enum):
    """Standard logging levels following industry conventions."""

    ERROR = "error"  # Only errors and critical issues
    WARN = "warn"  # Warnings and errors
    INFO = "info"  # General information (default)
    DEBUG = "debug"  # Detailed debugging information


class LogFormat(Enum):
    """Log output formats."""

    CONSOLE = "console"  # Human-readable console output
    JSON = "json"  # Structured JSON logs
    MARKDOWN = "markdown"  # Markdown format for documentation


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

        # Current session state
        self.current_session: Optional[AgentSessionLog] = None
        self.step_counter = 0
        self.start_time = 0.0

        # Color codes for console output (only used when outputting to console)
        self.colors = {
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "purple": "\033[95m",
            "cyan": "\033[96m",
            "bold": "\033[1m",
            "end": "\033[0m",
        }

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

        self._log_session_start()

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
        self._log_step_output(entry)

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
        self._log_tool_call_output(tool_log)

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
            if self.level == LogLevel.DEBUG and response_data:
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

        self._log_session_end()

        session = self.current_session
        self.current_session = None
        return session

    def _log_session_start(self) -> None:
        """Log the start of an agent session."""
        if not self.current_session:
            return

        if self.format == LogFormat.CONSOLE:
            # Console output with colors
            self._log_session_start_console()

            # Clean file output without colors
            if self.log_file:
                self._log_session_start_file()

    def _log_session_start_console(self) -> None:
        """Log session start to console with colors."""
        print(
            f"\n{self.colors['bold']}{self.colors['blue']}🤖 AGENT SESSION STARTED{self.colors['end']}"
        )
        print(f"{self.colors['cyan']}Session ID:{self.colors['end']} {self.session_id}")
        print(
            f"{self.colors['cyan']}Provider:{self.colors['end']} {self.current_session.provider}"
        )
        print(
            f"{self.colors['cyan']}Model:{self.colors['end']} {self.current_session.model}"
        )
        if self.current_session.system_prompt:
            print(
                f"{self.colors['cyan']}System:{self.colors['end']} {self.current_session.system_prompt}"
            )
        print(
            f"{self.colors['cyan']}Query:{self.colors['end']} {self.current_session.input_query}"
        )
        if self.current_session.tools_requested:
            tools_str = ", ".join(self.current_session.tools_requested)
            print(f"{self.colors['cyan']}Tools:{self.colors['end']} {tools_str}")
        print(f"{self.colors['blue']}{'=' * 80}{self.colors['end']}")

    def _log_session_start_file(self) -> None:
        """Log session start to file without colors."""
        if not self.log_file:
            return

        self.log_file.write("\n🤖 AGENT SESSION STARTED\n")
        self.log_file.write(f"Session ID: {self.session_id}\n")
        self.log_file.write(f"Provider: {self.current_session.provider}\n")
        self.log_file.write(f"Model: {self.current_session.model}\n")
        if self.current_session.system_prompt:
            self.log_file.write(f"System: {self.current_session.system_prompt}\n")
        self.log_file.write(f"Query: {self.current_session.input_query}\n")
        if self.current_session.tools_requested:
            tools_str = ", ".join(self.current_session.tools_requested)
            self.log_file.write(f"Tools: {tools_str}\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.flush()

    def _log_step_output(self, entry: LogEntry) -> None:
        """Output a step log entry."""
        if self.format == LogFormat.CONSOLE:
            # Console output with colors
            self._log_step_console(entry)
            # Clean file output without colors
            if self.log_file:
                self._log_step_file(entry)
        elif self.format == LogFormat.JSON:
            print(json.dumps(asdict(entry), indent=2))
        elif self.format == LogFormat.MARKDOWN:
            self._log_step_markdown(entry)

    def _log_step_console(self, entry: LogEntry) -> None:
        """Log step in console format."""
        icon_map = {
            "tool_selection": "🔧",
            "tool_execution": "⚙️",
            "provider_api_call": "🌐",
            "provider_tool_execution": "🛠️",
            "response_parsing": "📖",
            "error_handling": "❌",
        }

        icon = icon_map.get(entry.action, "📝")
        timestamp = entry.timestamp.split("T")[1][:8]  # HH:MM:SS

        print(
            f"\n{self.colors['yellow']}[{timestamp}] Step {entry.step}: {icon} {entry.action.replace('_', ' ').title()}{self.colors['end']}"
        )

        # Format details based on action type
        if entry.action == "tool_selection":
            tools = entry.details.get("selected_tools", [])
            print(
                f"  {self.colors['purple']}Selected tools:{self.colors['end']} {', '.join(tools)}"
            )

        elif entry.action == "tool_execution":
            tool_name = entry.details.get("tool_name")
            status = entry.details.get("status", "unknown")
            print(f"  {self.colors['purple']}Tool:{self.colors['end']} {tool_name}")
            print(f"  {self.colors['purple']}Status:{self.colors['end']} {status}")

        elif entry.action == "provider_api_call":
            api_type = entry.details.get("api_type")
            model = entry.details.get("model")
            tools_count = entry.details.get("tools_count", 0)
            print(f"  {self.colors['purple']}API:{self.colors['end']} {api_type}")
            print(f"  {self.colors['purple']}Model:{self.colors['end']} {model}")
            if tools_count > 0:
                print(
                    f"  {self.colors['purple']}Tools:{self.colors['end']} {tools_count} tools attached"
                )

        elif entry.action == "provider_tool_execution":
            tools_executed = entry.details.get("tools_executed", [])
            metadata_types = entry.details.get("metadata_types", [])
            print(
                f"  {self.colors['purple']}Tools Executed:{self.colors['end']} {', '.join(tools_executed)}"
            )
            if metadata_types:
                print(
                    f"  {self.colors['purple']}Metadata Generated:{self.colors['end']} {', '.join(metadata_types)}"
                )

        # Show detailed info for debug level
        if self.level == LogLevel.DEBUG:
            for key, value in entry.details.items():
                if key not in [
                    "api_type",
                    "model",
                    "tools_count",
                    "tool_name",
                    "status",
                    "selected_tools",
                ]:
                    if isinstance(value, dict) and len(str(value)) > 100:
                        print(
                            f"  {self.colors['purple']}{key}:{self.colors['end']} {type(value).__name__} ({len(value)} items)"
                        )
                    else:
                        print(
                            f"  {self.colors['purple']}{key}:{self.colors['end']} {value}"
                        )

    def _log_step_file(self, entry: LogEntry) -> None:
        """Log step to file without colors."""
        if not self.log_file:
            return

        timestamp = entry.timestamp.split("T")[1][:8]  # HH:MM:SS

        self.log_file.write(
            f"\n[{timestamp}] Step {entry.step}: {entry.action.replace('_', ' ').title()}\n"
        )

        # Format details based on action type
        if entry.action == "tool_selection":
            tools = entry.details.get("selected_tools", [])
            self.log_file.write(f"  Selected tools: {', '.join(tools)}\n")

        elif entry.action == "tool_execution":
            tool_name = entry.details.get("tool_name")
            status = entry.details.get("status", "unknown")
            self.log_file.write(f"  Tool: {tool_name}\n")
            self.log_file.write(f"  Status: {status}\n")

        elif entry.action == "provider_api_call":
            api_type = entry.details.get("api_type")
            model = entry.details.get("model")
            tools_count = entry.details.get("tools_count", 0)
            self.log_file.write(f"  API: {api_type}\n")
            self.log_file.write(f"  Model: {model}\n")
            if tools_count > 0:
                self.log_file.write(f"  Tools: {tools_count} tools attached\n")

        elif entry.action == "provider_tool_execution":
            tools_executed = entry.details.get("tools_executed", [])
            metadata_types = entry.details.get("metadata_types", [])
            self.log_file.write(f"  Tools Executed: {', '.join(tools_executed)}\n")
            if metadata_types:
                self.log_file.write(
                    f"  Metadata Generated: {', '.join(metadata_types)}\n"
                )

        # Show detailed info for debug level
        if self.level == LogLevel.DEBUG:
            for key, value in entry.details.items():
                if key not in [
                    "api_type",
                    "model",
                    "tools_count",
                    "tool_name",
                    "status",
                    "selected_tools",
                ]:
                    if isinstance(value, dict) and len(str(value)) > 100:
                        self.log_file.write(
                            f"  {key}: {type(value).__name__} ({len(value)} items)\n"
                        )
                    else:
                        self.log_file.write(f"  {key}: {value}\n")

        self.log_file.flush()

    def _log_tool_call_output(self, tool_log: ToolCallLog) -> None:
        """Output a tool call log entry."""
        if self.format == LogFormat.CONSOLE:
            # Console output with colors
            self._log_tool_call_console(tool_log)
            # Clean file output without colors
            if self.log_file:
                self._log_tool_call_file(tool_log)

    def _log_tool_call_console(self, tool_log: ToolCallLog) -> None:
        """Log tool call in console format."""
        status_icon = "✅" if not tool_log.error else "❌"
        exec_time = (
            f" ({tool_log.execution_time_ms:.1f}ms)"
            if tool_log.execution_time_ms
            else ""
        )

        print(
            f"\n{self.colors['green']}  🔨 TOOL CALL: {tool_log.tool_name} {status_icon}{exec_time}{self.colors['end']}"
        )
        print(
            f"    {self.colors['cyan']}ID:{self.colors['end']} {tool_log.tool_call_id}"
        )

        # Show inputs
        if tool_log.inputs:
            print(f"    {self.colors['cyan']}Inputs:{self.colors['end']}")
            for key, value in tool_log.inputs.items():
                print(f"      {key}: {value}")

        # Show outputs
        if tool_log.outputs:
            print(f"    {self.colors['cyan']}Output:{self.colors['end']}")
            print(f"      {tool_log.outputs}")

        # Show errors
        if tool_log.error:
            print(
                f"    {self.colors['red']}Error:{self.colors['end']} {tool_log.error}"
            )

    def _log_tool_call_file(self, tool_log: ToolCallLog) -> None:
        """Log tool call to file without colors."""
        if not self.log_file:
            return

        status_icon = "✅" if not tool_log.error else "❌"
        exec_time = (
            f" ({tool_log.execution_time_ms:.1f}ms)"
            if tool_log.execution_time_ms
            else ""
        )

        self.log_file.write(
            f"\n  🔨 TOOL CALL: {tool_log.tool_name} {status_icon}{exec_time}\n"
        )
        self.log_file.write(f"    ID: {tool_log.tool_call_id}\n")

        # Show inputs
        if tool_log.inputs:
            self.log_file.write(f"    Inputs:\n")
            for key, value in tool_log.inputs.items():
                self.log_file.write(f"      {key}: {value}\n")

        # Show outputs
        if tool_log.outputs:
            self.log_file.write(f"    Output:\n")
            self.log_file.write(f"      {tool_log.outputs}\n")

        # Show errors
        if tool_log.error:
            self.log_file.write(f"    Error: {tool_log.error}\n")

        self.log_file.flush()

    def _log_session_end(self) -> None:
        """Log the end of an agent session."""
        if not self.current_session:
            return

        if self.format == LogFormat.CONSOLE:
            # Console output with colors
            self._log_session_end_console()
            # Clean file output without colors
            if self.log_file:
                self._log_session_end_file()

    def _log_session_end_console(self) -> None:
        """Log session end to console with colors."""
        print(
            f"\n{self.colors['bold']}{self.colors['green']}✅ SESSION COMPLETED{self.colors['end']}"
        )
        print(f"{self.colors['cyan']}Final Result:{self.colors['end']}")
        print(f"  {self.current_session.final_result}")

        print(f"\n{self.colors['cyan']}Session Summary:{self.colors['end']}")
        print(
            f"  {self.colors['purple']}Total Time:{self.colors['end']} {self.current_session.total_execution_time_ms:.1f}ms"
        )
        print(
            f"  {self.colors['purple']}Steps:{self.colors['end']} {len(self.current_session.steps)}"
        )

        # Count both local and provider tool executions
        local_tool_calls = len(self.current_session.tool_calls)
        provider_tool_executions = len(
            [
                step
                for step in self.current_session.steps
                if step.action == "provider_tool_execution"
            ]
        )
        total_tool_usage = local_tool_calls + provider_tool_executions

        print(
            f"  {self.colors['purple']}Local Tool Calls:{self.colors['end']} {local_tool_calls}"
        )
        print(
            f"  {self.colors['purple']}Provider Tool Executions:{self.colors['end']} {provider_tool_executions}"
        )
        print(
            f"  {self.colors['purple']}Total Tool Usage:{self.colors['end']} {total_tool_usage}"
        )

        # Show metadata summary
        if self.current_session.metadata:
            print(
                f"  {self.colors['purple']}Metadata:{self.colors['end']} {list(self.current_session.metadata.keys())}"
            )

        print(f"{self.colors['blue']}{'=' * 80}{self.colors['end']}\n")

    def _log_session_end_file(self) -> None:
        """Log session end to file without colors."""
        if not self.log_file:
            return

        self.log_file.write(f"\n✅ SESSION COMPLETED\n")
        self.log_file.write(f"Final Result:\n")
        self.log_file.write(f"  {self.current_session.final_result}\n")

        self.log_file.write(f"\nSession Summary:\n")
        self.log_file.write(
            f"  Total Time: {self.current_session.total_execution_time_ms:.1f}ms\n"
        )
        self.log_file.write(f"  Steps: {len(self.current_session.steps)}\n")

        # Count both local and provider tool executions
        local_tool_calls = len(self.current_session.tool_calls)
        provider_tool_executions = len(
            [
                step
                for step in self.current_session.steps
                if step.action == "provider_tool_execution"
            ]
        )
        total_tool_usage = local_tool_calls + provider_tool_executions

        self.log_file.write(f"  Local Tool Calls: {local_tool_calls}\n")
        self.log_file.write(f"  Provider Tool Executions: {provider_tool_executions}\n")
        self.log_file.write(f"  Total Tool Usage: {total_tool_usage}\n")

        # Show metadata summary
        if self.current_session.metadata:
            self.log_file.write(
                f"  Metadata: {list(self.current_session.metadata.keys())}\n"
            )

        self.log_file.write("=" * 80 + "\n")
        self.log_file.flush()

    def _log_step_markdown(self, entry: LogEntry) -> None:
        """Log step in markdown format."""
        action_title = entry.action.replace("_", " ").title()
        print(f"\n### Step {entry.step}: {action_title}")
        print(f"**Timestamp:** {entry.timestamp}")

        for key, value in entry.details.items():
            print(f"**{key.replace('_', ' ').title()}:** {value}")

    def export_session(self, format: LogFormat = LogFormat.JSON) -> str:
        """Export current session in specified format."""
        if not self.current_session:
            return ""

        if format == LogFormat.JSON:
            return json.dumps(asdict(self.current_session), indent=2)
        elif format == LogFormat.MARKDOWN:
            return self._export_markdown()
        else:
            return str(self.current_session)

    def _export_markdown(self) -> str:
        """Export session as markdown documentation."""
        if not self.current_session:
            return ""

        md = f"# Agent Session Report\n\n"
        md += f"**Session ID:** {self.current_session.session_id}\n"
        md += f"**Provider:** {self.current_session.provider}\n"
        md += f"**Model:** {self.current_session.model}\n"
        md += f"**Start Time:** {self.current_session.start_time}\n"
        md += f"**Duration:** {self.current_session.total_execution_time_ms:.1f}ms\n\n"

        md += f"## Input Query\n```\n{self.current_session.input_query}\n```\n\n"

        if self.current_session.system_prompt:
            md += (
                f"## System Prompt\n```\n{self.current_session.system_prompt}\n```\n\n"
            )

        if self.current_session.tools_requested:
            md += f"## Tools Requested\n"
            for tool in self.current_session.tools_requested:
                md += f"- {tool}\n"
            md += "\n"

        md += f"## Execution Steps\n"
        for step in self.current_session.steps:
            md += f"### {step.step}. {step.action.replace('_', ' ').title()}\n"
            for key, value in step.details.items():
                md += f"**{key.replace('_', ' ').title()}:** {value}\n"
            md += "\n"

        if self.current_session.tool_calls:
            md += f"## Tool Calls\n"
            for i, call in enumerate(self.current_session.tool_calls, 1):
                md += f"### {i}. {call.tool_name}\n"
                md += f"**ID:** {call.tool_call_id}\n"
                if call.inputs:
                    md += f"**Inputs:**\n```json\n{json.dumps(call.inputs, indent=2)}\n```\n"
                if call.outputs:
                    md += f"**Output:**\n```\n{call.outputs}\n```\n"
                if call.error:
                    md += f"**Error:** {call.error}\n"
                if call.execution_time_ms:
                    md += f"**Execution Time:** {call.execution_time_ms:.1f}ms\n"
                md += "\n"

        md += f"## Final Result\n```\n{self.current_session.final_result}\n```\n"

        return md


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
        if exc_type:
            error_msg = f"Session failed: {exc_type.__name__}: {exc_val}"
            self.logger.end_session(error_msg)
        else:
            self.logger.end_session("Session completed successfully")


# Global logger instance for easy access
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


# Convenience functions for common logging patterns
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
