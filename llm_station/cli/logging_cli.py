#!/usr/bin/env python3
"""CLI interface for agent logging system."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..logging import setup_logging, LogLevel, LogFormat

# Default log directory
DEFAULT_LOG_DIR = "logs"

# Timestamp format for log filenames
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Characters to sanitize in model names for filenames
MODEL_NAME_SANITIZE_CHARS = {":": "_", "/": "_"}


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add logging arguments to an argument parser."""
    logging_group = parser.add_argument_group("Logging Options")

    logging_group.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Enable detailed logging of agent interactions",
    )

    logging_group.add_argument(
        "-ll",
        "--log-level",
        choices=["error", "warn", "info", "debug"],
        default="info",
        help="Logging verbosity level (default: info)",
    )

    logging_group.add_argument(
        "-lft",
        "--log-format",
        choices=["console", "json", "markdown"],
        default="console",
        help="Logging output format (default: console)",
    )

    logging_group.add_argument(
        "-lf",
        "--log-file",
        type=str,
        help="Custom log file path (default: auto-generated in logs/ directory)",
    )


def _ensure_log_directory(log_dir: str = DEFAULT_LOG_DIR) -> None:
    """Ensure log directory exists, creating it if necessary.

    Args:
        log_dir: Directory path to ensure exists

    Raises:
        OSError: If directory cannot be created
    """
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create log directory '{log_dir}': {e}") from e


def generate_log_filename(
    provider: str,
    model: str,
    custom_path: Optional[str] = None,
    log_dir: str = DEFAULT_LOG_DIR,
) -> str:
    """Generate timestamped log filename in logs/ directory.

    Args:
        provider: Provider name for filename
        model: Model name for filename
        custom_path: Optional custom file path
        log_dir: Directory for log files (default: logs/)

    Returns:
        Full path to log file
    """
    if custom_path:
        return custom_path

    # Ensure logs directory exists
    _ensure_log_directory(log_dir)

    # Generate timestamped filename
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    # Sanitize model name for filename safety
    sanitized_model = model
    for old_char, new_char in MODEL_NAME_SANITIZE_CHARS.items():
        sanitized_model = sanitized_model.replace(old_char, new_char)
    filename = f"{log_dir}/{timestamp}_{provider}_{sanitized_model}.log"

    return filename


def configure_logging_from_args(
    args: argparse.Namespace, provider: str = "unknown", model: str = "unknown"
) -> Optional[Callable[[], None]]:
    """Configure logging based on parsed command line arguments.

    Args:
        args: Parsed command line arguments
        provider: Provider name for log filename generation
        model: Model name for log filename generation

    Returns:
        Cleanup function if logging is enabled, None otherwise

    Raises:
        TypeError: If provider or model are not strings
        ValueError: If log file path is invalid
        OSError: If log file cannot be created
    """
    if not getattr(args, "log", False):
        return None

    # Validate inputs - raise errors instead of silent conversion
    if not isinstance(provider, str):
        raise TypeError(f"provider must be a string, got {type(provider).__name__}")
    if not isinstance(model, str):
        raise TypeError(f"model must be a string, got {type(model).__name__}")

    # Parse log level
    level_map = {
        "error": LogLevel.ERROR,
        "warn": LogLevel.WARN,
        "info": LogLevel.INFO,
        "debug": LogLevel.DEBUG,
    }
    level = level_map.get(args.log_level, LogLevel.INFO)

    # Parse log format (avoid shadowing built-in format())
    format_map = {
        "console": LogFormat.CONSOLE,
        "json": LogFormat.JSON,
        "markdown": LogFormat.MARKDOWN,
    }
    log_format = format_map.get(args.log_format, LogFormat.CONSOLE)

    # Setup logging
    logger = setup_logging(level=level, format=log_format, enabled=True)

    # Handle log file output
    log_file_path = getattr(args, "log_file", None)

    # By default, always save timestamped file to logs/ directory
    if not log_file_path:
        log_file_path = generate_log_filename(provider, model)
    elif not isinstance(log_file_path, str):
        raise ValueError(
            f"log_file must be a string, got {type(log_file_path).__name__}"
        )

    # Validate and create directory if needed
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        _ensure_log_directory(log_dir)

    # Open log file for writing
    try:
        log_file = open(log_file_path, "w", encoding="utf-8")
    except OSError as e:
        raise OSError(f"Failed to open log file '{log_file_path}': {e}") from e

    # Always save to timestamped file, show console output for console format
    if log_format == LogFormat.CONSOLE:
        # Console format: show on screen AND save clean version to file
        logger.log_file = log_file
        print(f"Logging enabled: {log_file_path}")

        # Return cleanup function
        def cleanup() -> None:
            if logger.log_file:
                try:
                    logger.log_file.close()
                except (OSError, IOError) as e:
                    # Log but don't fail on cleanup errors
                    print(f"Warning: Error closing log file: {e}", file=sys.stderr)
                finally:
                    logger.log_file = None
            print(f"Session saved: {log_file_path}")

        return cleanup

    else:
        # JSON/Markdown formats - redirect stdout to file
        # Note: This is a design choice for these formats, but should be used carefully
        original_stdout = sys.stdout
        sys.stdout = log_file

        # Return cleanup function
        def cleanup() -> None:
            try:
                sys.stdout = original_stdout
                log_file.close()
            except (OSError, IOError) as e:
                # Ensure stdout is restored even if close fails
                sys.stdout = original_stdout
                print(f"Warning: Error closing log file: {e}", file=sys.stderr)
            else:
                print(f"Logs saved to {log_file_path}")

        return cleanup


def create_logging_parser() -> argparse.ArgumentParser:
    """Create a standalone argument parser for logging options."""
    parser = argparse.ArgumentParser(
        description="Agent Logging Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  -l                             # Enable info-level logging (default)
  -l --log-level warn            # Warning-level logging
  -l --log-level debug           # Debug logging with full details
  -l --log-format json           # JSON structured logging (auto-saves to logs/)
  -lf custom_session.log         # Save logs to custom file
  -l --log-format markdown       # Markdown format (auto-saves to logs/)
        """,
    )

    add_logging_args(parser)
    return parser


def parse_logging_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse logging-related command line arguments."""
    parser = create_logging_parser()
    return parser.parse_args(args)
