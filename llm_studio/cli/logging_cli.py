#!/usr/bin/env python3
"""CLI interface for agent logging system."""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

from ..logging import setup_logging, LogLevel, LogFormat


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


def generate_log_filename(
    provider: str, model: str, custom_path: Optional[str] = None
) -> str:
    """Generate timestamped log filename in logs/ directory."""
    if custom_path:
        return custom_path

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"logs/{timestamp}_{provider}_{model.replace(':', '_').replace('/', '_')}.log"
    )

    return filename


def configure_logging_from_args(
    args: argparse.Namespace, provider: str = "unknown", model: str = "unknown"
) -> Optional[object]:
    """Configure logging based on parsed command line arguments."""
    if not getattr(args, "log", False):
        return None

    # Parse log level
    level_map = {
        "error": LogLevel.ERROR,
        "warn": LogLevel.WARN,
        "info": LogLevel.INFO,
        "debug": LogLevel.DEBUG,
    }
    level = level_map[args.log_level]

    # Parse log format
    format_map = {
        "console": LogFormat.CONSOLE,
        "json": LogFormat.JSON,
        "markdown": LogFormat.MARKDOWN,
    }
    format = format_map[args.log_format]

    # Setup logging
    logger = setup_logging(level=level, format=format, enabled=True)

    # Always create logs directory when logging is enabled
    os.makedirs("logs", exist_ok=True)

    # Handle log file output
    log_file_path = getattr(args, "log_file", None)

    # By default, always save timestamped file to logs/ directory
    if not log_file_path:
        log_file_path = generate_log_filename(provider, model)

    # Create directory if needed
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Open log file for writing
    log_file = open(log_file_path, "w", encoding="utf-8")

    # Always save to timestamped file, show console output for console format
    if format == LogFormat.CONSOLE:
        # Console format: show on screen AND save clean version to file
        logger.log_file = log_file
        print(f"Logging enabled: {log_file_path}")

        # Return cleanup function
        def cleanup():
            if logger.log_file:
                logger.log_file.close()
                logger.log_file = None
            print(f"Session saved: {log_file_path}")

        return cleanup

    else:
        # JSON/Markdown formats - redirect stdout to file
        original_stdout = sys.stdout
        sys.stdout = log_file

        # Return cleanup function
        def cleanup():
            sys.stdout = original_stdout
            log_file.close()
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


def demo_cli_logging():
    """Demo CLI logging interface."""
    print("Agent Logging CLI Demo")
    print("=" * 40)

    # Simulate different CLI configurations
    test_configs = [
        ["-l"],
        ["-l", "--log-level", "warn"],
        ["-l", "--log-level", "debug", "--log-format", "json"],
        ["-lf", "demo_session.log"],
        ["-l", "--log-format", "markdown"],
    ]

    for config in test_configs:
        print(f"\nTesting config: {' '.join(config)}")

        try:
            args = parse_logging_args(config)
            cleanup = configure_logging_from_args(args, "openai", "gpt-4o-mini")

            print(
                f"Logging configured: level={args.log_level}, format={args.log_format}"
            )

            if cleanup:
                cleanup()

        except Exception as e:
            print(f"Configuration failed: {e}")


if __name__ == "__main__":
    demo_cli_logging()
