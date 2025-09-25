#!/usr/bin/env python3
"""
Simple CLI Logging Demo

Demonstrates the clean CLI interface for agent logging:
- Short flags: -l for --log, -lf for --log-file
- Automatic timestamped logs in logs/ directory
- Provider/model specific filenames
"""

import os
import subprocess
import sys
from pathlib import Path


def run_demo_command(cmd: list, description: str) -> None:
    """Run a demo command and show the result."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print("Output:")
                print(
                    result.stdout[:500] + "..."
                    if len(result.stdout) > 500
                    else result.stdout
                )
        else:
            print("❌ Failed!")
            if result.stderr:
                print("Error:")
                print(
                    result.stderr[:300] + "..."
                    if len(result.stderr) > 300
                    else result.stderr
                )

    except subprocess.TimeoutExpired:
        print("⏰ Command timed out (expected for real API calls)")
    except Exception as e:
        print(f"❌ Error running command: {e}")


def demo_cli_flags():
    """Demo the clean CLI flags."""
    print("🚀 Clean CLI Logging Demo")
    print("=" * 50)
    print("Demonstrating short flags and automatic log file generation")

    # Demo commands (using mock provider to avoid API costs)
    demos = [
        {
            "cmd": [
                "python",
                "examples/agent_with_logging.py",
                "-l",
                "--provider",
                "mock",
                "Hello world",
            ],
            "desc": "Basic logging with -l flag (console output)",
        },
        {
            "cmd": [
                "python",
                "examples/agent_with_logging.py",
                "-l",
                "--log-level",
                "detailed",
                "--provider",
                "mock",
                "Calculate 5 factorial",
            ],
            "desc": "Detailed logging (-l with --log-level detailed)",
        },
        {
            "cmd": [
                "python",
                "examples/agent_with_logging.py",
                "-l",
                "--log-format",
                "json",
                "--provider",
                "mock",
                "Format data as JSON",
            ],
            "desc": "JSON logging (auto-saves to logs/ directory with timestamp)",
        },
        {
            "cmd": [
                "python",
                "examples/agent_with_logging.py",
                "-lf",
                "custom_demo.log",
                "--provider",
                "mock",
                "Custom log file demo",
            ],
            "desc": "Custom log file with -lf flag",
        },
        {
            "cmd": [
                "python",
                "examples/agent_with_logging.py",
                "-l",
                "--log-format",
                "markdown",
                "--provider",
                "mock",
                "Markdown export demo",
            ],
            "desc": "Markdown logging (auto-saves to logs/ directory)",
        },
    ]

    for demo in demos:
        run_demo_command(demo["cmd"], demo["desc"])

    # Show generated log files
    print("\n📁 Generated Log Files:")
    print("-" * 30)

    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        for log_file in sorted(log_files)[-5:]:  # Show last 5 files
            size = log_file.stat().st_size
            print(f"📄 {log_file.name} ({size} bytes)")
    else:
        print("No logs directory found")

    # Show custom log file
    custom_log = Path("custom_demo.log")
    if custom_log.exists():
        size = custom_log.stat().st_size
        print(f"📄 custom_demo.log ({size} bytes)")


def demo_real_api_logging():
    """Demo real API logging (if API key available)."""
    print("\n🌐 Real API Logging Demo (Optional)")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OPENAI_API_KEY not found - skipping real API demos")
        return

    print("Running real API calls with logging...")

    real_demos = [
        {
            "cmd": ["python", "examples/agent_with_logging.py", "-l", "What's 2+2?"],
            "desc": "Simple math with real API logging",
        },
        {
            "cmd": [
                "python",
                "examples/agent_with_logging.py",
                "-l",
                "--log-level",
                "detailed",
                "Search for Python tutorials",
                "--tools",
                "openai_web_search",
            ],
            "desc": "Web search with detailed logging",
        },
    ]

    for demo in real_demos:
        print(f"\n⚠ Making real API call: {' '.join(demo['cmd'])}")
        print("This will use OpenAI credits...")
        # Don't actually run to avoid costs in demo
        print("✅ Command prepared (not executed to save credits)")


def show_log_file_formats():
    """Show examples of different log file formats."""
    print("\n📋 Log File Format Examples")
    print("=" * 50)

    # Show what different formats look like
    format_examples = {
        "Console": """
🤖 AGENT SESSION STARTED
Session ID: session_1703123456
Provider: openai
Model: gpt-4o-mini
Query: Hello world
Tools: json_format
================================================================================

[14:30:15] Step 1: 🔧 Tool Selection
  Selected tools: json_format
  Tool count: 1
  Local tools: json_format

[14:30:16] Step 2: 🌐 Provider Api Call
  API: chat_completions
  Model: gpt-4o-mini
  Tools: 1 tools attached

  🔨 TOOL CALL: json_format ✅ (45.2ms)
    ID: call_123
    Inputs:
      data: {"message": "Hello world"}
    Output:
      {"message": "Hello world"}

✅ SESSION COMPLETED
Final Result: I've formatted your data as JSON: {"message": "Hello world"}
        """,
        "JSON": """{
  "session_id": "session_1703123456",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "input_query": "Hello world",
  "tools_requested": ["json_format"],
  "tool_calls": [
    {
      "tool_name": "json_format",
      "inputs": {"data": {"message": "Hello world"}},
      "outputs": "{\\"message\\": \\"Hello world\\"}",
      "execution_time_ms": 45.2
    }
  ],
  "final_result": "I've formatted your data as JSON...",
  "total_execution_time_ms": 1247.8
}""",
        "Markdown": """# Agent Session Report

**Provider:** openai
**Model:** gpt-4o-mini
**Duration:** 1247.8ms

## Input Query
```
Hello world
```

## Tools Requested
- json_format

## Tool Calls
### 1. json_format
**Inputs:**
```json
{"data": {"message": "Hello world"}}
```
**Output:**
```
{"message": "Hello world"}
```
**Execution Time:** 45.2ms

## Final Result
```
I've formatted your data as JSON: {"message": "Hello world"}
```""",
    }

    for format_name, example in format_examples.items():
        print(f"\n{format_name} Format:")
        print(example[:300] + "..." if len(example) > 300 else example)


def main():
    """Main demo function."""
    print("🔍 LLM Studio Logging CLI Demo")
    print("=" * 70)
    print("Demonstrating the clean CLI interface for comprehensive logging")
    print("=" * 70)

    demo_cli_flags()
    demo_real_api_logging()
    show_log_file_formats()

    print("\n✅ CLI Logging Demo Complete!")
    print("\n💡 Key Features:")
    print("  ✅ Short flags: -l (--log), -lf (--log-file)")
    print("  ✅ Auto-timestamped logs: logs/20241221_143015_openai_gpt-4o-mini.log")
    print("  ✅ Provider/model specific filenames")
    print("  ✅ Multiple formats: console, JSON, markdown")
    print("  ✅ Auto-saves JSON/Markdown to logs/ directory")
    print("  ✅ Comprehensive interaction tracking")

    print("\n🚀 Usage:")
    print("  python examples/agent_with_logging.py -l 'Your query here'")
    print(
        "  python examples/agent_with_logging.py -l --log-level detailed 'Complex task'"
    )
    print(
        "  python examples/agent_with_logging.py -lf custom.log 'Save to custom file'"
    )


if __name__ == "__main__":
    main()
