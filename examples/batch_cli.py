#!/usr/bin/env python3
"""
OpenAI Batch API CLI Tool

Command-line interface for OpenAI Batch API operations:
- Create and submit batch jobs
- Monitor batch job status
- Download results and errors
- Manage batch files

Usage:
    python batch_cli.py create movie_descriptions.txt --model gpt-4o-mini
    python batch_cli.py status batch_abc123
    python batch_cli.py results batch_abc123 --output results.jsonl
    python batch_cli.py list
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from llm_studio.batch import OpenAIBatchProcessor, BatchStatus
from llm_studio.schemas.messages import SystemMessage, UserMessage


def create_text_batch(args):
    """Create a batch job from text file."""
    processor = OpenAIBatchProcessor(api_key=os.getenv("OPENAI_API_KEY"))

    # Read input texts
    if not Path(args.input_file).exists():
        print(f"❌ Input file not found: {args.input_file}")
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"📝 Processing {len(texts)} text inputs")

    # Create tasks
    system_prompt = args.system_prompt or "You are a helpful assistant."
    tasks = []

    for i, text in enumerate(texts):
        task = processor.create_task(
            custom_id=f"task-{i}",
            model=args.model,
            messages=[SystemMessage(system_prompt), UserMessage(text)],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            response_format={"type": "json_object"} if args.json_output else None,
        )
        tasks.append(task)

    # Submit batch
    try:
        batch_job = processor.submit_batch(
            tasks, metadata={"source": args.input_file, "model": args.model}
        )

        print(f"✅ Batch job created: {batch_job.id}")
        print(f"   Status: {batch_job.status.value}")
        print(f"   Input file: {batch_job.input_file_id}")
        print(f"   Estimated completion: 24h (usually faster)")

        # Save job ID for future reference
        job_file = f"batch_job_{batch_job.id}.txt"
        with open(job_file, "w") as f:
            f.write(batch_job.id)
        print(f"📄 Job ID saved to: {job_file}")

    except Exception as e:
        print(f"❌ Failed to create batch job: {e}")


def check_batch_status(args):
    """Check status of a batch job."""
    processor = OpenAIBatchProcessor(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        batch_job = processor.get_batch_status(args.batch_id)

        print(f"📊 Batch Job Status: {args.batch_id}")
        print(f"   Status: {batch_job.status.value}")

        if batch_job.request_counts:
            counts = batch_job.request_counts
            print(f"   Requests: {counts}")

        if batch_job.created_at:
            print(f"   Created: {batch_job.created_at}")
        if batch_job.completed_at:
            print(f"   Completed: {batch_job.completed_at}")
        if batch_job.metadata:
            print(f"   Metadata: {batch_job.metadata}")

        # Show next steps based on status
        if batch_job.status == BatchStatus.COMPLETED:
            print(f"\n✅ Job completed! Download results with:")
            print(f"   python batch_cli.py results {args.batch_id}")
        elif batch_job.status == BatchStatus.IN_PROGRESS:
            print(f"\n⏳ Job in progress. Check again later.")
        elif batch_job.status in [BatchStatus.FAILED, BatchStatus.EXPIRED]:
            print(f"\n❌ Job {batch_job.status.value}")
            if batch_job.error_file_id:
                print(f"   Download errors with:")
                print(f"   python batch_cli.py errors {args.batch_id}")

    except Exception as e:
        print(f"❌ Failed to get batch status: {e}")


def download_results(args):
    """Download results from a completed batch job."""
    processor = OpenAIBatchProcessor(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        batch_job = processor.get_batch_status(args.batch_id)

        if batch_job.status != BatchStatus.COMPLETED:
            print(f"❌ Batch job not completed. Status: {batch_job.status.value}")
            return

        # Download results
        output_file = args.output or f"results_{args.batch_id}.jsonl"
        results = processor.download_results(batch_job, output_file)

        print(f"✅ Downloaded {len(results)} results to: {output_file}")

        # Show sample results
        if results and args.show_sample:
            print(f"\n📋 Sample Results (first 3):")
            for i, result in enumerate(results[:3]):
                print(f"   {result.custom_id}:")
                if result.error:
                    print(f"      Error: {result.error}")
                else:
                    content = (
                        result.response.get("body", {})
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    print(
                        f"      Content: {content[:100]}..."
                        if len(content) > 100
                        else f"      Content: {content}"
                    )

    except Exception as e:
        print(f"❌ Failed to download results: {e}")


def list_batch_jobs(args):
    """List recent batch jobs."""
    processor = OpenAIBatchProcessor(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        jobs = processor.list_batch_jobs(limit=args.limit)

        print(f"📊 Recent Batch Jobs ({len(jobs)} found):")

        for job in jobs:
            print(f"\n   {job.id}")
            print(f"      Status: {job.status.value}")
            if job.metadata:
                print(f"      Metadata: {job.metadata}")
            if job.request_counts:
                print(f"      Requests: {job.request_counts}")

        if not jobs:
            print("   No batch jobs found")

    except Exception as e:
        print(f"❌ Failed to list batch jobs: {e}")


def cancel_batch_job(args):
    """Cancel a batch job."""
    processor = OpenAIBatchProcessor(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        batch_job = processor.cancel_batch_job(args.batch_id)
        print(f"✅ Batch job cancelled: {batch_job.id}")
        print(f"   Status: {batch_job.status.value}")

    except Exception as e:
        print(f"❌ Failed to cancel batch job: {e}")


def create_sample_input():
    """Create sample input files for testing."""
    print("📝 Creating sample input files...")

    # Sample movie descriptions
    movies = [
        "The aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son.",
        "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
        "A fairy tale adventure about a beautiful young woman and her one true love.",
        "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland.",
        "The adventures of a legendary concierge and his protégé at a famous European hotel.",
    ]

    with open("sample_movies.txt", "w") as f:
        for movie in movies:
            f.write(movie + "\n")

    print("✅ Created sample_movies.txt")
    print("💡 Usage: python batch_cli.py create sample_movies.txt --model gpt-4o-mini")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create batch job from text file
  python batch_cli.py create movie_descriptions.txt --model gpt-4o-mini
  
  # Check batch job status
  python batch_cli.py status batch_abc123
  
  # Download results
  python batch_cli.py results batch_abc123 --output my_results.jsonl
  
  # List recent batch jobs
  python batch_cli.py list
  
  # Cancel a batch job
  python batch_cli.py cancel batch_abc123
  
  # Create sample input file
  python batch_cli.py sample
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new batch job")
    create_parser.add_argument(
        "input_file", help="Input text file (one prompt per line)"
    )
    create_parser.add_argument(
        "--model", default="gpt-4o-mini", help="OpenAI model to use"
    )
    create_parser.add_argument("--system-prompt", help="System prompt for all tasks")
    create_parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature (0.0-2.0)"
    )
    create_parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens per response"
    )
    create_parser.add_argument(
        "--json-output", action="store_true", help="Force JSON output format"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check batch job status")
    status_parser.add_argument("batch_id", help="Batch job ID")

    # Results command
    results_parser = subparsers.add_parser("results", help="Download batch job results")
    results_parser.add_argument("batch_id", help="Batch job ID")
    results_parser.add_argument("--output", help="Output file path")
    results_parser.add_argument(
        "--show-sample", action="store_true", help="Show sample results"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List recent batch jobs")
    list_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum jobs to show"
    )

    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a batch job")
    cancel_parser.add_argument("batch_id", help="Batch job ID")

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Create sample input files")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    if not args.command:
        parser.print_help()
        return

    # Check API key
    if args.command != "sample" and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Set it in .env file or export OPENAI_API_KEY=your-key")
        return

    # Route to appropriate function
    if args.command == "create":
        create_text_batch(args)
    elif args.command == "status":
        check_batch_status(args)
    elif args.command == "results":
        download_results(args)
    elif args.command == "list":
        list_batch_jobs(args)
    elif args.command == "cancel":
        cancel_batch_job(args)
    elif args.command == "sample":
        create_sample_input()


if __name__ == "__main__":
    main()
