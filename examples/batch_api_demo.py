#!/usr/bin/env python3
"""
OpenAI Batch API Demo

Demonstrates comprehensive batch processing capabilities including:
- Movie categorization from descriptions
- Image captioning with vision models
- Async batch job management
- Cost-effective large-scale processing

Based on the official OpenAI Batch API cookbook examples.
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from llm_studio.batch import (
    OpenAIBatchProcessor,
    BatchTask,
    BatchStatus,
    create_movie_categorization_batch,
    create_image_captioning_batch,
)
from llm_studio.schemas.messages import SystemMessage, UserMessage


def demo_movie_categorization():
    """Demo movie categorization using Batch API (from cookbook)."""
    print("🎬 DEMO 1: Movie Categorization Batch")
    print("=" * 60)

    load_dotenv()
    processor = OpenAIBatchProcessor(api_key=os.getenv("OPENAI_API_KEY"))

    # Sample movie descriptions
    movies = [
        {
            "title": "The Godfather",
            "description": "The aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son.",
        },
        {
            "title": "Inception",
            "description": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
        },
        {
            "title": "The Princess Bride",
            "description": "A fairy tale adventure about a beautiful young woman and her one true love.",
        },
        {
            "title": "Mad Max: Fury Road",
            "description": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland.",
        },
        {
            "title": "The Grand Budapest Hotel",
            "description": "The adventures of a legendary concierge and his protégé at a famous European hotel.",
        },
    ]

    # Create batch tasks
    tasks = []
    system_prompt = """
Your goal is to extract movie categories from movie descriptions, as well as a 1-sentence summary for these movies.
You will be provided with a movie description, and you will output a json object containing the following information:

{
    categories: string[] // Array of categories based on the movie description,
    summary: string // 1-sentence summary of the movie based on the movie description
}

Categories refer to the genre or type of the movie, like "action", "romance", "comedy", etc. Keep category names simple and use only lower case letters.
Movies can have several categories, but try to keep it under 3-4. Only mention the categories that are the most obvious based on the description.
"""

    for i, movie in enumerate(movies):
        task = processor.create_task(
            custom_id=f"movie-{i}",
            model="gpt-4o-mini",
            messages=[SystemMessage(system_prompt), UserMessage(movie["description"])],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        tasks.append(task)

    print(f"📝 Created {len(tasks)} movie categorization tasks")

    # Create and upload batch file
    file_path = processor.create_batch_file(tasks, "batch_movies.jsonl")
    print(f"📄 Batch file created: {file_path}")

    file_id = processor.upload_batch_file(file_path)
    print(f"☁️ File uploaded: {file_id}")

    # Submit batch job
    batch_job = processor.create_batch_job(
        file_id, metadata={"demo": "movie_categorization"}
    )
    print(f"🚀 Batch job submitted: {batch_job.id}")
    print(f"   Status: {batch_job.status.value}")
    print(f"   Estimated completion: 24h (usually faster)")

    # Show how to check status (don't wait for completion in demo)
    print(f"\n💡 To check status later:")
    print(f"   batch_job = processor.get_batch_status('{batch_job.id}')")
    print(f"   print(batch_job.status)")

    return batch_job.id


def demo_image_captioning():
    """Demo image captioning using Batch API (from cookbook)."""
    print("\n🖼️ DEMO 2: Image Captioning Batch")
    print("=" * 60)

    load_dotenv()
    processor = OpenAIBatchProcessor(api_key=os.getenv("OPENAI_API_KEY"))

    # Sample furniture images (placeholder URLs)
    furniture_items = [
        {
            "title": "Modern Office Chair",
            "image_url": "https://example.com/office-chair.jpg",
        },
        {
            "title": "Vintage Wooden Table",
            "image_url": "https://example.com/wooden-table.jpg",
        },
        {
            "title": "Leather Sofa Set",
            "image_url": "https://example.com/leather-sofa.jpg",
        },
    ]

    # Create vision tasks
    tasks = []
    system_prompt = """
Your goal is to generate short, descriptive captions for images of items.
You will be provided with an item image and the name of that item and you will output a caption that captures the most important information about the item.
If there are multiple items depicted, refer to the name provided to understand which item you should describe.
Your generated caption should be short (1 sentence), and include only the most important information about the item.
The most important information could be: the type of item, the style (if mentioned), the material or color if especially relevant and/or any distinctive features.
Keep it short and to the point.
"""

    for i, item in enumerate(furniture_items):
        task = processor.create_task(
            custom_id=f"furniture-{i}",
            model="gpt-4o-mini",
            messages=[
                SystemMessage(system_prompt),
                UserMessage(
                    [
                        {"type": "text", "text": item["title"]},
                        {"type": "image_url", "image_url": {"url": item["image_url"]}},
                    ]
                ),
            ],
            temperature=0.2,
            max_tokens=300,
        )
        tasks.append(task)

    print(f"📝 Created {len(tasks)} image captioning tasks")

    # Create batch file (don't submit for demo)
    file_path = processor.create_batch_file(tasks, "batch_images.jsonl")
    print(f"📄 Batch file created: {file_path}")
    print(f"💡 Ready for upload and batch processing")

    return file_path


def demo_batch_management():
    """Demo batch job management operations."""
    print("\n⚙️ DEMO 3: Batch Job Management")
    print("=" * 60)

    load_dotenv()
    processor = OpenAIBatchProcessor(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # List recent batch jobs
        recent_jobs = processor.list_batch_jobs(limit=5)
        print(f"📊 Found {len(recent_jobs)} recent batch jobs:")

        for job in recent_jobs:
            print(f"   {job.id}: {job.status.value}")
            if job.metadata:
                print(f"      Metadata: {job.metadata}")

        # Show status checking
        if recent_jobs:
            latest_job = recent_jobs[0]
            print(f"\n🔍 Checking status of latest job: {latest_job.id}")
            current_status = processor.get_batch_status(latest_job.id)
            print(f"   Current status: {current_status.status.value}")

            if current_status.request_counts:
                print(f"   Request counts: {current_status.request_counts}")

    except Exception as e:
        print(f"⚠️ No recent batch jobs or API error: {e}")


def demo_advanced_batch_creation():
    """Demo advanced batch creation with custom parameters."""
    print("\n🔧 DEMO 4: Advanced Batch Creation")
    print("=" * 60)

    load_dotenv()
    processor = OpenAIBatchProcessor(api_key=os.getenv("OPENAI_API_KEY"))

    # Create tasks with different parameters
    tasks = []

    # Task 1: Creative writing with high temperature
    tasks.append(
        processor.create_task(
            custom_id="creative-1",
            model="gpt-4o-mini",
            messages=[
                SystemMessage("You are a creative writer"),
                UserMessage("Write a short story about a robot learning to paint"),
            ],
            temperature=0.9,
            max_tokens=500,
        )
    )

    # Task 2: Technical analysis with low temperature
    tasks.append(
        processor.create_task(
            custom_id="technical-1",
            model="gpt-4o-mini",
            messages=[
                SystemMessage("You are a technical analyst"),
                UserMessage("Explain quantum computing in simple terms"),
            ],
            temperature=0.1,
            max_tokens=300,
        )
    )

    # Task 3: JSON structured output
    tasks.append(
        processor.create_task(
            custom_id="structured-1",
            model="gpt-4o-mini",
            messages=[
                SystemMessage("Extract key information as JSON"),
                UserMessage(
                    "Analyze this: 'Tesla stock up 5% on new AI partnership news'"
                ),
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    )

    print(f"📝 Created {len(tasks)} advanced batch tasks")

    # Create batch file
    file_path = processor.create_batch_file(tasks, "batch_advanced.jsonl")
    print(f"📄 Advanced batch file: {file_path}")

    # Show file contents
    print(f"\n📋 Sample batch file content:")
    with open(file_path, "r") as f:
        first_line = f.readline()
        sample_task = json.loads(first_line)
        print(f"   Custom ID: {sample_task['custom_id']}")
        print(f"   Model: {sample_task['body']['model']}")
        print(f"   Temperature: {sample_task['body'].get('temperature', 'default')}")

    return file_path


def demo_result_processing():
    """Demo how to process batch results when available."""
    print("\n📊 DEMO 5: Result Processing")
    print("=" * 60)

    # Simulate a completed batch result
    sample_results = [
        {
            "custom_id": "movie-0",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": '{"categories": ["crime", "drama"], "summary": "An aging crime boss transfers power to his reluctant son."}'
                            }
                        }
                    ]
                }
            },
        },
        {
            "custom_id": "movie-1",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": '{"categories": ["sci-fi", "thriller"], "summary": "A thief plants ideas in dreams using advanced technology."}'
                            }
                        }
                    ]
                }
            },
        },
    ]

    print("📋 Sample batch results processing:")
    for result in sample_results:
        custom_id = result["custom_id"]
        content = result["response"]["body"]["choices"][0]["message"]["content"]

        # Parse JSON result
        try:
            parsed = json.loads(content)
            categories = parsed.get("categories", [])
            summary = parsed.get("summary", "")

            print(f"\n   {custom_id}:")
            print(f"      Categories: {', '.join(categories)}")
            print(f"      Summary: {summary}")

        except json.JSONDecodeError:
            print(f"   {custom_id}: Failed to parse JSON result")


def main():
    """Run all batch API demos."""
    print("🚀 OpenAI Batch API Demonstration")
    print("=" * 80)
    print(
        "Demonstrating async batch processing for cost-effective large-scale AI tasks"
    )
    print("⚠️ Note: Actual batch jobs take up to 24h to complete")
    print("=" * 80)

    try:
        # Run demos
        batch_id = demo_movie_categorization()
        demo_image_captioning()
        demo_batch_management()
        demo_advanced_batch_creation()
        demo_result_processing()

        print("\n✅ Batch API demos completed!")
        print("\n💡 Key Benefits:")
        print("   ✅ Lower costs compared to real-time API")
        print("   ✅ Higher rate limits for batch processing")
        print("   ✅ Supports all Chat Completions parameters")
        print("   ✅ Vision capabilities for image processing")
        print("   ✅ JSON structured outputs")

        print("\n🔄 Typical Workflow:")
        print("   1. Create BatchTask objects")
        print("   2. Generate JSONL batch file")
        print("   3. Upload file to OpenAI")
        print("   4. Submit batch job")
        print("   5. Monitor status (polling)")
        print("   6. Download results when completed")

        print(f"\n📊 Example batch job submitted: {batch_id}")
        print("   Check status with: processor.get_batch_status(batch_id)")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("   Make sure OPENAI_API_KEY is set in .env file")


if __name__ == "__main__":
    main()
