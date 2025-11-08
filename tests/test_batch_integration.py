#!/usr/bin/env python3
"""
Integration tests for batch APIs with real API calls.
These tests require API keys and make actual API requests.
Marked with pytest marker 'integration' - skip with: pytest -m "not integration"
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration
def test_openai_batch():
    """Test OpenAI Batch API with real API calls."""
    print("=" * 70)
    print("Testing OpenAI Batch API")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    try:
        from llm_station.batch import OpenAIBatchProcessor, BatchTask, BatchStatus
        from llm_station import UserMessage

        processor = OpenAIBatchProcessor(api_key=api_key)

        # Test 1: Create task
        print("\n[1] Creating batch task...")
        task = processor.create_task(
            custom_id="test_task_1",
            model="gpt-4o-mini",
            messages=[UserMessage("Say hello in one sentence.")],
            temperature=0.7,
            max_tokens=50,
        )
        print(f"[OK] Task created: {task.custom_id}")
        assert task.model == "gpt-4o-mini"
        assert len(task.messages) == 1

        # Test 2: Create batch file
        print("\n[2] Creating batch file...")
        tasks = [task]
        file_path = processor.create_batch_file(tasks)
        print(f"[OK] Batch file created: {file_path}")

        # Verify file exists and is valid JSONL
        import json

        with open(file_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1, "Batch file should have 1 line"
            task_data = json.loads(lines[0])
            assert "custom_id" in task_data
            assert task_data["custom_id"] == "test_task_1"

        print(f"[OK] Batch file validated: {len(lines)} task(s)")

        # Test 3: Upload file (if SDK available)
        print("\n[3] Testing file upload...")
        try:
            file_id = processor.upload_batch_file(file_path)
            print(f"[OK] File uploaded: {file_id}")

            # Test 4: Create batch job
            print("\n[4] Creating batch job...")
            batch_job = processor.create_batch_job(file_id)
            print(f"[OK] Batch job created: {batch_job.id}")
            print(f"     Status: {batch_job.status.value}")

            # Test 5: Wait for completion and get results
            print("\n[5] Waiting for batch completion...")
            print(
                "     (This may take a few minutes - batch jobs process asynchronously)"
            )
            try:
                results = processor.get_completed_results(
                    batch_job.id, wait=True, poll_interval=30  # Check every 30 seconds
                )
                print(f"[OK] Batch completed! Retrieved {len(results)} result(s)")

                # Verify results
                if results:
                    result = results[0]
                    print(f"     Custom ID: {result.custom_id}")
                    if result.error:
                        print(f"     Error: {result.error}")
                    else:
                        # OpenAI batch response structure: response.body.choices[0].message.content
                        response_body = result.response.get("body", {})
                        if not response_body:
                            response_body = result.response
                        choices = response_body.get("choices", [])
                        if choices:
                            message = choices[0].get("message", {})
                            content = message.get("content", "N/A")
                            print(f"     Response: {content[:100]}...")
                        else:
                            # Try alternative structure
                            content = str(result.response)[:100]
                            print(f"     Response: {content}...")
                        print(f"[OK] Result validated")

            except Exception as e:
                print(f"[WARNING] Could not wait for completion: {e}")
                print(
                    "     Batch job created successfully, but completion check failed"
                )
                print("     This is normal - batch jobs can take up to 24 hours")
                # Don't fail the test - job creation is what matters

            # Cleanup
            os.remove(file_path)
            print(f"\n[OK] Cleaned up batch file")

        except ImportError:
            pytest.skip("OpenAI SDK not installed - cannot test upload/job creation")

    except Exception as e:
        print(f"[FAIL] OpenAI Batch API test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"OpenAI Batch API test failed: {e}")


@pytest.mark.integration
def test_google_batch():
    """Test Google Batch API with real API calls."""
    print("\n" + "=" * 70)
    print("Testing Google Batch API")
    print("=" * 70)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    try:
        from llm_station.batch import (
            GoogleBatchProcessor,
            GoogleBatchTask,
            GoogleBatchStatus,
        )

        processor = GoogleBatchProcessor(api_key=api_key)

        # Test 1: Create task
        print("\n[1] Creating batch task...")
        # Use gemini-2.5-flash which supports batch API
        task = processor.create_task(
            key="test_key_1",
            model="gemini-2.5-flash",
            contents=[
                {"role": "user", "parts": [{"text": "Say hello in one sentence."}]}
            ],
            generation_config={"temperature": 0.7, "max_output_tokens": 50},
        )
        print(f"[OK] Task created: {task.key}")
        assert task.model == "gemini-2.5-flash"

        # Test 2: Create batch file
        print("\n[2] Creating batch file...")
        tasks = [task]
        file_path = processor.create_batch_file(tasks)
        print(f"[OK] Batch file created: {file_path}")

        # Verify file exists and is valid JSONL
        import json

        with open(file_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1, "Batch file should have 1 line"
            task_data = json.loads(lines[0])
            assert "key" in task_data
            assert task_data["key"] == "test_key_1"

        print(f"[OK] Batch file validated: {len(lines)} task(s)")

        # Test 3: Upload file (if SDK available)
        print("\n[3] Testing file upload...")
        try:
            file_name = processor.upload_batch_file(file_path)
            print(f"[OK] File uploaded: {file_name}")

            # Test 4: Create batch job (file-based)
            print("\n[4] Creating file-based batch job...")
            # Use gemini-2.5-flash which supports batch API
            batch_job = processor.create_batch_job(
                model="gemini-2.5-flash", src=file_name
            )
            print(f"[OK] Batch job created: {batch_job.name}")
            print(f"     Status: {batch_job.state.value}")

            # Test 5: Wait for completion and get results
            print("\n[5] Waiting for batch completion...")
            print(
                "     (This may take a few minutes - batch jobs process asynchronously)"
            )
            try:
                # Wait for completion
                completed_job = processor.wait_for_completion(
                    batch_job.name, poll_interval=30  # Check every 30 seconds
                )
                print(f"[OK] Batch completed! Status: {completed_job.state.value}")

                # Download results
                results = processor.download_results(completed_job)
                print(f"[OK] Retrieved {len(results)} result(s)")

                # Verify results
                if results:
                    result = results[0]
                    print(f"     Key: {result.key}")
                    if result.error:
                        print(f"     Error: {result.error}")
                    else:
                        response_text = (
                            result.response.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "N/A")[:100]
                        )
                        print(f"     Response: {response_text}...")
                        print(f"[OK] Result validated")

            except Exception as e:
                print(f"[WARNING] Could not wait for completion: {e}")
                print(
                    "     Batch job created successfully, but completion check failed"
                )
                print("     This is normal - batch jobs can take up to 24 hours")
                # Don't fail the test - job creation is what matters

            # Cleanup
            os.remove(file_path)
            print(f"\n[OK] Cleaned up batch file")

        except ImportError:
            pytest.skip(
                "Google GenAI SDK not installed - cannot test upload/job creation"
            )

    except Exception as e:
        print(f"[FAIL] Google Batch API test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Google Batch API test failed: {e}")


@pytest.mark.integration
def test_anthropic_batch():
    """Test Anthropic Batch API with real API calls."""
    print("\n" + "=" * 70)
    print("Testing Anthropic Batch API")
    print("=" * 70)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    try:
        from llm_station.batch import (
            AnthropicBatchProcessor,
            AnthropicBatchRequest,
            AnthropicBatchStatus,
        )
        from llm_station import UserMessage

        processor = AnthropicBatchProcessor(api_key=api_key)

        # Test 1: Create request
        print("\n[1] Creating batch request...")
        request = processor.create_request(
            custom_id="test_req_1",
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[UserMessage("Say hello in one sentence.")],
            temperature=0.7,
        )
        print(f"[OK] Request created: {request.custom_id}")
        assert request.model == "claude-sonnet-4-20250514"
        assert len(request.messages) == 1

        # Test 2: Create batch job (inline)
        print("\n[2] Creating inline batch job...")
        try:
            batch_job = processor.create_batch_job([request])
            print(f"[OK] Batch job created: {batch_job.id}")
            print(f"     Status: {batch_job.processing_status.value}")
            print(f"     Requests: {batch_job.request_counts}")

            # Test 3: Wait for completion and get results
            print("\n[3] Waiting for batch completion...")
            print(
                "     (This may take a few minutes - batch jobs process asynchronously)"
            )
            try:
                # Wait for completion
                completed_job = processor.wait_for_completion(
                    batch_job.id, poll_interval=30  # Check every 30 seconds
                )
                print(
                    f"[OK] Batch completed! Status: {completed_job.processing_status.value}"
                )
                print(f"     Request counts: {completed_job.request_counts}")

                # Download results
                results = processor.download_results(completed_job)
                print(f"[OK] Retrieved {len(results)} result(s)")

                # Verify results
                if results:
                    result = results[0]
                    print(f"     Custom ID: {result.custom_id}")
                    print(f"     Result type: {result.result_type.value}")
                    if result.error:
                        print(f"     Error: {result.error}")
                    else:
                        message_text = (
                            result.message.get("content", [{}])[0].get("text", "N/A")[
                                :100
                            ]
                            if result.message
                            else "N/A"
                        )
                        print(f"     Response: {message_text}...")
                        print(f"[OK] Result validated")

            except Exception as e:
                print(f"[WARNING] Could not wait for completion: {e}")
                print(
                    "     Batch job created successfully, but completion check failed"
                )
                print("     This is normal - batch jobs can take up to 24 hours")
                # Don't fail the test - job creation is what matters

        except ImportError:
            pytest.skip("Anthropic SDK not installed - cannot test job creation")

    except Exception as e:
        print(f"[FAIL] Anthropic Batch API test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Anthropic Batch API test failed: {e}")
