#!/usr/bin/env python3

"""
OpenAI Batch API Implementation.
- Async batch job creation and management
- Lower costs and higher rate limits compared to regular API
- 24-hour completion window with potential faster processing
- File upload/download management
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..schemas.messages import Message, UserMessage, SystemMessage
from ..schemas.tooling import ToolSpec


class BatchStatus(Enum):
    """Batch job status values."""

    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class CompletionWindow(Enum):
    """Batch completion window options."""

    HOURS_24 = "24h"  # Standard 24-hour completion window


@dataclass
class BatchTask:
    """Single task for batch processing."""

    custom_id: str
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[str] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


@dataclass
class BatchResult:
    """Result from a completed batch task."""

    custom_id: str
    response: Dict[str, Any]
    error: Optional[Dict[str, Any]] = None


@dataclass
class BatchJob:
    """Batch job information and management."""

    id: str
    status: BatchStatus
    input_file_id: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: Optional[int] = None
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class OpenAIBatchProcessor:
    """OpenAI Batch API processor for async batch operations."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize batch processor.

        Args:
            api_key: OpenAI API key (if not provided, uses environment variable)
        """
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy-loaded OpenAI client."""
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "OpenAI SDK not installed. Install with: pip install openai"
                )
        return self._client

    def create_task(
        self,
        custom_id: str,
        model: str,
        messages: Union[List[Message], List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> BatchTask:
        """Create a single batch task.

        Args:
            custom_id: Unique identifier for this task
            model: OpenAI model to use (e.g., any supported OpenAI model)
            messages: List of messages or Message objects
            **kwargs: Additional Chat Completions parameters

        Returns:
            BatchTask object ready for batch processing

        Examples:
            # Simple text task
            task = processor.create_task(
                custom_id="movie-1",
                model="your-model",
                messages=[
                    SystemMessage("You are a movie categorizer"),
                    UserMessage("Categorize this movie: The Godfather")
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            # Vision task with image
            task = processor.create_task(
                custom_id="image-1",
                model="your-model",
                messages=[
                    SystemMessage("Caption this image briefly"),
                    UserMessage([
                        {"type": "text", "text": "Describe this furniture"},
                        {"type": "image_url", "image_url": {"url": "https://..."}}
                    ])
                ],
                max_tokens=300
            )
        """
        # Convert dict messages to Message objects if needed
        if messages and isinstance(messages[0], dict):
            message_objects = []
            for msg in messages:
                if msg["role"] == "system":
                    message_objects.append(SystemMessage(msg["content"]))
                elif msg["role"] == "user":
                    message_objects.append(UserMessage(msg["content"]))
                # Add other message types as needed
            messages = message_objects

        return BatchTask(custom_id=custom_id, model=model, messages=messages, **kwargs)

    def create_batch_file(
        self, tasks: List[BatchTask], file_path: Optional[str] = None
    ) -> str:
        """Create JSONL batch file from tasks.

        Args:
            tasks: List of BatchTask objects
            file_path: Optional file path (auto-generated if not provided)

        Returns:
            Path to created batch file

        Format:
            Each line contains a JSON object:
            {
                "custom_id": "task-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [...],
                    "temperature": 0.1,
                    ...
                }
            }
        """
        if not file_path:
            timestamp = int(time.time())
            file_path = f"batch_tasks_{timestamp}.jsonl"

        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            for task in tasks:
                # Convert task to API request format
                request = self._task_to_request(task)
                file.write(json.dumps(request) + "\n")

        return file_path

    def _task_to_request(self, task: BatchTask) -> Dict[str, Any]:
        """Convert BatchTask to API request format."""
        # Convert messages to API format
        api_messages = []
        for msg in task.messages:
            if hasattr(msg, "role"):
                # Message object
                api_messages.append({"role": msg.role, "content": msg.content})
            else:
                # Already in dict format
                api_messages.append(msg)

        # Build request body
        body = {"model": task.model, "messages": api_messages}

        # Add optional parameters
        if task.temperature is not None:
            body["temperature"] = task.temperature
        if task.max_tokens is not None:
            body["max_tokens"] = task.max_tokens
        if task.response_format is not None:
            body["response_format"] = task.response_format
        if task.tools is not None:
            # Convert ToolSpec to API format
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in task.tools
            ]
        if task.tool_choice is not None:
            body["tool_choice"] = task.tool_choice
        if task.top_p is not None:
            body["top_p"] = task.top_p
        if task.presence_penalty is not None:
            body["presence_penalty"] = task.presence_penalty
        if task.frequency_penalty is not None:
            body["frequency_penalty"] = task.frequency_penalty
        if task.logit_bias is not None:
            body["logit_bias"] = task.logit_bias
        if task.user is not None:
            body["user"] = task.user

        return {
            "custom_id": task.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def upload_batch_file(self, file_path: str) -> str:
        """Upload batch file to OpenAI.

        Args:
            file_path: Path to JSONL batch file

        Returns:
            File ID for the uploaded batch file
        """
        with open(file_path, "rb") as file:
            batch_file = self.client.files.create(file=file, purpose="batch")
        return batch_file.id

    def create_batch_job(
        self,
        input_file_id: str,
        completion_window: CompletionWindow = CompletionWindow.HOURS_24,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """Create a batch job.

        Args:
            input_file_id: ID of uploaded batch file
            completion_window: How long to allow for completion (default: 24h)
            metadata: Optional metadata to attach to the batch job

        Returns:
            BatchJob object with job information
        """
        batch_response = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window.value,
            metadata=metadata,
        )

        return self._response_to_batch_job(batch_response)

    def get_batch_status(self, batch_id: str) -> BatchJob:
        """Get current status of a batch job.

        Args:
            batch_id: ID of the batch job

        Returns:
            Updated BatchJob object with current status
        """
        batch_response = self.client.batches.retrieve(batch_id)
        return self._response_to_batch_job(batch_response)

    def cancel_batch_job(self, batch_id: str) -> BatchJob:
        """Cancel a batch job.

        Args:
            batch_id: ID of the batch job to cancel

        Returns:
            BatchJob object with cancellation status
        """
        batch_response = self.client.batches.cancel(batch_id)
        return self._response_to_batch_job(batch_response)

    def list_batch_jobs(self, limit: int = 20) -> List[BatchJob]:
        """List recent batch jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of BatchJob objects
        """
        batches_response = self.client.batches.list(limit=limit)
        return [self._response_to_batch_job(batch) for batch in batches_response.data]

    def download_results(
        self, batch_job: BatchJob, output_file_path: Optional[str] = None
    ) -> List[BatchResult]:
        """Download and parse batch job results.

        Args:
            batch_job: Completed BatchJob object
            output_file_path: Optional path to save results file

        Returns:
            List of BatchResult objects

        Raises:
            ValueError: If batch job is not completed
        """
        if batch_job.status != BatchStatus.COMPLETED:
            raise ValueError(
                f"Batch job not completed. Status: {batch_job.status.value}"
            )

        if not batch_job.output_file_id:
            raise ValueError("No output file ID available")

        # Download results
        result_content = self.client.files.content(batch_job.output_file_id).content

        # Save to file if path provided
        if output_file_path:
            Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "wb") as file:
                file.write(result_content)

        # Parse results
        results = []
        result_text = result_content.decode("utf-8")
        for line in result_text.strip().split("\n"):
            if line.strip():
                result_data = json.loads(line.strip())
                results.append(
                    BatchResult(
                        custom_id=result_data["custom_id"],
                        response=result_data.get("response", {}),
                        error=result_data.get("error"),
                    )
                )

        return results

    def download_errors(
        self, batch_job: BatchJob, error_file_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Download batch job errors if any.

        Args:
            batch_job: BatchJob object with errors
            error_file_path: Optional path to save errors file

        Returns:
            List of error objects
        """
        if not batch_job.error_file_id:
            return []

        # Download errors
        error_content = self.client.files.content(batch_job.error_file_id).content

        # Save to file if path provided
        if error_file_path:
            Path(error_file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(error_file_path, "wb") as file:
                file.write(error_content)

        # Parse errors
        errors = []
        error_text = error_content.decode("utf-8")
        for line in error_text.strip().split("\n"):
            if line.strip():
                errors.append(json.loads(line.strip()))

        return errors

    def wait_for_completion(
        self, batch_id: str, poll_interval: int = 60, timeout: Optional[int] = None
    ) -> BatchJob:
        """Wait for batch job completion with polling.

        Args:
            batch_id: ID of the batch job
            poll_interval: Seconds between status checks (default: 60)
            timeout: Maximum time to wait in seconds (default: no timeout)

        Returns:
            Completed BatchJob object

        Raises:
            TimeoutError: If timeout is reached
            RuntimeError: If batch job fails
        """
        start_time = time.time()

        while True:
            batch_job = self.get_batch_status(batch_id)

            if batch_job.status == BatchStatus.COMPLETED:
                return batch_job
            elif batch_job.status in [
                BatchStatus.FAILED,
                BatchStatus.EXPIRED,
                BatchStatus.CANCELLED,
            ]:
                raise RuntimeError(
                    f"Batch job {batch_id} ended with status: {batch_job.status.value}"
                )

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Batch job {batch_id} did not complete within {timeout} seconds"
                )

            # Wait before next poll
            time.sleep(poll_interval)

    def _response_to_batch_job(self, response: Any) -> BatchJob:
        """Convert OpenAI batch response to BatchJob object."""
        # Handle both dict and object responses
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = response

        return BatchJob(
            id=data["id"],
            status=BatchStatus(data["status"]),
            input_file_id=data["input_file_id"],
            output_file_id=data.get("output_file_id"),
            error_file_id=data.get("error_file_id"),
            created_at=data.get("created_at"),
            in_progress_at=data.get("in_progress_at"),
            expires_at=data.get("expires_at"),
            finalizing_at=data.get("finalizing_at"),
            completed_at=data.get("completed_at"),
            failed_at=data.get("failed_at"),
            expired_at=data.get("expired_at"),
            cancelling_at=data.get("cancelling_at"),
            cancelled_at=data.get("cancelled_at"),
            request_counts=data.get("request_counts"),
            metadata=data.get("metadata"),
        )

    # Convenience methods for common use cases

    def process_text_batch(
        self,
        texts: List[str],
        system_prompt: str,
        model: str,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Process a batch of text inputs with the same system prompt.

        Args:
            texts: List of text inputs to process
            system_prompt: System prompt to apply to all inputs
            model: OpenAI model to use
            response_format: Optional response format (e.g., {"type": "json_object"})
            **kwargs: Additional parameters

        Returns:
            Batch file path ready for upload

        Example:
            file_path = processor.process_text_batch(
                texts=["Text 1", "Text 2", "Text 3"],
                system_prompt="Categorize this text",
                model="your-model",
                response_format={"type": "json_object"}
            )
        """
        tasks = []
        for i, text in enumerate(texts):
            task = self.create_task(
                custom_id=f"text-{i}",
                model=model,
                messages=[SystemMessage(system_prompt), UserMessage(text)],
                response_format=response_format,
                **kwargs,
            )
            tasks.append(task)

        return self.create_batch_file(tasks)

    def process_image_batch(
        self,
        image_urls: List[str],
        texts: List[str],
        system_prompt: str,
        model: str,
        **kwargs: Any,
    ) -> str:
        """Process a batch of images with text prompts.

        Args:
            image_urls: List of image URLs
            texts: List of text prompts (same length as image_urls)
            system_prompt: System prompt for image analysis
            model: Vision-capable model (e.g., "gpt-4o-mini")
            **kwargs: Additional parameters

        Returns:
            Batch file path ready for upload

        Example:
            file_path = processor.process_image_batch(
                image_urls=["https://img1.jpg", "https://img2.jpg"],
                texts=["Furniture item 1", "Furniture item 2"],
                system_prompt="Generate a short caption for this furniture image",
                max_tokens=300
            )
        """
        if len(image_urls) != len(texts):
            raise ValueError("image_urls and texts must have the same length")

        tasks = []
        for i, (image_url, text) in enumerate(zip(image_urls, texts)):
            task = self.create_task(
                custom_id=f"image-{i}",
                model=model,
                messages=[
                    SystemMessage(system_prompt),
                    UserMessage(
                        [
                            {"type": "text", "text": text},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ]
                    ),
                ],
                **kwargs,
            )
            tasks.append(task)

        return self.create_batch_file(tasks)

    # Full workflow methods

    def submit_batch(
        self,
        tasks: List[BatchTask],
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """Complete workflow: create file, upload, and submit batch job.

        Args:
            tasks: List of BatchTask objects
            file_path: Optional batch file path
            metadata: Optional job metadata

        Returns:
            BatchJob object with job information
        """
        # Create batch file
        file_path = self.create_batch_file(tasks, file_path)

        # Upload file
        file_id = self.upload_batch_file(file_path)

        # Create batch job
        return self.create_batch_job(file_id, metadata=metadata)

    def get_completed_results(
        self,
        batch_id: str,
        output_file_path: Optional[str] = None,
        wait: bool = True,
        poll_interval: int = 60,
    ) -> List[BatchResult]:
        """Get results from a batch job, waiting for completion if needed.

        Args:
            batch_id: ID of the batch job
            output_file_path: Optional path to save results
            wait: Whether to wait for completion (default: True)
            poll_interval: Polling interval in seconds

        Returns:
            List of BatchResult objects

        Raises:
            RuntimeError: If batch job fails or is not completed
        """
        if wait:
            batch_job = self.wait_for_completion(batch_id, poll_interval)
        else:
            batch_job = self.get_batch_status(batch_id)
            if batch_job.status != BatchStatus.COMPLETED:
                raise RuntimeError(
                    f"Batch job not completed. Status: {batch_job.status.value}"
                )

        return self.download_results(batch_job, output_file_path)


# Convenience functions for common patterns


def create_movie_categorization_batch(
    movie_descriptions: List[str],
    processor: OpenAIBatchProcessor,
    model: str = "gpt-4o-mini",
) -> str:
    """Create a batch for movie categorization (example from cookbook).

    Args:
        movie_descriptions: List of movie descriptions
        processor: OpenAIBatchProcessor instance
        model: OpenAI model to use for processing

    Returns:
        Batch file path
    """
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

    return processor.process_text_batch(
        texts=movie_descriptions,
        system_prompt=system_prompt,
        model=model,
        response_format={"type": "json_object"},
        temperature=0.1,
    )


def create_image_captioning_batch(
    image_urls: List[str],
    titles: List[str],
    processor: OpenAIBatchProcessor,
    model: str = "gpt-4o-mini",
) -> str:
    """Create a batch for image captioning (example from cookbook).

    Args:
        image_urls: List of image URLs
        titles: List of item titles/names
        processor: OpenAIBatchProcessor instance
        model: OpenAI model to use for processing

    Returns:
        Batch file path
    """
    system_prompt = """
Your goal is to generate short, descriptive captions for images of items.
You will be provided with an item image and the name of that item and you will output a caption that captures the most important information about the item.
If there are multiple items depicted, refer to the name provided to understand which item you should describe.
Your generated caption should be short (1 sentence), and include only the most important information about the item.
The most important information could be: the type of item, the style (if mentioned), the material or color if especially relevant and/or any distinctive features.
Keep it short and to the point.
"""

    return processor.process_image_batch(
        image_urls=image_urls,
        texts=titles,
        system_prompt=system_prompt,
        model=model,
        temperature=0.2,
        max_tokens=300,
    )
