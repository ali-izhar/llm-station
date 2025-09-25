#!/usr/bin/env python3

"""
Google Gemini Batch API Implementation.
- Async batch job creation and management
- 50% cost savings compared to standard API
- 24-hour completion window with high throughput
- File upload/download management
- Support for both file-based and inline requests
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..schemas.messages import Message
from ..schemas.tooling import ToolSpec


class GoogleBatchStatus(Enum):
    """Google Batch job status values."""

    JOB_STATE_UNSPECIFIED = "JOB_STATE_UNSPECIFIED"
    JOB_STATE_PENDING = "JOB_STATE_PENDING"
    JOB_STATE_RUNNING = "JOB_STATE_RUNNING"
    JOB_STATE_SUCCEEDED = "JOB_STATE_SUCCEEDED"
    JOB_STATE_FAILED = "JOB_STATE_FAILED"
    JOB_STATE_CANCELLED = "JOB_STATE_CANCELLED"


@dataclass
class GoogleBatchTask:
    """Single task for Google batch processing."""

    key: str  # Required for file-based jobs (correlation ID)
    model: str
    contents: List[Dict[str, Any]]  # Gemini content format
    generation_config: Optional[Dict[str, Any]] = None
    tools: Optional[List[ToolSpec]] = None
    system_instruction: Optional[str] = None


@dataclass
class GoogleBatchResult:
    """Result from a completed Google batch task."""

    key: Optional[str] = None  # Only for file-based jobs
    response: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class GoogleBatchJob:
    """Google Batch job information and management."""

    name: str
    display_name: Optional[str]
    state: GoogleBatchStatus
    model: str
    create_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[str] = None

    # File-based job details
    src_file_name: Optional[str] = None
    dest_file_name: Optional[str] = None

    # Inline job details
    inlined_responses: Optional[List[GoogleBatchResult]] = None

    # Job metrics
    request_count: Optional[int] = None
    completed_request_count: Optional[int] = None


class GoogleBatchProcessor:
    """Google Gemini Batch API processor for async batch operations."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize batch processor.

        Args:
            api_key: Google API key (if not provided, uses environment variable)
        """
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy-loaded Google GenAI client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "Google GenAI SDK not installed. Install with: pip install -U google-genai"
                )
        return self._client

    def create_task(
        self,
        key: str,
        model: str,
        contents: Union[List[Message], List[Dict[str, Any]], str],
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> GoogleBatchTask:
        """Create a single batch task.

        Args:
            key: Unique identifier for this task (required for file-based jobs)
            model: Google model to use (e.g., gemini-2.5-flash, gemini-1.5-pro)
            contents: Content as Message objects, dict format, or simple string
            system_instruction: Optional system instruction
            **kwargs: Additional generation parameters

        Returns:
            GoogleBatchTask object ready for batch processing

        Examples:
            # Simple text task
            task = processor.create_task(
                key="task-1",
                model="gemini-2.5-flash",
                contents="Explain quantum computing in simple terms",
                generation_config={"temperature": 0.1}
            )

            # Multimodal task with image
            task = processor.create_task(
                key="image-task-1",
                model="gemini-2.5-flash",
                contents=[
                    {"text": "Describe this image in detail"},
                    {"file_data": {"file_uri": "gs://bucket/image.jpg", "mime_type": "image/jpeg"}}
                ]
            )
        """
        # Convert different content formats to Gemini format
        gemini_contents = self._convert_contents(contents)

        # Extract generation config
        generation_config = kwargs.pop("generation_config", None)
        if not generation_config and kwargs:
            # Build generation config from common parameters
            gen_config = {}
            if "temperature" in kwargs:
                gen_config["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                gen_config["max_output_tokens"] = kwargs.pop("max_tokens")
            if "top_p" in kwargs:
                gen_config["top_p"] = kwargs.pop("top_p")
            if gen_config:
                generation_config = gen_config

        return GoogleBatchTask(
            key=key,
            model=model,
            contents=gemini_contents,
            generation_config=generation_config,
            system_instruction=system_instruction,
            **kwargs,
        )

    def _convert_contents(
        self, contents: Union[List[Message], List[Dict[str, Any]], str]
    ) -> List[Dict[str, Any]]:
        """Convert various content formats to Gemini API format."""
        if isinstance(contents, str):
            # Simple string - convert to Gemini format
            return [{"parts": [{"text": contents}]}]

        elif isinstance(contents, list):
            if contents and isinstance(contents[0], Message):
                # Message objects - convert to Gemini format
                gemini_contents = []
                for msg in contents:
                    if msg.role != "system":  # System handled separately
                        gemini_contents.append(
                            {"role": msg.role, "parts": [{"text": msg.content}]}
                        )
                return gemini_contents
            else:
                # Already in dict format - validate and return
                return contents

        else:
            raise ValueError(f"Unsupported contents format: {type(contents)}")

    def create_batch_file(
        self, tasks: List[GoogleBatchTask], file_path: Optional[str] = None
    ) -> str:
        """Create JSONL batch file from tasks.

        Args:
            tasks: List of GoogleBatchTask objects
            file_path: Optional file path (auto-generated if not provided)

        Returns:
            Path to created batch file

        Format:
            Each line contains a JSON object:
            {
                "key": "task-1",
                "request": {
                    "contents": [...],
                    "generation_config": {...},
                    "system_instruction": "..."
                }
            }
        """
        if not file_path:
            timestamp = int(time.time())
            file_path = f"google_batch_tasks_{timestamp}.jsonl"

        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            for task in tasks:
                # Convert task to API request format
                request = self._task_to_request(task)
                file.write(json.dumps(request) + "\n")

        return file_path

    def _task_to_request(self, task: GoogleBatchTask) -> Dict[str, Any]:
        """Convert GoogleBatchTask to API request format."""
        request_body = {"contents": task.contents}

        # Add generation config if provided
        if task.generation_config:
            request_body["generation_config"] = task.generation_config

        # Add system instruction if provided
        if task.system_instruction:
            request_body["system_instruction"] = task.system_instruction

        # Add tools if provided (convert ToolSpec to Gemini format)
        if task.tools:
            tools_list = []
            for tool in task.tools:
                if tool.provider == "google":
                    if tool.provider_type == "google_search":
                        tools_list.append({"google_search": {}})
                    elif tool.provider_type == "code_execution":
                        tools_list.append({"code_execution": {}})
                    elif tool.provider_type == "url_context":
                        tools_list.append({"url_context": {}})
                else:
                    # Local function tool
                    tools_list.append(
                        {
                            "function_declarations": [
                                {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.input_schema,
                                }
                            ]
                        }
                    )

            if tools_list:
                request_body["tools"] = tools_list

        return {"key": task.key, "request": request_body}

    def upload_batch_file(
        self, file_path: str, display_name: Optional[str] = None
    ) -> str:
        """Upload batch file to Google Files API.

        Args:
            file_path: Path to JSONL batch file
            display_name: Optional display name for the file

        Returns:
            File name for the uploaded batch file
        """
        from google.genai.types import UploadFileConfig

        config = UploadFileConfig(mime_type="application/json")
        if display_name:
            config.display_name = display_name

        with open(file_path, "rb") as file:
            batch_file = self.client.files.upload(file=file, config=config)

        return batch_file.name

    def create_batch_job(
        self,
        model: str,
        src: Union[str, List[Dict[str, Any]]],
        display_name: Optional[str] = None,
        **config_kwargs: Any,
    ) -> GoogleBatchJob:
        """Create a batch job.

        Args:
            model: Gemini model to use
            src: Either file name (for file-based) or list of requests (for inline)
            display_name: Optional display name for the job
            **config_kwargs: Additional job configuration

        Returns:
            GoogleBatchJob object with job information
        """
        config = {"display_name": display_name} if display_name else {}
        config.update(config_kwargs)

        if isinstance(src, str):
            # File-based batch job
            batch_response = self.client.batches.create(
                model=model, src=src, config=config
            )
        else:
            # Inline batch job
            batch_response = self.client.batches.create(
                model=model, src=src, config=config
            )

        return self._response_to_batch_job(batch_response)

    def get_batch_status(self, batch_name: str) -> GoogleBatchJob:
        """Get current status of a batch job.

        Args:
            batch_name: Name of the batch job (e.g., "batches/...")

        Returns:
            Updated GoogleBatchJob object with current status
        """
        batch_response = self.client.batches.get(name=batch_name)
        return self._response_to_batch_job(batch_response)

    def cancel_batch_job(self, batch_name: str) -> GoogleBatchJob:
        """Cancel a batch job.

        Args:
            batch_name: Name of the batch job to cancel

        Returns:
            GoogleBatchJob object with cancellation status
        """
        batch_response = self.client.batches.cancel(name=batch_name)
        return self._response_to_batch_job(batch_response)

    def list_batch_jobs(self, page_size: int = 20) -> List[GoogleBatchJob]:
        """List recent batch jobs.

        Args:
            page_size: Maximum number of jobs to return

        Returns:
            List of GoogleBatchJob objects
        """
        batches_response = self.client.batches.list(config={"page_size": page_size})
        return [self._response_to_batch_job(batch) for batch in batches_response.page]

    def download_results(
        self, batch_job: GoogleBatchJob, output_file_path: Optional[str] = None
    ) -> List[GoogleBatchResult]:
        """Download and parse batch job results.

        Args:
            batch_job: Completed GoogleBatchJob object
            output_file_path: Optional path to save results file

        Returns:
            List of GoogleBatchResult objects

        Raises:
            ValueError: If batch job is not completed
        """
        if batch_job.state != GoogleBatchStatus.JOB_STATE_SUCCEEDED:
            raise ValueError(
                f"Batch job not completed. Status: {batch_job.state.value}"
            )

        # Handle inline responses
        if batch_job.inlined_responses:
            return batch_job.inlined_responses

        # Handle file-based responses
        if not batch_job.dest_file_name:
            raise ValueError("No output file or inline responses available")

        # Download results file
        result_content = self.client.files.download(file=batch_job.dest_file_name)

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
                    GoogleBatchResult(
                        key=result_data.get("key"),
                        response=result_data.get("response"),
                        error=result_data.get("error"),
                    )
                )

        return results

    def wait_for_completion(
        self, batch_name: str, poll_interval: int = 60, timeout: Optional[int] = None
    ) -> GoogleBatchJob:
        """Wait for batch job completion with polling.

        Args:
            batch_name: Name of the batch job
            poll_interval: Seconds between status checks (default: 60)
            timeout: Maximum time to wait in seconds (default: no timeout)

        Returns:
            Completed GoogleBatchJob object

        Raises:
            TimeoutError: If timeout is reached
            RuntimeError: If batch job fails
        """
        start_time = time.time()

        while True:
            batch_job = self.get_batch_status(batch_name)

            if batch_job.state == GoogleBatchStatus.JOB_STATE_SUCCEEDED:
                return batch_job
            elif batch_job.state in [
                GoogleBatchStatus.JOB_STATE_FAILED,
                GoogleBatchStatus.JOB_STATE_CANCELLED,
            ]:
                raise RuntimeError(
                    f"Batch job {batch_name} ended with status: {batch_job.state.value}"
                )

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Batch job {batch_name} did not complete within {timeout} seconds"
                )

            # Wait before next poll
            time.sleep(poll_interval)

    def _response_to_batch_job(self, response: Any) -> GoogleBatchJob:
        """Convert Google batch response to GoogleBatchJob object."""
        # Handle both dict and object responses
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif hasattr(response, "__dict__"):
            data = response.__dict__
        else:
            data = response

        # Parse state
        state_name = data.get("state", {}).get("name", "JOB_STATE_UNSPECIFIED")
        state = GoogleBatchStatus(state_name)

        # Parse timestamps
        create_time = None
        start_time = None
        end_time = None

        if "create_time" in data:
            create_time = data["create_time"]
        if "start_time" in data:
            start_time = data["start_time"]
        if "end_time" in data:
            end_time = data["end_time"]

        # Parse source and destination
        src_file_name = None
        dest_file_name = None
        inlined_responses = None

        if "src" in data:
            src_data = data["src"]
            if isinstance(src_data, dict) and "file_name" in src_data:
                src_file_name = src_data["file_name"]

        if "dest" in data and data["dest"]:
            dest_data = data["dest"]
            if isinstance(dest_data, dict):
                if "file_name" in dest_data:
                    dest_file_name = dest_data["file_name"]
                if "inlined_responses" in dest_data:
                    # Parse inline responses
                    inline_data = dest_data["inlined_responses"]
                    inlined_responses = []
                    for inline_resp in inline_data:
                        inlined_responses.append(
                            GoogleBatchResult(
                                response=inline_resp.get("response"),
                                error=inline_resp.get("error"),
                            )
                        )

        return GoogleBatchJob(
            name=data.get("name", ""),
            display_name=data.get("display_name"),
            state=state,
            model=data.get("model", ""),
            create_time=create_time,
            start_time=start_time,
            end_time=end_time,
            error=data.get("error"),
            src_file_name=src_file_name,
            dest_file_name=dest_file_name,
            inlined_responses=inlined_responses,
            request_count=data.get("request_count"),
            completed_request_count=data.get("completed_request_count"),
        )

    # Convenience methods for common use cases

    def process_text_batch(
        self,
        texts: List[str],
        system_instruction: str,
        model: str,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Process a batch of text inputs with the same system instruction.

        Args:
            texts: List of text inputs to process
            system_instruction: System instruction to apply to all inputs
            model: Google model to use
            temperature: Optional temperature parameter
            **kwargs: Additional parameters

        Returns:
            Batch file path ready for upload

        Example:
            file_path = processor.process_text_batch(
                texts=["Text 1", "Text 2", "Text 3"],
                system_instruction="Summarize this text in one sentence",
                model="gemini-2.5-flash",
                temperature=0.1
            )
        """
        tasks = []
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature

        for i, text in enumerate(texts):
            task = self.create_task(
                key=f"text-{i}",
                model=model,
                contents=text,
                system_instruction=system_instruction,
                generation_config=generation_config if generation_config else None,
                **kwargs,
            )
            tasks.append(task)

        return self.create_batch_file(tasks)

    def process_multimodal_batch(
        self,
        content_pairs: List[List[Dict[str, Any]]],
        system_instruction: str,
        model: str,
        **kwargs: Any,
    ) -> str:
        """Process a batch of multimodal inputs.

        Args:
            content_pairs: List of content arrays (text + images/files per request)
            system_instruction: System instruction for all requests
            model: Google model to use (should support multimodal)
            **kwargs: Additional parameters

        Returns:
            Batch file path ready for upload

        Example:
            content_pairs = [
                [
                    {"text": "Describe this image"},
                    {"file_data": {"file_uri": "gs://bucket/img1.jpg", "mime_type": "image/jpeg"}}
                ],
                [
                    {"text": "Analyze this document"},
                    {"file_data": {"file_uri": "gs://bucket/doc1.pdf", "mime_type": "application/pdf"}}
                ]
            ]

            file_path = processor.process_multimodal_batch(
                content_pairs=content_pairs,
                system_instruction="Provide detailed analysis",
                model="gemini-2.5-flash"
            )
        """
        tasks = []
        for i, content_parts in enumerate(content_pairs):
            task = self.create_task(
                key=f"multimodal-{i}",
                model=model,
                contents=[{"parts": content_parts}],
                system_instruction=system_instruction,
                **kwargs,
            )
            tasks.append(task)

        return self.create_batch_file(tasks)

    # Full workflow methods

    def submit_batch(
        self,
        tasks: List[GoogleBatchTask],
        file_path: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> GoogleBatchJob:
        """Complete workflow: create file, upload, and submit batch job.

        Args:
            tasks: List of GoogleBatchTask objects
            file_path: Optional batch file path
            display_name: Optional job display name

        Returns:
            GoogleBatchJob object with job information
        """
        # Create batch file
        file_path = self.create_batch_file(tasks, file_path)

        # Upload file
        file_name = self.upload_batch_file(file_path, display_name)

        # Create batch job (use model from first task)
        model = tasks[0].model if tasks else "gemini-2.5-flash"
        return self.create_batch_job(
            model=model, src=file_name, display_name=display_name
        )

    def submit_inline_batch(
        self,
        tasks: List[GoogleBatchTask],
        display_name: Optional[str] = None,
    ) -> GoogleBatchJob:
        """Submit inline batch job (no file upload needed).

        Args:
            tasks: List of GoogleBatchTask objects
            display_name: Optional job display name

        Returns:
            GoogleBatchJob object with job information
        """
        # Convert tasks to inline request format
        inline_requests = []
        for task in tasks:
            request = self._task_to_request(task)
            inline_requests.append(request["request"])  # No key needed for inline

        # Create inline batch job
        model = tasks[0].model if tasks else "gemini-2.5-flash"
        return self.create_batch_job(
            model=model, src=inline_requests, display_name=display_name
        )

    def get_completed_results(
        self,
        batch_name: str,
        output_file_path: Optional[str] = None,
        wait: bool = True,
        poll_interval: int = 60,
    ) -> List[GoogleBatchResult]:
        """Get results from a batch job, waiting for completion if needed.

        Args:
            batch_name: Name of the batch job
            output_file_path: Optional path to save results
            wait: Whether to wait for completion (default: True)
            poll_interval: Polling interval in seconds

        Returns:
            List of GoogleBatchResult objects

        Raises:
            RuntimeError: If batch job fails or is not completed
        """
        if wait:
            batch_job = self.wait_for_completion(batch_name, poll_interval)
        else:
            batch_job = self.get_batch_status(batch_name)
            if batch_job.state != GoogleBatchStatus.JOB_STATE_SUCCEEDED:
                raise RuntimeError(
                    f"Batch job not completed. Status: {batch_job.state.value}"
                )

        return self.download_results(batch_job, output_file_path)

    # Specialized batch methods

    def create_embeddings_batch(
        self,
        texts: List[str],
        model: str = "gemini-embedding-001",
        output_dimensionality: Optional[int] = None,
    ) -> GoogleBatchJob:
        """Create batch job for text embeddings.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            output_dimensionality: Optional output dimension

        Returns:
            GoogleBatchJob for embeddings processing
        """
        # Create embeddings tasks
        tasks = []
        for i, text in enumerate(texts):
            request_body = {"content": {"parts": [{"text": text}]}}
            if output_dimensionality:
                request_body["output_dimensionality"] = output_dimensionality

            task_data = {"key": f"embed-{i}", "request": request_body}
            tasks.append(task_data)

        # Create and upload file
        timestamp = int(time.time())
        file_path = f"embeddings_batch_{timestamp}.jsonl"

        with open(file_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")

        # Upload and create embeddings job
        file_name = self.upload_batch_file(file_path, "embeddings-batch")

        from google.genai.types import EmbeddingsBatchJobSource

        batch_response = self.client.batches.create_embeddings(
            model=model, src=EmbeddingsBatchJobSource(file_name=file_name)
        )

        return self._response_to_batch_job(batch_response)


# Convenience functions for common patterns


def create_research_batch(
    topics: List[str],
    processor: GoogleBatchProcessor,
    model: str = "gemini-2.5-flash",
) -> str:
    """Create a batch for research topics analysis.

    Args:
        topics: List of research topics
        processor: GoogleBatchProcessor instance
        model: Google model to use for processing

    Returns:
        Batch file path
    """
    system_prompt = """
You are a research analyst. For each topic, provide:
1. Current status and recent developments
2. Key challenges and opportunities  
3. Future outlook and predictions
4. Relevant statistics and data points

Use web search to find current information and provide citations.
"""

    tasks = []
    for i, topic in enumerate(topics):
        task = processor.create_task(
            key=f"research-{i}",
            model=model,
            contents=f"Research and analyze: {topic}",
            system_instruction=system_prompt,
            generation_config={"temperature": 0.2},
        )
        tasks.append(task)

    return processor.create_batch_file(tasks)


def create_content_analysis_batch(
    content_items: List[str],
    analysis_type: str,
    processor: GoogleBatchProcessor,
    model: str = "gemini-2.5-flash",
) -> str:
    """Create a batch for content analysis.

    Args:
        content_items: List of content to analyze
        analysis_type: Type of analysis (sentiment, category, summary, etc.)
        processor: GoogleBatchProcessor instance
        model: Google model to use

    Returns:
        Batch file path
    """
    system_prompts = {
        "sentiment": "Analyze the sentiment of the following content. Classify as positive, negative, or neutral with confidence score.",
        "category": "Categorize the following content into relevant topics and themes.",
        "summary": "Provide a concise summary of the following content, highlighting key points.",
        "keywords": "Extract the most important keywords and phrases from the following content.",
    }

    system_prompt = system_prompts.get(
        analysis_type, f"Perform {analysis_type} analysis on the following content:"
    )

    tasks = []
    for i, content in enumerate(content_items):
        task = processor.create_task(
            key=f"{analysis_type}-{i}",
            model=model,
            contents=content,
            system_instruction=system_prompt,
            generation_config={"temperature": 0.1},
        )
        tasks.append(task)

    return processor.create_batch_file(tasks)


def create_image_generation_batch(
    prompts: List[str],
    processor: GoogleBatchProcessor,
    model: str = "gemini-2.5-flash-image-preview",
) -> str:
    """Create a batch for image generation.

    Args:
        prompts: List of image generation prompts
        processor: GoogleBatchProcessor instance
        model: Image generation model

    Returns:
        Batch file path
    """
    tasks = []
    for i, prompt in enumerate(prompts):
        task = processor.create_task(
            key=f"image-{i}",
            model=model,
            contents=prompt,
            generation_config={"response_modalities": ["TEXT", "IMAGE"]},
        )
        tasks.append(task)

    return processor.create_batch_file(tasks)
