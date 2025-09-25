#!/usr/bin/env python3

"""
Anthropic Claude Message Batches API Implementation.
- Async batch message processing
- 50% cost savings compared to standard API
- Up to 24-hour completion window with high throughput
- Support for all Messages API features including tools
- Prompt caching integration for additional savings
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..schemas.messages import Message, UserMessage, SystemMessage
from ..schemas.tooling import ToolSpec


class AnthropicBatchStatus(Enum):
    """Anthropic Message Batch processing status values."""

    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    CANCELED = "canceled"
    EXPIRED = "expired"


class AnthropicBatchResultType(Enum):
    """Anthropic Message Batch result types."""

    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELED = "canceled"
    EXPIRED = "expired"


@dataclass
class AnthropicBatchRequest:
    """Single request for Anthropic batch processing."""

    custom_id: str
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    temperature: Optional[float] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnthropicBatchResult:
    """Result from a completed Anthropic batch request."""

    custom_id: str
    result_type: AnthropicBatchResultType
    message: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class AnthropicBatchJob:
    """Anthropic Message Batch job information and management."""

    id: str
    processing_status: AnthropicBatchStatus
    request_counts: Dict[str, int]
    created_at: str
    ended_at: Optional[str] = None
    expires_at: Optional[str] = None
    cancel_initiated_at: Optional[str] = None
    results_url: Optional[str] = None


class AnthropicBatchProcessor:
    """Anthropic Message Batches API processor for async batch operations."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize batch processor.

        Args:
            api_key: Anthropic API key (if not provided, uses environment variable)
        """
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy-loaded Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    default_headers={"anthropic-version": "2023-06-01"},
                )
            except ImportError:
                raise RuntimeError(
                    "Anthropic SDK not installed. Install with: pip install anthropic"
                )
        return self._client

    def create_request(
        self,
        custom_id: str,
        model: str,
        messages: Union[List[Message], List[Dict[str, Any]]],
        max_tokens: int = 1024,
        system: Optional[Union[str, List[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> AnthropicBatchRequest:
        """Create a single batch request.

        Args:
            custom_id: Unique identifier for this request
            model: Claude model to use
            messages: List of Message objects or dict format
            max_tokens: Maximum tokens for response
            system: Optional system prompt (string or structured)
            **kwargs: Additional parameters (temperature, tools, etc.)

        Returns:
            AnthropicBatchRequest object ready for batch processing

        Examples:
            # Simple text request
            request = processor.create_request(
                custom_id="analysis-1",
                model="claude-sonnet-4-20250514",
                messages=[UserMessage("Analyze this data...")],
                max_tokens=2048
            )

            # Request with tools
            request = processor.create_request(
                custom_id="research-1",
                model="claude-opus-4-1-20250805",
                messages=[UserMessage("Research renewable energy trends")],
                tools=["anthropic_web_search"],
                temperature=0.7
            )
        """
        # Convert dict messages to Message objects if needed
        if messages and isinstance(messages[0], dict):
            message_objects = []
            for msg in messages:
                if msg["role"] == "system":
                    # System messages handled separately in Anthropic
                    continue
                elif msg["role"] == "user":
                    message_objects.append(UserMessage(msg["content"]))
                # Add other message types as needed
            messages = message_objects

        return AnthropicBatchRequest(
            custom_id=custom_id,
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            **kwargs,
        )

    def create_batch_job(
        self, requests: List[AnthropicBatchRequest]
    ) -> AnthropicBatchJob:
        """Create a Message Batch job.

        Args:
            requests: List of AnthropicBatchRequest objects

        Returns:
            AnthropicBatchJob object with job information
        """
        # Convert requests to API format
        api_requests = []
        for req in requests:
            # Build params object
            params = {
                "model": req.model,
                "max_tokens": req.max_tokens,
                "messages": self._convert_messages(req.messages),
            }

            # Add optional parameters
            if req.system:
                params["system"] = req.system
            if req.temperature is not None:
                params["temperature"] = req.temperature
            if req.tools:
                params["tools"] = self._convert_tools(req.tools)
            if req.tool_choice:
                params["tool_choice"] = req.tool_choice
            if req.metadata:
                params["metadata"] = req.metadata

            api_requests.append({"custom_id": req.custom_id, "params": params})

        # Create batch
        batch_response = self.client.messages.batches.create(requests=api_requests)
        return self._response_to_batch_job(batch_response)

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to Anthropic API format."""
        api_messages = []
        for msg in messages:
            if msg.role != "system":  # System handled separately
                api_messages.append({"role": msg.role, "content": msg.content})
        return api_messages

    def _convert_tools(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        """Convert ToolSpec objects to Anthropic API format."""
        api_tools = []
        for tool in tools:
            if tool.provider == "anthropic":
                # Server tools
                tool_def = {"type": tool.provider_type, "name": tool.name}
                if tool.provider_config:
                    tool_def.update(tool.provider_config)
                api_tools.append(tool_def)
            else:
                # Local function tools
                api_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                )
        return api_tools

    def get_batch_status(self, batch_id: str) -> AnthropicBatchJob:
        """Get current status of a batch job.

        Args:
            batch_id: ID of the batch job

        Returns:
            Updated AnthropicBatchJob object with current status
        """
        batch_response = self.client.messages.batches.retrieve(batch_id)
        return self._response_to_batch_job(batch_response)

    def cancel_batch_job(self, batch_id: str) -> AnthropicBatchJob:
        """Cancel a batch job.

        Args:
            batch_id: ID of the batch job to cancel

        Returns:
            AnthropicBatchJob object with cancellation status
        """
        batch_response = self.client.messages.batches.cancel(batch_id)
        return self._response_to_batch_job(batch_response)

    def list_batch_jobs(self, limit: int = 20) -> List[AnthropicBatchJob]:
        """List recent batch jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of AnthropicBatchJob objects
        """
        batches_response = self.client.messages.batches.list(limit=limit)
        return [self._response_to_batch_job(batch) for batch in batches_response.data]

    def download_results(
        self, batch_job: AnthropicBatchJob
    ) -> List[AnthropicBatchResult]:
        """Download and parse batch job results.

        Args:
            batch_job: Completed AnthropicBatchJob object

        Returns:
            List of AnthropicBatchResult objects

        Raises:
            ValueError: If batch job is not completed
        """
        if batch_job.processing_status != AnthropicBatchStatus.ENDED:
            raise ValueError(
                f"Batch job not completed. Status: {batch_job.processing_status.value}"
            )

        if not batch_job.results_url:
            raise ValueError("No results URL available")

        # Stream results for memory efficiency
        results = []
        for result in self.client.messages.batches.results(batch_job.id):
            result_type = AnthropicBatchResultType(result.result.type)

            batch_result = AnthropicBatchResult(
                custom_id=result.custom_id, result_type=result_type
            )

            if result_type == AnthropicBatchResultType.SUCCEEDED:
                batch_result.message = (
                    result.result.message.model_dump()
                    if hasattr(result.result.message, "model_dump")
                    else result.result.message
                )
            elif result_type == AnthropicBatchResultType.ERRORED:
                batch_result.error = (
                    result.result.error.model_dump()
                    if hasattr(result.result.error, "model_dump")
                    else result.result.error
                )

            results.append(batch_result)

        return results

    def wait_for_completion(
        self, batch_id: str, poll_interval: int = 300, timeout: Optional[int] = None
    ) -> AnthropicBatchJob:
        """Wait for batch job completion with polling.

        Args:
            batch_id: ID of the batch job
            poll_interval: Seconds between status checks (default: 300)
            timeout: Maximum time to wait in seconds (default: 24 hours)

        Returns:
            Completed AnthropicBatchJob object

        Raises:
            TimeoutError: If timeout is reached
            RuntimeError: If batch job fails
        """
        if timeout is None:
            timeout = 24 * 60 * 60  # 24 hours default

        start_time = time.time()

        while True:
            batch_job = self.get_batch_status(batch_id)

            if batch_job.processing_status == AnthropicBatchStatus.ENDED:
                return batch_job
            elif batch_job.processing_status in [
                AnthropicBatchStatus.CANCELED,
                AnthropicBatchStatus.EXPIRED,
            ]:
                raise RuntimeError(
                    f"Batch job {batch_id} ended with status: {batch_job.processing_status.value}"
                )

            # Check timeout
            if (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Batch job {batch_id} did not complete within {timeout} seconds"
                )

            # Wait before next poll
            time.sleep(poll_interval)

    def _response_to_batch_job(self, response: Any) -> AnthropicBatchJob:
        """Convert Anthropic batch response to AnthropicBatchJob object."""
        # Handle both dict and object responses
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = response

        # Parse status
        status = AnthropicBatchStatus(data["processing_status"])

        return AnthropicBatchJob(
            id=data["id"],
            processing_status=status,
            request_counts=data.get("request_counts", {}),
            created_at=data["created_at"],
            ended_at=data.get("ended_at"),
            expires_at=data.get("expires_at"),
            cancel_initiated_at=data.get("cancel_initiated_at"),
            results_url=data.get("results_url"),
        )

    # Convenience methods for common use cases

    def process_text_batch(
        self,
        texts: List[str],
        system_prompt: str,
        model: str,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AnthropicBatchJob:
        """Process a batch of text inputs with the same system prompt.

        Args:
            texts: List of text inputs to process
            system_prompt: System prompt to apply to all inputs
            model: Claude model to use
            max_tokens: Maximum tokens per response
            **kwargs: Additional parameters

        Returns:
            AnthropicBatchJob object

        Example:
            batch_job = processor.process_text_batch(
                texts=["Text 1", "Text 2", "Text 3"],
                system_prompt="Summarize this text in one sentence",
                model="claude-sonnet-4-20250514",
                max_tokens=100
            )
        """
        requests = []
        for i, text in enumerate(texts):
            request = self.create_request(
                custom_id=f"text-{i}",
                model=model,
                messages=[UserMessage(text)],
                max_tokens=max_tokens,
                system=system_prompt,
                **kwargs,
            )
            requests.append(request)

        return self.create_batch_job(requests)

    def process_research_batch(
        self,
        topics: List[str],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
    ) -> AnthropicBatchJob:
        """Process a batch of research topics with web search.

        Args:
            topics: List of research topics
            model: Claude model to use
            max_tokens: Maximum tokens per response

        Returns:
            AnthropicBatchJob object
        """
        from ..tools.web_search.anthropic import AnthropicWebSearch

        search_tool = AnthropicWebSearch(max_uses=3)
        requests = []

        for i, topic in enumerate(topics):
            request = self.create_request(
                custom_id=f"research-{i}",
                model=model,
                messages=[
                    UserMessage(
                        f"Research and provide a comprehensive analysis of: {topic}"
                    )
                ],
                max_tokens=max_tokens,
                system="You are a research analyst. Provide detailed analysis with citations.",
                tools=[search_tool.spec()],
                temperature=0.2,
            )
            requests.append(request)

        return self.create_batch_job(requests)

    def process_content_analysis_batch(
        self,
        content_items: List[str],
        analysis_type: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ) -> AnthropicBatchJob:
        """Process a batch of content for analysis.

        Args:
            content_items: List of content to analyze
            analysis_type: Type of analysis (sentiment, category, summary, etc.)
            model: Claude model to use
            max_tokens: Maximum tokens per response

        Returns:
            AnthropicBatchJob object
        """
        system_prompts = {
            "sentiment": "Analyze the sentiment of the content. Classify as positive, negative, or neutral with confidence.",
            "category": "Categorize the content into relevant topics and themes.",
            "summary": "Provide a concise summary highlighting key points.",
            "keywords": "Extract the most important keywords and phrases.",
        }

        system_prompt = system_prompts.get(
            analysis_type, f"Perform {analysis_type} analysis on the content."
        )

        requests = []
        for i, content in enumerate(content_items):
            request = self.create_request(
                custom_id=f"{analysis_type}-{i}",
                model=model,
                messages=[UserMessage(content)],
                max_tokens=max_tokens,
                system=system_prompt,
                temperature=0.1,
            )
            requests.append(request)

        return self.create_batch_job(requests)

    # Full workflow methods

    def submit_and_wait(
        self,
        requests: List[AnthropicBatchRequest],
        poll_interval: int = 300,
        timeout: Optional[int] = None,
    ) -> List[AnthropicBatchResult]:
        """Complete workflow: submit batch and wait for results.

        Args:
            requests: List of AnthropicBatchRequest objects
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time (default: 24 hours)

        Returns:
            List of AnthropicBatchResult objects
        """
        # Submit batch
        batch_job = self.create_batch_job(requests)
        print(f"Batch submitted: {batch_job.id}")

        # Wait for completion
        completed_job = self.wait_for_completion(batch_job.id, poll_interval, timeout)
        print(f"Batch completed: {completed_job.processing_status.value}")

        # Download results
        return self.download_results(completed_job)


# Convenience functions for common patterns


def create_literary_analysis_batch(
    texts: List[str],
    processor: AnthropicBatchProcessor,
    model: str = "claude-opus-4-1-20250805",
) -> AnthropicBatchJob:
    """Create a batch for literary text analysis.

    Args:
        texts: List of literary texts to analyze
        processor: AnthropicBatchProcessor instance
        model: Claude model to use

    Returns:
        AnthropicBatchJob object
    """
    system_prompt = """
You are an AI assistant tasked with analyzing literary works. Your goal is to provide 
insightful commentary on themes, characters, and writing style. For each text:

1. Identify major themes and their development
2. Analyze key characters and their relationships
3. Comment on writing style and literary techniques
4. Provide overall assessment and significance

Be thorough but concise in your analysis.
"""

    requests = []
    for i, text in enumerate(texts):
        request = processor.create_request(
            custom_id=f"literary-{i}",
            model=model,
            messages=[UserMessage(text)],
            max_tokens=2048,
            system=system_prompt,
            temperature=0.3,
        )
        requests.append(request)

    return processor.create_batch_job(requests)


def create_code_analysis_batch(
    code_samples: List[str],
    processor: AnthropicBatchProcessor,
    model: str = "claude-sonnet-4-20250514",
) -> AnthropicBatchJob:
    """Create a batch for code analysis with execution.

    Args:
        code_samples: List of code samples to analyze
        processor: AnthropicBatchProcessor instance
        model: Claude model to use

    Returns:
        AnthropicBatchJob object
    """
    from ..tools.code_execution.anthropic import AnthropicCodeExecution

    code_tool = AnthropicCodeExecution()
    requests = []

    for i, code in enumerate(code_samples):
        request = processor.create_request(
            custom_id=f"code-{i}",
            model=model,
            messages=[
                UserMessage(
                    f"Analyze this code and test its functionality:\n\n```\n{code}\n```"
                )
            ],
            max_tokens=3072,
            system="You are a code review expert. Analyze code quality, test functionality, and suggest improvements.",
            tools=[code_tool.spec()],
            temperature=0.1,
        )
        requests.append(request)

    return processor.create_batch_job(requests)
