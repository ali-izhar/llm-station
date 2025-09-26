#!/usr/bin/env python3
"""
Unit tests for batch processors using mocks.
Tests batch functionality without making actual API calls.
"""

import pytest
from unittest.mock import Mock, patch, mock_open

from llm_station.batch import (
    OpenAIBatchProcessor,
    GoogleBatchProcessor,
    AnthropicBatchProcessor,
    BatchTask,
    GoogleBatchTask,
    AnthropicBatchRequest,
)
from llm_station.schemas.messages import UserMessage, SystemMessage


class TestOpenAIBatchProcessor:
    """Test OpenAI batch processor functionality."""

    def test_processor_creation(self):
        """Test batch processor creation."""
        processor = OpenAIBatchProcessor(api_key="test")
        assert processor.api_key == "test"

    def test_task_creation(self):
        """Test batch task creation."""
        processor = OpenAIBatchProcessor(api_key="test")

        task = processor.create_task(
            custom_id="test-1",
            model="gpt-4o-mini",
            messages=[UserMessage("Test message")],
            temperature=0.1,
        )

        assert isinstance(task, BatchTask)
        assert task.custom_id == "test-1"
        assert task.model == "gpt-4o-mini"
        assert task.temperature == 0.1

    def test_batch_file_creation(self):
        """Test batch file creation."""
        processor = OpenAIBatchProcessor(api_key="test")

        tasks = [
            processor.create_task("test-1", "gpt-4o-mini", [UserMessage("Test 1")]),
            processor.create_task("test-2", "gpt-4o-mini", [UserMessage("Test 2")]),
        ]

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.mkdir"):
                file_path = processor.create_batch_file(tasks, "test_batch.jsonl")

        assert file_path == "test_batch.jsonl"
        mock_file.assert_called_once()

    def test_text_batch_processing(self):
        """Test convenience text batch processing."""
        processor = OpenAIBatchProcessor(api_key="test")

        texts = ["Text 1", "Text 2", "Text 3"]

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.mkdir"):
                file_path = processor.process_text_batch(
                    texts=texts,
                    system_prompt="Test system prompt",
                    model="gpt-4o-mini",
                    temperature=0.1,
                )

        assert file_path.endswith(".jsonl")


class TestGoogleBatchProcessor:
    """Test Google batch processor functionality."""

    def test_processor_creation(self):
        """Test Google batch processor creation."""
        processor = GoogleBatchProcessor(api_key="test")
        assert processor.api_key == "test"

    def test_task_creation(self):
        """Test Google batch task creation."""
        processor = GoogleBatchProcessor(api_key="test")

        task = processor.create_task(
            key="test-1",
            model="gemini-2.5-flash",
            contents="Test message",
            system_instruction="Test system",
        )

        assert isinstance(task, GoogleBatchTask)
        assert task.key == "test-1"
        assert task.model == "gemini-2.5-flash"
        assert task.system_instruction == "Test system"

    def test_task_to_request_conversion(self):
        """Test task to request format conversion."""
        processor = GoogleBatchProcessor(api_key="test")

        task = processor.create_task(
            key="test-1", model="gemini-2.5-flash", contents="Test message"
        )

        request = processor._task_to_request(task)

        assert request["key"] == "test-1"
        assert "request" in request
        # Google batch request should have correct format
        assert request["key"] == "test-1"
        assert "request" in request
        assert "contents" in request["request"]


class TestAnthropicBatchProcessor:
    """Test Anthropic batch processor functionality."""

    def test_processor_creation(self):
        """Test Anthropic batch processor creation."""
        processor = AnthropicBatchProcessor(api_key="test")
        assert processor.api_key == "test"

    def test_request_creation(self):
        """Test Anthropic batch request creation."""
        processor = AnthropicBatchProcessor(api_key="test")

        request = processor.create_request(
            custom_id="test-1",
            model="claude-sonnet-4-20250514",
            messages=[UserMessage("Test message")],
            max_tokens=1024,
            system="Test system",
        )

        assert isinstance(request, AnthropicBatchRequest)
        assert request.custom_id == "test-1"
        assert request.model == "claude-sonnet-4-20250514"
        assert request.max_tokens == 1024
        assert request.system == "Test system"

    def test_text_batch_processing(self):
        """Test Anthropic text batch processing."""
        processor = AnthropicBatchProcessor(api_key="test")

        texts = ["Text 1", "Text 2"]

        with patch.object(processor, "create_batch_job") as mock_create:
            mock_create.return_value = Mock(id="batch_123")

            result = processor.process_text_batch(
                texts=texts,
                system_prompt="Test system",
                model="claude-sonnet-4-20250514",
            )

            mock_create.assert_called_once()


class TestBatchToolIntegration:
    """Test batch processors with smart tools."""

    def test_anthropic_batch_with_smart_tools(self):
        """Test Anthropic batch with smart tools (already updated)."""
        processor = AnthropicBatchProcessor(api_key="test")

        # The research batch function should use smart tools
        with patch.object(processor, "create_batch_job") as mock_create:
            mock_create.return_value = Mock(id="batch_123")

            from llm_station.batch.anthropic_batch import create_literary_analysis_batch

            texts = ["Text 1", "Text 2"]
            result = create_literary_analysis_batch(texts, processor)

            mock_create.assert_called_once()

    def test_batch_processors_import_correctly(self):
        """Test that all batch processors can be imported."""
        # Test imports work
        assert OpenAIBatchProcessor is not None
        assert GoogleBatchProcessor is not None
        assert AnthropicBatchProcessor is not None

        # Test they can be instantiated
        processors = [
            OpenAIBatchProcessor(api_key="test"),
            GoogleBatchProcessor(api_key="test"),
            AnthropicBatchProcessor(api_key="test"),
        ]

        for processor in processors:
            assert hasattr(processor, "api_key")
            assert processor.api_key == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
