"""
Unit tests for logger utility module.
"""

import logging
from pathlib import Path
from unittest.mock import patch

from src.utils.logger import (
    ContextLogger,
    PerformanceLogger,
    get_context_logger,
    get_logger,
    log_performance,
    setup_logging,
)


class TestSetupLogging:
    """Test cases for setup_logging function."""

    def test_setup_logging(self) -> None:
        """Test setup_logging function."""
        # Should not raise error
        setup_logging()

    def test_setup_logging_multiple_calls(self) -> None:
        """Test setup_logging can be called multiple times."""
        setup_logging()
        setup_logging()  # Should not raise error

    def test_setup_logging_with_log_file(self, tmp_path: Path) -> None:
        """Test setup_logging with log_file parameter (covers lines 117-120)."""

        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)

        # Verify log file was created
        assert log_file.exists()

    def test_setup_logging_with_log_file_nested_path(self, tmp_path: Path) -> None:
        """Test setup_logging with log_file in nested directory (covers line 117 mkdir)."""

        log_file = tmp_path / "logs" / "nested" / "test.log"
        setup_logging(log_file=log_file)

        # Verify log file was created and parent directories were created
        assert log_file.exists()
        assert log_file.parent.exists()


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Test get_logger returns a logger instance."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_same_name(self) -> None:
        """Test get_logger with same name returns same logger."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")

        assert logger1 is logger2

    def test_get_logger_different_names(self) -> None:
        """Test get_logger with different names returns different loggers."""
        logger1 = get_logger("test_module1")
        logger2 = get_logger("test_module2")

        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_logger_logging(self) -> None:
        """Test logger actually logs messages."""
        logger = get_logger("test_module")

        with patch.object(logger, "info") as mock_info:
            logger.info("Test message")
            mock_info.assert_called_once_with("Test message")

    def test_get_logger_error_logging(self) -> None:
        """Test logger error logging."""
        logger = get_logger("test_module")

        with patch.object(logger, "error") as mock_error:
            logger.error("Error message")
            mock_error.assert_called_once_with("Error message")


class TestContextLogger:
    """Test cases for ContextLogger class."""

    def test_context_logger_with_context(self) -> None:
        """Test ContextLogger adds context to messages (lines 36-38)."""
        base_logger = get_logger("test_context_module")
        context_logger = ContextLogger(base_logger, {"ticker": "KRW-BTC", "order_id": "123"})

        # Test process method directly to verify context is added
        msg, kwargs = context_logger.process("Order placed", {})
        assert "[ticker=KRW-BTC | order_id=123]" in msg
        assert "Order placed" in msg

    def test_context_logger_without_context(self) -> None:
        """Test ContextLogger without context (line 36 if False)."""
        base_logger = get_logger("test_context_module2")
        context_logger = ContextLogger(base_logger, {})

        # Test process method directly
        msg, kwargs = context_logger.process("Simple message", {})
        assert msg == "Simple message"


class TestPerformanceLogger:
    """Test cases for PerformanceLogger class."""

    def test_performance_logger_init(self) -> None:
        """Test PerformanceLogger initialization (lines 61-64)."""
        logger = get_logger("test_module")
        perf_logger = PerformanceLogger(logger, "test_operation", ticker="KRW-BTC")

        assert perf_logger.logger is logger
        assert perf_logger.operation == "test_operation"
        assert perf_logger.context == {"ticker": "KRW-BTC"}
        assert perf_logger.start_time is None

    def test_performance_logger_with_context(self) -> None:
        """Test PerformanceLogger with context (lines 69-71, 80-82)."""
        logger = get_logger("test_module")
        perf_logger = PerformanceLogger(logger, "test_operation", ticker="KRW-BTC")

        with patch.object(logger, "debug") as mock_debug, perf_logger:
            pass

        # Check that both start and complete messages were logged with context
        assert mock_debug.call_count == 2
        start_call = mock_debug.call_args_list[0][0][0]
        complete_call = mock_debug.call_args_list[1][0][0]

        assert "[PERF] Starting test_operation [ticker=KRW-BTC]" in start_call
        assert "[PERF] Completed test_operation" in complete_call
        assert "ticker=KRW-BTC" in complete_call

    def test_performance_logger_without_context(self) -> None:
        """Test PerformanceLogger without context (lines 72-73, 83-84)."""
        logger = get_logger("test_module")
        perf_logger = PerformanceLogger(logger, "test_operation")

        with patch.object(logger, "debug") as mock_debug, perf_logger:
            pass

        # Check that messages don't include context brackets
        assert mock_debug.call_count == 2
        start_call = mock_debug.call_args_list[0][0][0]
        complete_call = mock_debug.call_args_list[1][0][0]

        assert start_call == "[PERF] Starting test_operation"
        assert "[PERF] Completed test_operation" in complete_call
        assert "[" not in complete_call or "]" not in complete_call.split(" in ")[1]


class TestGetContextLogger:
    """Test cases for get_context_logger function."""

    def test_get_context_logger(self) -> None:
        """Test get_context_logger function (lines 158-159)."""
        context_logger = get_context_logger(
            "test_context_module3", ticker="KRW-BTC", order_id="123"
        )

        assert isinstance(context_logger, ContextLogger)
        assert context_logger.extra == {"ticker": "KRW-BTC", "order_id": "123"}

        # Verify process method works correctly
        msg, kwargs = context_logger.process("Test message", {})
        assert "ticker=KRW-BTC" in msg
        assert "order_id=123" in msg
        assert "Test message" in msg


class TestLogPerformance:
    """Test cases for log_performance context manager."""

    def test_log_performance(self) -> None:
        """Test log_performance context manager (lines 176-177)."""
        logger = get_logger("test_module")

        with (
            patch.object(logger, "debug") as mock_debug,
            log_performance(logger, "test_operation", ticker="KRW-BTC"),
        ):
            pass

        # Verify performance logging occurred
        assert mock_debug.call_count == 2
        start_call = mock_debug.call_args_list[0][0][0]
        assert "[PERF] Starting test_operation" in start_call
