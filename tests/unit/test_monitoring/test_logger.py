"""Tests for monitoring.logger module."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import patch

import pytest

from src.monitoring.logger import (
    JSONFormatter,
    StructuredLogger,
    _loggers,
    get_logger,
    log_execution,
)


@pytest.fixture(autouse=True)
def _clear_logger_cache() -> None:
    """Clear cached loggers between tests."""
    _loggers.clear()


# =============================================================================
# JSONFormatter
# =============================================================================


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_fields(self) -> None:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        formatter = JSONFormatter()
        output = json.loads(formatter.format(record))

        assert output["message"] == "hello"
        assert output["level"] == "INFO"
        assert output["logger"] == "test"
        assert "timestamp" in output

    def test_disable_timestamp(self) -> None:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        formatter = JSONFormatter(include_timestamp=False)
        output = json.loads(formatter.format(record))

        assert "timestamp" not in output

    def test_disable_level_and_logger(self) -> None:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        formatter = JSONFormatter(include_level=False, include_logger=False)
        output = json.loads(formatter.format(record))

        assert "level" not in output
        assert "logger" not in output

    def test_include_pathname(self) -> None:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="/foo/bar.py", lineno=42,
            msg="msg", args=(), exc_info=None,
        )
        record.funcName = "my_func"
        formatter = JSONFormatter(include_pathname=True)
        output = json.loads(formatter.format(record))

        assert output["pathname"] == "/foo/bar.py"
        assert output["lineno"] == 42
        assert output["funcname"] == "my_func"

    def test_exception_info(self) -> None:
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="failed", args=(), exc_info=exc_info,
        )
        formatter = JSONFormatter()
        output = json.loads(formatter.format(record))

        assert "exception" in output
        assert output["exception"]["type"] == "ValueError"
        assert output["exception"]["message"] == "test error"
        assert "traceback" in output["exception"]

    def test_extra_fields(self) -> None:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        formatter = JSONFormatter(extra_fields={"env": "prod", "version": "1.0"})
        output = json.loads(formatter.format(record))

        assert output["env"] == "prod"
        assert output["version"] == "1.0"

    def test_record_extra_fields(self) -> None:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        record.custom_field = "custom_value"  # type: ignore[attr-defined]
        formatter = JSONFormatter()
        output = json.loads(formatter.format(record))

        assert output["custom_field"] == "custom_value"


# =============================================================================
# StructuredLogger
# =============================================================================


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_creation(self) -> None:
        logger = StructuredLogger("test-svc")
        assert logger.name == "test-svc"

    def test_log_methods(self) -> None:
        logger = StructuredLogger("test-svc", level=logging.DEBUG)
        with patch.object(logger._logger, "log") as mock_log:
            logger.debug("d")
            mock_log.assert_called_with(logging.DEBUG, "d", extra={})

            logger.info("i")
            mock_log.assert_called_with(logging.INFO, "i", extra={})

            logger.warning("w")
            mock_log.assert_called_with(logging.WARNING, "w", extra={})

            logger.error("e")
            mock_log.assert_called_with(logging.ERROR, "e", extra={})

            logger.critical("c")
            mock_log.assert_called_with(logging.CRITICAL, "c", extra={})

    def test_log_with_kwargs(self) -> None:
        logger = StructuredLogger("test-svc")
        with patch.object(logger._logger, "log") as mock_log:
            logger.info("order placed", symbol="BTC", amount=1000)
            mock_log.assert_called_with(
                logging.INFO, "order placed",
                extra={"symbol": "BTC", "amount": 1000},
            )

    def test_exception_method(self) -> None:
        logger = StructuredLogger("test-svc")
        with patch.object(logger._logger, "exception") as mock_exc:
            logger.exception("boom", code=500)
            mock_exc.assert_called_with("boom", extra={"code": 500})

    def test_context_manager(self) -> None:
        logger = StructuredLogger("test-svc")
        with patch.object(logger._logger, "log") as mock_log:
            with logger.context(request_id="abc"):
                logger.info("inside")
                mock_log.assert_called_with(
                    logging.INFO, "inside",
                    extra={"request_id": "abc"},
                )

            logger.info("outside")
            mock_log.assert_called_with(logging.INFO, "outside", extra={})

    def test_context_restores_previous(self) -> None:
        logger = StructuredLogger("test-svc")
        logger.bind(env="prod")

        with logger.context(request_id="abc"):
            assert logger._context == {"env": "prod", "request_id": "abc"}

        assert logger._context == {"env": "prod"}

    def test_bind_and_unbind(self) -> None:
        logger = StructuredLogger("test-svc")

        result = logger.bind(user="alice")
        assert result is logger
        assert logger._context == {"user": "alice"}

        result = logger.unbind("user")
        assert result is logger
        assert logger._context == {}

    def test_unbind_nonexistent_key(self) -> None:
        logger = StructuredLogger("test-svc")
        logger.unbind("nonexistent")
        assert logger._context == {}

    def test_plain_text_output(self) -> None:
        logger = StructuredLogger("test-svc", json_output=False)
        handler = logger._logger.handlers[0]
        assert not isinstance(handler.formatter, JSONFormatter)

    def test_file_handler(self, tmp_path: Any) -> None:
        log_file = tmp_path / "test.log"
        logger = StructuredLogger("test-svc", log_file=str(log_file))
        assert len(logger._logger.handlers) == 2
        logger.info("file test")
        assert log_file.exists()


# =============================================================================
# get_logger factory
# =============================================================================


class TestGetLogger:
    """Tests for get_logger factory."""

    def test_creates_logger(self) -> None:
        logger = get_logger("test-factory")
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "test-factory"

    def test_returns_same_instance(self) -> None:
        logger1 = get_logger("singleton")
        logger2 = get_logger("singleton")
        assert logger1 is logger2

    def test_different_names_different_instances(self) -> None:
        logger1 = get_logger("a")
        logger2 = get_logger("b")
        assert logger1 is not logger2


# =============================================================================
# log_execution decorator
# =============================================================================


class TestLogExecution:
    """Tests for log_execution decorator."""

    def test_basic_execution(self) -> None:
        logger = StructuredLogger("test-dec", level=logging.DEBUG)

        @log_execution(logger, log_args=False)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_logs_entry_and_exit(self) -> None:
        logger = StructuredLogger("test-dec", level=logging.DEBUG)

        with patch.object(logger, "_log") as mock_log:

            @log_execution(logger)
            def my_func() -> str:
                return "ok"

            my_func()

            assert mock_log.call_count == 2
            entry_call = mock_log.call_args_list[0]
            assert "Executing" in entry_call.args[1]
            exit_call = mock_log.call_args_list[1]
            assert "Completed" in exit_call.args[1]

    def test_logs_exception(self) -> None:
        logger = StructuredLogger("test-dec", level=logging.DEBUG)

        @log_execution(logger, log_args=False)
        def bad_func() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            bad_func()

    def test_log_result(self) -> None:
        logger = StructuredLogger("test-dec", level=logging.DEBUG)

        with patch.object(logger, "_log") as mock_log:

            @log_execution(logger, log_result=True)
            def get_value() -> int:
                return 42

            get_value()

            exit_call = mock_log.call_args_list[1]
            assert exit_call.kwargs.get("result") == "42"

    def test_no_args_logging(self) -> None:
        logger = StructuredLogger("test-dec", level=logging.DEBUG)

        with patch.object(logger, "_log") as mock_log:

            @log_execution(logger, log_args=False)
            def my_func() -> None:
                pass

            my_func()

            entry_call = mock_log.call_args_list[0]
            assert "args" not in entry_call.kwargs

    def test_default_logger(self) -> None:
        @log_execution(log_args=False)
        def my_func() -> int:
            return 1

        result = my_func()
        assert result == 1
