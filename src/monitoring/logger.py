"""
Structured JSON logging for crypto-quant-system.

Provides consistent, structured logging across all services with support for:
- JSON output format
- Correlation IDs for request tracing
- Context enrichment
- Multiple output handlers (console, file, remote)
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

# =============================================================================
# JSON Formatter
# =============================================================================


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_pathname: bool = False,
        extra_fields: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_pathname = include_pathname
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data: dict[str, Any] = {}

        # Core fields
        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_logger:
            log_data["logger"] = record.name

        # Message
        log_data["message"] = record.getMessage()

        # Location (optional)
        if self.include_pathname:
            log_data["pathname"] = record.pathname
            log_data["lineno"] = record.lineno
            log_data["funcname"] = record.funcName

        # Exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": "".join(traceback.format_exception(*record.exc_info)),
            }

        # Extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "message",
                "thread",
                "threadName",
                "taskName",
            }:
                log_data[key] = value

        # Static extra fields
        log_data.update(self.extra_fields)

        return json.dumps(log_data, default=str, ensure_ascii=False)


# =============================================================================
# Structured Logger
# =============================================================================


class StructuredLogger:
    """
    Structured logger with JSON output and context support.

    Usage:
        logger = StructuredLogger("trading-bot")
        logger.info("Order placed", symbol="BTC", amount=1000000)
        logger.error("Order failed", symbol="ETH", error="Insufficient balance")

        # With context
        with logger.context(request_id="abc123", user="trader1"):
            logger.info("Processing request")
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        json_output: bool = True,
        log_file: str | Path | None = None,
        extra_fields: dict[str, Any] | None = None,
    ):
        self.name = name
        self._context: dict[str, Any] = {}
        self._extra_fields = extra_fields or {}

        # Create logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if json_output:
            console_handler.setFormatter(
                JSONFormatter(extra_fields={"service": name, **self._extra_fields})
            )
        else:
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
        self._logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(
                JSONFormatter(extra_fields={"service": name, **self._extra_fields})
            )
            self._logger.addHandler(file_handler)

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Log a message with context and extra fields."""
        extra = {**self._context, **kwargs}
        self._logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        extra = {**self._context, **kwargs}
        self._logger.exception(message, extra=extra)

    class _ContextManager:
        """Context manager for temporary logging context."""

        def __init__(self, logger: StructuredLogger, context: dict[str, Any]):
            self._logger = logger
            self._context = context
            self._previous_context: dict[str, Any] = {}

        def __enter__(self) -> StructuredLogger._ContextManager:
            self._previous_context = self._logger._context.copy()
            self._logger._context.update(self._context)
            return self

        def __exit__(self, *args: Any) -> None:
            self._logger._context = self._previous_context

    def context(self, **kwargs: Any) -> _ContextManager:
        """Create a context manager with additional logging context."""
        return self._ContextManager(self, kwargs)

    def bind(self, **kwargs: Any) -> StructuredLogger:
        """Add permanent context to the logger."""
        self._context.update(kwargs)
        return self

    def unbind(self, *keys: str) -> StructuredLogger:
        """Remove context keys from the logger."""
        for key in keys:
            self._context.pop(key, None)
        return self


# =============================================================================
# Logger Factory
# =============================================================================

_loggers: dict[str, StructuredLogger] = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    json_output: bool = True,
    log_file: str | Path | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name (usually module or service name)
        level: Logging level
        json_output: Use JSON formatting
        log_file: Optional file path for logging
        extra_fields: Extra fields to include in all log records

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(
            name=name,
            level=level,
            json_output=json_output,
            log_file=log_file,
            extra_fields=extra_fields,
        )
    return _loggers[name]


# =============================================================================
# Decorators
# =============================================================================


def log_execution(
    logger: StructuredLogger | None = None,
    level: int = logging.INFO,
    log_args: bool = True,
    log_result: bool = False,
) -> Callable[..., Any]:
    """
    Decorator to log function execution.

    Usage:
        @log_execution(logger)
        def process_order(symbol: str, amount: float):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or get_logger(func.__module__)
            func_name = func.__qualname__

            # Log entry
            log_data: dict[str, Any] = {"function": func_name}
            if log_args:
                log_data["args"] = str(args)[:500]
                log_data["kwargs"] = str(kwargs)[:500]

            _logger._log(level, f"Executing {func_name}", **log_data)

            try:
                result = func(*args, **kwargs)

                # Log success
                result_data: dict[str, Any] = {"function": func_name, "status": "success"}
                if log_result:
                    result_data["result"] = str(result)[:500]

                _logger._log(level, f"Completed {func_name}", **result_data)
                return result

            except Exception as e:
                _logger.exception(
                    f"Failed {func_name}",
                    function=func_name,
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise

        return wrapper

    return decorator
