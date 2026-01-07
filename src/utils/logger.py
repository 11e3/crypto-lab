"""
Centralized logging configuration with structured logging support.

Provides consistent logging setup across the entire application with:
- Structured logging with context
- Performance logging
- File and console output separation
- Configurable log levels
"""

import logging
import sys
import time
from collections.abc import MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Logging format constants (defined here to avoid circular import)
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


class ContextLogger(logging.LoggerAdapter[logging.Logger]):
    """
    Logger adapter that adds context information to log records.

    Usage:
        logger = get_logger(__name__)
        context_logger = ContextLogger(logger, {"ticker": "KRW-BTC", "order_id": "123"})
        context_logger.info("Order placed")
    """

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        """Add context to log message."""
        if self.extra:
            context_str = " | ".join(f"{k}={v}" for k, v in self.extra.items())
            msg = f"[{context_str}] {msg}"
        return msg, kwargs


class PerformanceLogger:
    """
    Context manager for performance logging.

    Usage:
        with PerformanceLogger(logger, "fetch_data"):
            # code to measure
            pass
    """

    def __init__(self, logger: logging.Logger, operation: str, **context: Any) -> None:
        """
        Initialize performance logger.

        Args:
            logger: Logger instance
            operation: Operation name
            context: Additional context information
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: float | None = None

    def __enter__(self) -> "PerformanceLogger":
        """Start timing."""
        self.start_time = time.perf_counter()
        context_str = (
            " | ".join(f"{k}={v}" for k, v in self.context.items()) if self.context else ""
        )
        if context_str:
            self.logger.debug(f"[PERF] Starting {self.operation} [{context_str}]")
        else:
            self.logger.debug(f"[PERF] Starting {self.operation}")
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Log elapsed time."""
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            context_str = (
                " | ".join(f"{k}={v}" for k, v in self.context.items()) if self.context else ""
            )
            if context_str:
                self.logger.debug(
                    f"[PERF] Completed {self.operation} in {elapsed:.3f}s [{context_str}]"
                )
            else:
                self.logger.debug(f"[PERF] Completed {self.operation} in {elapsed:.3f}s")


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    format_string: str | None = None,
    enable_performance_logging: bool = False,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
        format_string: Custom format string (uses default if None)
        enable_performance_logging: Enable performance logging at DEBUG level
    """
    format_string = format_string or LOG_FORMAT

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(format_string, datefmt=LOG_DATE_FORMAT))
        handlers.append(file_handler)

    # Set up console handler with formatter
    console_handler = handlers[0]
    console_handler.setFormatter(logging.Formatter(format_string, datefmt=LOG_DATE_FORMAT))

    # Adjust level for performance logging
    effective_level = logging.DEBUG if enable_performance_logging else level

    logging.basicConfig(
        level=effective_level,
        format=format_string,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_context_logger(name: str, **context: Any) -> ContextLogger:
    """
    Get a logger with context information.

    Args:
        name: Logger name (typically __name__)
        context: Context key-value pairs to include in all log messages

    Returns:
        ContextLogger instance

    Example:
        logger = get_context_logger(__name__, ticker="KRW-BTC", order_id="123")
        logger.info("Order placed")  # Logs: [ticker=KRW-BTC | order_id=123] Order placed
    """
    base_logger = get_logger(name)
    return ContextLogger(base_logger, context)


@contextmanager
def log_performance(logger: logging.Logger, operation: str, **context: Any) -> Any:
    """
    Context manager for performance logging.

    Args:
        logger: Logger instance
        operation: Operation name
        context: Additional context information

    Example:
        with log_performance(logger, "fetch_ohlcv", ticker="KRW-BTC"):
            data = exchange.get_ohlcv("KRW-BTC")
    """
    with PerformanceLogger(logger, operation, **context):
        yield
