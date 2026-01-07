"""
Base exception classes for the trading system.
"""

from typing import Any


class TradingSystemError(Exception):
    """
    Base exception for all trading system errors.

    All custom exceptions should inherit from this class to enable
    generic error handling and consistent error reporting.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """
        Initialize trading system error.

        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message

    def __repr__(self) -> str:
        """Return detailed error representation."""
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"
