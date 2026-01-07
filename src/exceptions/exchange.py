"""
Exchange-related exceptions.
"""

from typing import Any

from src.exceptions.base import TradingSystemError


class ExchangeError(TradingSystemError):
    """Base exception for exchange-related errors."""

    pass


class ExchangeConnectionError(ExchangeError):
    """Error connecting to exchange API."""

    def __init__(
        self, message: str = "Failed to connect to exchange", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, details)


class ExchangeAuthenticationError(ExchangeError):
    """Error authenticating with exchange API."""

    def __init__(
        self, message: str = "Authentication failed", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, details)


class ExchangeOrderError(ExchangeError):
    """Error placing or managing orders."""

    def __init__(
        self, message: str = "Order operation failed", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, details)


class InsufficientBalanceError(ExchangeOrderError):
    """Insufficient balance for order."""

    def __init__(
        self,
        message: str = "Insufficient balance",
        currency: str | None = None,
        required: float | None = None,
        available: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize insufficient balance error.

        Args:
            message: Error message
            currency: Currency that has insufficient balance
            required: Required amount
            available: Available amount
            details: Additional error details
        """
        if details is None:
            details = {}
        if currency:
            details["currency"] = currency
        if required is not None:
            details["required"] = required
        if available is not None:
            details["available"] = available

        super().__init__(message, details)
        self.currency = currency
        self.required = required
        self.available = available
