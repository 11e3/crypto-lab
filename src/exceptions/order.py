"""
Order-related exceptions.
"""

from typing import Any

from src.exceptions.base import TradingSystemError


class OrderError(TradingSystemError):
    """Base exception for order-related errors."""

    pass


class OrderNotFoundError(OrderError):
    """Order not found."""

    def __init__(
        self,
        message: str = "Order not found",
        order_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize order not found error.

        Args:
            message: Error message
            order_id: Order identifier
            details: Additional error details
        """
        if details is None:
            details = {}
        if order_id:
            details["order_id"] = order_id

        super().__init__(message, details)
        self.order_id = order_id


class OrderExecutionError(OrderError):
    """Error executing an order."""

    def __init__(
        self,
        message: str = "Order execution failed",
        order_id: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize order execution error.

        Args:
            message: Error message
            order_id: Order identifier
            reason: Reason for failure
            details: Additional error details
        """
        if details is None:
            details = {}
        if order_id:
            details["order_id"] = order_id
        if reason:
            details["reason"] = reason

        super().__init__(message, details)
        self.order_id = order_id
        self.reason = reason
