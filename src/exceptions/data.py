"""
Data source-related exceptions.
"""

from typing import Any

from src.exceptions.base import TradingSystemError


class DataSourceError(TradingSystemError):
    """Base exception for data source-related errors."""

    pass


class DataSourceConnectionError(DataSourceError):
    """Error connecting to data source."""

    def __init__(
        self,
        message: str = "Failed to connect to data source",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)


class DataSourceNotFoundError(DataSourceError):
    """Data source not found or unavailable."""

    def __init__(
        self,
        message: str = "Data source not found",
        source: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize data source not found error.

        Args:
            message: Error message
            source: Name of the data source
            details: Additional error details
        """
        if details is None:
            details = {}
        if source:
            details["source"] = source

        super().__init__(message, details)
        self.source = source


__all__ = ["DataSourceError", "DataSourceConnectionError", "DataSourceNotFoundError"]
