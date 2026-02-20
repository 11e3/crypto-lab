"""
Strategy-related exceptions.
"""

from typing import Any

from src.exceptions.base import TradingSystemError


class StrategyError(TradingSystemError):
    """Base exception for strategy-related errors."""

    pass


class StrategyConfigurationError(StrategyError):
    """Error in strategy configuration."""

    def __init__(
        self,
        message: str = "Strategy configuration error",
        strategy_name: str | None = None,
        parameter: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize strategy configuration error.

        Args:
            message: Error message
            strategy_name: Name of the strategy
            parameter: Parameter that caused the error
            details: Additional error details
        """
        if details is None:
            details = {}
        if strategy_name:
            details["strategy_name"] = strategy_name
        if parameter:
            details["parameter"] = parameter

        super().__init__(message, details)
        self.strategy_name = strategy_name
        self.parameter = parameter


class StrategyExecutionError(StrategyError):
    """Error executing a strategy."""

    def __init__(
        self,
        message: str = "Strategy execution error",
        strategy_name: str | None = None,
        ticker: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize strategy execution error.

        Args:
            message: Error message
            strategy_name: Name of the strategy
            ticker: Ticker that caused the error
            details: Additional error details
        """
        if details is None:
            details = {}
        if strategy_name:
            details["strategy_name"] = strategy_name
        if ticker:
            details["ticker"] = ticker

        super().__init__(message, details)
        self.strategy_name = strategy_name
        self.ticker = ticker


__all__ = ["StrategyError", "StrategyConfigurationError", "StrategyExecutionError"]
