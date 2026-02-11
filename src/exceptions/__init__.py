"""
Custom exceptions for the trading system.

Provides a hierarchical exception structure for better error handling.
"""

from src.exceptions.base import TradingSystemError
from src.exceptions.data import (
    DataSourceConnectionError,
    DataSourceError,
    DataSourceNotFoundError,
)
from src.exceptions.strategy import (
    StrategyConfigurationError,
    StrategyError,
    StrategyExecutionError,
)

__all__ = [
    # Base
    "TradingSystemError",
    # Data
    "DataSourceError",
    "DataSourceConnectionError",
    "DataSourceNotFoundError",
    # Strategy
    "StrategyError",
    "StrategyConfigurationError",
    "StrategyExecutionError",
]
