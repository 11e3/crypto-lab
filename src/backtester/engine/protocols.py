"""
Protocol definitions for backtest engines.

Provides a common interface for all backtest engine implementations.
"""

from datetime import date
from pathlib import Path
from typing import Protocol

from src.backtester.models import BacktestConfig, BacktestResult
from src.strategies.base import Strategy


class BacktestEngineProtocol(Protocol):
    """
    Protocol defining the interface for backtest engines.

    All backtest engines must implement this interface to ensure
    they can be used interchangeably (LSP compliance).
    """

    config: BacktestConfig

    def run(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult:
        """
        Run a backtest with the given strategy and data.

        Args:
            strategy: Trading strategy to backtest
            data_files: Dictionary mapping ticker to data file path
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResult with performance metrics and trades
        """
        ...
