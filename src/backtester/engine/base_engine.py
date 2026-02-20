"""
Abstract base class for backtest engines.

Provides shared initialisation and enforces the common interface that both
the VectorizedBacktestEngine and EventDrivenBacktestEngine must implement.

The BacktestEngineProtocol (protocols.py) remains the preferred type hint for
external callers (duck typing). This ABC is for engine implementors — inheriting
from it ensures the shared config attribute and method signature are enforced
at class-definition time rather than at first call.
"""

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from src.backtester.models import BacktestConfig, BacktestResult
from src.strategies.base import Strategy

__all__ = ["BaseBacktestEngine"]


class BaseBacktestEngine(ABC):
    """Abstract base for backtest engines.

    Subclasses must implement:
    - ``run(strategy, data_files, start_date, end_date) -> BacktestResult``

    Both engines share ``self.config`` set by this base ``__init__``.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Initialise engine with the given config.

        Args:
            config: Backtest configuration. Defaults to ``BacktestConfig()``
                    with built-in defaults.
        """
        self.config: BacktestConfig = config or BacktestConfig()

    @abstractmethod
    def run(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult:
        """Run a backtest with the given strategy and data.

        Args:
            strategy: Trading strategy to backtest
            data_files: Mapping of ticker → parquet file path
            start_date: Optional start date filter (inclusive)
            end_date: Optional end date filter (inclusive)

        Returns:
            BacktestResult with performance metrics and trades
        """
