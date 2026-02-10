"""Backtest execution service.

Service for backtest execution and result management.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

from src.backtester.engine import (
    BacktestEngine,
    EventDrivenBacktestEngine,
    VectorizedBacktestEngine,
)
from src.backtester.models import BacktestConfig, BacktestResult
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["run_backtest_service", "BacktestService"]


class BacktestService:
    """Backtest execution service.

    Wraps BacktestEngine to execute backtests
    and manage results in Streamlit environment.
    """

    def __init__(
        self,
        config: BacktestConfig,
        engine: BacktestEngine | None = None,
        use_vectorized: bool = True,
    ) -> None:
        """Initialize backtest service.

        Args:
            config: Backtest configuration
            engine: Optional BacktestEngine (uses VectorizedBacktestEngine if not provided)
            use_vectorized: Whether to use VectorizedBacktestEngine (default: True, performance improvement)
        """
        self.config = config
        if engine:
            self.engine = engine
        elif use_vectorized:
            # Use VectorizedBacktestEngine by default (10-100x faster)
            self.engine = VectorizedBacktestEngine(config)
        else:
            self.engine = EventDrivenBacktestEngine(config)

    def run(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult | None:
        """Execute backtest.

        Args:
            strategy: Strategy instance
            data_files: {ticker: file_path} dictionary
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            BacktestResult or None (on failure)
        """
        try:
            logger.info(f"Starting backtest: {strategy.name} with {len(data_files)} assets")

            result = self.engine.run(
                strategy=strategy,
                data_files=data_files,
                start_date=start_date,
                end_date=end_date,
            )

            logger.info(
                f"Backtest completed: "
                f"Return={result.total_return:.2f}%, "
                f"Trades={result.total_trades}"
            )

            return result

        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            return None


@st.cache_data(show_spinner="Running backtest...")
def run_backtest_service(
    strategy_name: str,
    strategy_params: dict[str, Any],
    data_files_dict: dict[str, str],  # {ticker: file_path_str}
    config_dict: dict[str, Any],
    start_date_str: str | None,
    end_date_str: str | None,
) -> BacktestResult | None:
    """Cacheable backtest execution wrapper.

    Convert all parameters to serializable types for Streamlit caching.

    Args:
        strategy_name: Strategy name
        strategy_params: Strategy parameters
        data_files_dict: Data file path dictionary
        config_dict: Backtest configuration dictionary
        start_date_str: Start date string
        end_date_str: End date string

    Returns:
        BacktestResult or None
    """
    try:
        # Create Strategy instance
        from src.web.components.sidebar.strategy_selector import (
            create_strategy_instance,
        )

        strategy = create_strategy_instance(strategy_name, strategy_params)
        if not strategy:
            logger.error("Failed to create strategy instance")
            return None

        # Restore Path objects
        data_files = {ticker: Path(path) for ticker, path in data_files_dict.items()}

        # Restore dates
        start_date = date.fromisoformat(start_date_str) if start_date_str else None
        end_date = date.fromisoformat(end_date_str) if end_date_str else None

        # Create BacktestConfig
        config = BacktestConfig(**config_dict)

        # Execute backtest
        service = BacktestService(config)
        result = service.run(strategy, data_files, start_date, end_date)

        return result

    except Exception as e:
        logger.exception(f"Backtest service failed: {e}")
        return None
