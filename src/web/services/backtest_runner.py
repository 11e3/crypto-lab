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


def execute_backtest(
    strategy_name: str,
    strategy_params: dict[str, Any],
    data_files_dict: dict[str, str],
    config_dict: dict[str, Any],
    start_date_str: str | None,
    end_date_str: str | None,
) -> BacktestResult | None:
    """Execute a backtest from serializable parameters (no Streamlit dependency).

    Args:
        strategy_name: Strategy name
        strategy_params: Strategy parameters
        data_files_dict: Data file path dictionary {ticker: file_path_str}
        config_dict: Backtest configuration dictionary
        start_date_str: Start date ISO string
        end_date_str: End date ISO string

    Returns:
        BacktestResult or None
    """
    try:
        from src.web.components.sidebar.strategy_selector import (
            create_strategy_instance,
        )

        strategy = create_strategy_instance(strategy_name, strategy_params)
        if not strategy:
            logger.error("Failed to create strategy instance")
            return None

        data_files = {ticker: Path(path) for ticker, path in data_files_dict.items()}
        start_date = date.fromisoformat(start_date_str) if start_date_str else None
        end_date = date.fromisoformat(end_date_str) if end_date_str else None
        config = BacktestConfig(**config_dict)

        service = BacktestService(config)
        return service.run(strategy, data_files, start_date, end_date)

    except Exception as e:
        logger.exception(f"Backtest service failed: {e}")
        return None


@st.cache_data(show_spinner="Running backtest...")
def run_backtest_service(
    strategy_name: str,
    strategy_params: dict[str, Any],
    data_files_dict: dict[str, str],
    config_dict: dict[str, Any],
    start_date_str: str | None,
    end_date_str: str | None,
) -> BacktestResult | None:
    """Streamlit-cached wrapper around execute_backtest."""
    return execute_backtest(
        strategy_name,
        strategy_params,
        data_files_dict,
        config_dict,
        start_date_str,
        end_date_str,
    )
