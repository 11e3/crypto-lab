"""Vectorized Backtesting Engine."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester.engine.array_builder import (
    build_numpy_arrays,
    collect_valid_dates,
    filter_valid_dates,
)
from src.backtester.engine.data_loader import (
    get_cache_params,
    load_parquet_data,
    load_ticker_data,
)
from src.backtester.engine.entry_processor import process_entries
from src.backtester.engine.result_builder import build_backtest_result
from src.backtester.engine.signal_processor import add_price_columns
from src.backtester.engine.trade_simulator import (
    SimulationState,
    calculate_daily_equity,
    finalize_open_positions,
    initialize_simulation_state,
    process_exits,
    process_stop_loss_take_profit,
    track_asset_returns,
)
from src.backtester.models import BacktestConfig, BacktestResult
from src.orders.advanced_orders import AdvancedOrderManager
from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.utils.memory import optimize_dtypes

logger = get_logger(__name__)


class VectorizedBacktestEngine:
    """Vectorized backtesting engine using pandas/numpy."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Initialize backtest engine with config."""
        self.config = config or BacktestConfig()
        self.advanced_order_manager = AdvancedOrderManager()

    def load_data(self, filepath: Path) -> pd.DataFrame:
        """Load OHLCV data from parquet file."""
        return load_parquet_data(filepath)

    def run(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult:
        """Run vectorized backtest for a strategy on multiple assets."""
        ticker_data, ticker_historical_data = self._load_all_ticker_data(strategy, data_files)

        if not ticker_data:
            logger.warning("No data available for backtesting")
            return BacktestResult(strategy_name=strategy.name)

        all_dates = collect_valid_dates(ticker_data)
        sorted_dates = np.array(sorted(all_dates))

        if len(sorted_dates) == 0:
            return BacktestResult(strategy_name=strategy.name)

        tickers, n_tickers, n_dates, arrays = build_numpy_arrays(ticker_data, sorted_dates)
        sorted_dates, n_dates, arrays = filter_valid_dates(sorted_dates, arrays, n_dates)

        state = self._run_simulation(
            sorted_dates, n_dates, tickers, n_tickers, arrays, ticker_historical_data
        )

        return build_backtest_result(strategy, state, sorted_dates, self.config)

    def _load_all_ticker_data(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Load data for all tickers."""
        cache_params = get_cache_params(strategy)
        ticker_data: dict[str, pd.DataFrame] = {}
        ticker_historical_data: dict[str, pd.DataFrame] = {}

        for ticker, filepath in data_files.items():
            try:
                df, hist_df = load_ticker_data(
                    ticker,
                    filepath,
                    strategy,
                    cache_params,
                    use_cache=self.config.use_cache,
                    position_sizing=self.config.position_sizing,
                )
                df = add_price_columns(df, self.config)
                df = optimize_dtypes(df)
                ticker_data[ticker] = df
                if hist_df is not None:
                    ticker_historical_data[ticker] = hist_df
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}", exc_info=True)

        return ticker_data, ticker_historical_data

    def _run_simulation(
        self,
        sorted_dates: np.ndarray,
        n_dates: int,
        tickers: list[str],
        n_tickers: int,
        arrays: dict[str, np.ndarray],
        ticker_historical_data: dict[str, pd.DataFrame],
    ) -> SimulationState:
        """Run the main simulation loop."""
        state = initialize_simulation_state(
            self.config.initial_capital, n_tickers, n_dates, tickers
        )

        for d_idx in range(n_dates):
            current_date = sorted_dates[d_idx]
            valid_data = ~np.isnan(arrays["closes"][:, d_idx])

            track_asset_returns(state, d_idx, n_tickers, tickers, arrays["closes"], valid_data)

            process_stop_loss_take_profit(
                state,
                self.config,
                d_idx,
                current_date,
                sorted_dates,
                tickers,
                arrays["closes"],
                arrays["exit_prices"],
                valid_data,
                self.advanced_order_manager,
            )

            process_exits(
                state,
                self.config,
                d_idx,
                current_date,
                sorted_dates,
                tickers,
                arrays["exit_signals"],
                arrays["exit_prices"],
                valid_data,
                self.advanced_order_manager,
            )

            process_entries(
                state,
                self.config,
                d_idx,
                current_date,
                sorted_dates,
                tickers,
                n_tickers,
                arrays,
                valid_data,
                ticker_historical_data,
            )

            calculate_daily_equity(state, d_idx, n_tickers, arrays["closes"], valid_data)

        finalize_open_positions(
            state, sorted_dates, tickers, n_tickers, arrays["closes"], self.config
        )
        return state


BacktestEngine = VectorizedBacktestEngine
