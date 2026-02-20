"""
Event-driven Backtesting Engine.

Clear, debuggable implementation that processes data sequentially.
Ideal for strategy development and testing.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester.engine.base_engine import BaseBacktestEngine
from src.backtester.engine.event_data_loader import Position, load_event_data
from src.backtester.engine.event_loop import (
    calculate_portfolio_equity,
    close_remaining_positions,
    process_entries,
    process_exits,
)
from src.backtester.metrics import calculate_metrics
from src.backtester.models import BacktestConfig, BacktestResult, Trade
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EventDrivenBacktestEngine(BaseBacktestEngine):
    """
    Event-driven backtesting engine.

    Processes market data chronologically (day-by-day):
    1. Check exit conditions (signals, stops) -> Close positions
    2. Check entry signals -> Open new positions
    3. Update equity curve

    Best for strategy development, testing, and clear trade logs.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        super().__init__(config)

    def run(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult:
        """Run backtest on multiple assets."""
        ticker_data, all_dates = self._load_and_validate(strategy, data_files, start_date, end_date)

        if ticker_data is None or all_dates is None:
            return BacktestResult(strategy_name=strategy.name)

        logger.info(f"Backtesting {len(ticker_data)} assets from {all_dates[0]} to {all_dates[-1]}")
        logger.info(f"Total dates: {len(all_dates)}")

        equity_curve, dates_list, trades_list = self._simulate(ticker_data, all_dates)

        return calculate_metrics(
            equity_curve=np.array(equity_curve, dtype=np.float32),
            dates=np.array(dates_list),
            trades=trades_list,
            config=self.config,
            strategy_name=strategy.name,
        )

    def _load_and_validate(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
        start_date: date | None,
        end_date: date | None,
    ) -> tuple[dict[str, pd.DataFrame] | None, list[date] | None]:
        """Load data and compute unified date range. Returns (None, None) on failure."""
        ticker_data = load_event_data(strategy, data_files, start_date, end_date)

        if not ticker_data:
            logger.warning("No data loaded")
            return None, None

        all_dates: list[date] = sorted(
            set().union(*[set(df["index_date"].to_list()) for df in ticker_data.values()])
        )

        if not all_dates:
            logger.warning("No valid dates")
            return None, None

        return ticker_data, all_dates

    def _simulate(
        self,
        ticker_data: dict[str, pd.DataFrame],
        all_dates: list[date],
    ) -> tuple[list[float], list[date], list[Trade]]:
        """Run the day-by-day simulation loop."""
        cash = self.config.initial_capital
        positions: dict[str, Position] = {}
        equity_curve: list[float] = []
        dates_list: list[date] = []
        trades_list: list[Trade] = []
        total_entry_signals = 0
        total_positions_opened = 0

        for current_date in all_dates:
            dates_list.append(current_date)

            # Get data for this date for all tickers
            current_data: dict[str, pd.Series] = {}
            for ticker, df in ticker_data.items():
                date_data = df[df["index_date"] == current_date]
                if not date_data.empty:
                    current_data[ticker] = date_data.iloc[0]

            # Exit logic
            exit_trades, revenue, positions = process_exits(
                positions, current_data, current_date, self.config
            )
            trades_list.extend(exit_trades)
            cash += revenue

            # Entry logic
            positions_before = len(positions)
            positions, cash, signals = process_entries(
                positions, current_data, current_date, cash, self.config
            )
            total_entry_signals += signals
            total_positions_opened += len(positions) - positions_before

            # Calculate equity
            equity = calculate_portfolio_equity(positions, current_data, cash)
            equity_curve.append(equity)

        # Close remaining positions at end
        if positions and all_dates:
            final_trades = close_remaining_positions(
                positions, ticker_data, all_dates[-1], self.config
            )
            trades_list.extend(final_trades)

        logger.info(f"Total entry signals detected: {total_entry_signals}")
        logger.info(f"Total positions opened: {total_positions_opened}")
        logger.info(f"Total trades completed: {len(trades_list)}")

        return equity_curve, dates_list, trades_list
