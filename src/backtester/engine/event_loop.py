"""
Event loop helpers for event-driven backtesting.

Provides modular functions for processing daily market events.
"""

from datetime import date

import pandas as pd

from src.backtester.engine.event_exec import (
    check_exit_condition,
    execute_entry,
    execute_exit,
)
from src.backtester.engine.trade_costs import TradeCostCalculator
from src.backtester.models import BacktestConfig, Trade
from src.strategies.base_models import Position

__all__ = [
    "process_exits",
    "process_entries",
    "calculate_portfolio_equity",
    "close_remaining_positions",
]


def process_exits(
    positions: dict[str, Position],
    current_data: dict[str, pd.Series],
    current_date: date,
    config: BacktestConfig,
) -> tuple[list[Trade], float, dict[str, Position]]:
    """
    Process exit conditions for all positions.

    Args:
        positions: Current positions dict
        current_data: Today's market data by ticker
        current_date: Current date
        config: Backtest configuration

    Returns:
        Tuple of (trades, total_revenue, remaining_positions)
    """
    trades: list[Trade] = []
    total_revenue = 0.0
    remaining_positions = dict(positions)
    positions_to_close: list[tuple[str, str]] = []

    for ticker, position in remaining_positions.items():
        if ticker not in current_data:
            continue

        row = current_data[ticker]
        should_exit, exit_reason = check_exit_condition(position, row, config)

        if should_exit:
            positions_to_close.append((ticker, exit_reason))

    for ticker, exit_reason in positions_to_close:
        position = remaining_positions[ticker]
        row = current_data[ticker]

        trade, revenue = execute_exit(position, row, current_date, exit_reason, config)
        trades.append(trade)
        total_revenue += revenue
        del remaining_positions[ticker]

    return trades, total_revenue, remaining_positions


def process_entries(
    positions: dict[str, Position],
    current_data: dict[str, pd.Series],
    current_date: date,
    cash: float,
    config: BacktestConfig,
) -> tuple[dict[str, Position], float, int]:
    """
    Process entry signals and open new positions.

    Args:
        positions: Current positions dict
        current_data: Today's market data by ticker
        current_date: Current date
        cash: Available cash
        config: Backtest configuration

    Returns:
        Tuple of (updated_positions, remaining_cash, entry_signals_count)
    """
    updated_positions = dict(positions)
    remaining_cash = cash
    entry_signals = 0
    available_slots = config.max_slots - len(updated_positions)

    if available_slots <= 0:
        return updated_positions, remaining_cash, entry_signals

    entry_candidates: list[tuple[str, pd.Series]] = []

    for ticker, row in current_data.items():
        if ticker in updated_positions:
            continue

        if "entry_signal" not in row:
            continue

        if row["entry_signal"]:
            entry_signals += 1
            entry_candidates.append((ticker, row))

    entry_candidates = entry_candidates[:available_slots]

    for ticker, row in entry_candidates:
        remaining_slots = config.max_slots - len(updated_positions)
        if remaining_slots <= 0:
            continue

        new_position, cost = execute_entry(
            ticker, row, current_date, remaining_cash, remaining_slots, config
        )

        if new_position is not None:
            updated_positions[ticker] = new_position
            remaining_cash -= cost

    return updated_positions, remaining_cash, entry_signals


def calculate_portfolio_equity(
    positions: dict[str, Position],
    current_data: dict[str, pd.Series],
    cash: float,
) -> float:
    """
    Calculate total portfolio equity.

    Args:
        positions: Current positions dict
        current_data: Today's market data by ticker
        cash: Available cash

    Returns:
        Total equity value
    """
    portfolio_value = 0.0
    for ticker, pos in positions.items():
        if ticker in current_data:
            close_price = current_data[ticker]["close"]
            if pd.notna(close_price):
                portfolio_value += pos.amount * float(close_price)
            else:
                # Fallback to entry price if close is NaN
                portfolio_value += pos.amount * pos.entry_price
    return cash + portfolio_value


def close_remaining_positions(
    positions: dict[str, Position],
    ticker_data: dict[str, pd.DataFrame],
    final_date: date,
    config: BacktestConfig,
) -> list[Trade]:
    """
    Close all remaining positions at backtest end.

    Args:
        positions: Remaining positions dict
        ticker_data: All ticker DataFrames
        final_date: Final backtest date
        config: Backtest configuration

    Returns:
        List of closing trades
    """
    trades: list[Trade] = []
    calculator = TradeCostCalculator(config.fee_rate, config.slippage_rate)

    for ticker, position in positions.items():
        if ticker not in ticker_data:
            continue

        df = ticker_data[ticker]
        final_data = df[df["index_date"] == final_date]

        if final_data.empty:
            continue

        row = final_data.iloc[0]
        exit_price = calculator.apply_slippage(float(row["close"]), is_buy=False)
        costs = calculator.calculate_exit_costs(position.entry_price, exit_price, position.amount)

        trade = Trade(
            ticker=ticker,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=final_date,
            exit_price=exit_price,
            amount=position.amount,
            pnl=costs.pnl,
            pnl_pct=costs.pnl_pct,
            exit_reason="open",
        )
        trades.append(trade)

    return trades
