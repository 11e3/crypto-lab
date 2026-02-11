"""
Event-driven engine execution logic.

Contains entry and exit execution functions.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.backtester.engine.event_data_loader import Position
from src.backtester.models import BacktestConfig, Trade
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_exit_condition(
    position: Position,
    row: pd.Series[float],
    config: BacktestConfig,
) -> tuple[bool, str]:
    """
    Check if position should be exited.

    Priority order: signal > trailing_stop > stop_loss > take_profit

    Args:
        position: Current position
        row: Current bar data
        config: Backtest configuration

    Returns:
        Tuple of (should_exit, exit_reason)
    """
    close_price = float(row["close"])

    # Update highest price for trailing stop
    if close_price > position.highest_price:
        position.highest_price = close_price

    # Signal exit
    if "exit_signal" in row and row["exit_signal"]:
        return True, "signal"

    # Trailing stop
    if config.trailing_stop_pct is not None:
        stop_price = position.highest_price * (1 - config.trailing_stop_pct)
        if close_price <= stop_price:
            return True, "trailing_stop"

    # Stop loss
    if config.stop_loss_pct is not None:
        stop_price = position.entry_price * (1 - config.stop_loss_pct)
        if close_price <= stop_price:
            return True, "stop_loss"

    # Take profit
    if config.take_profit_pct is not None:
        target_price = position.entry_price * (1 + config.take_profit_pct)
        if close_price >= target_price:
            return True, "take_profit"

    return False, ""


def execute_exit(
    position: Position,
    row: pd.Series[float],
    current_date: date,
    exit_reason: str,
    config: BacktestConfig,
) -> tuple[Trade, float]:
    """
    Execute exit and create trade record.

    Args:
        position: Position to close
        row: Current bar data
        current_date: Current date
        exit_reason: Reason for exit
        config: Backtest configuration

    Returns:
        Tuple of (Trade, revenue)
    """
    # Event-driven path: apply slippage here (not pre-computed like vectorized path)
    exit_price = float(row.get("exit_price", row["close"]))
    exit_price = exit_price * (1 - config.slippage_rate)

    revenue = position.amount * exit_price * (1 - config.fee_rate)
    # Match the actual cost deducted from cash during entry:
    # execute_entry: cost = amount * entry_price * (1 + fee_rate)
    cost = position.amount * position.entry_price * (1 + config.fee_rate)
    pnl = revenue - cost
    pnl_pct = (exit_price / position.entry_price - 1) * 100

    trade = Trade(
        ticker=position.ticker,
        entry_date=position.entry_date,
        entry_price=position.entry_price,
        exit_date=current_date,
        exit_price=exit_price,
        amount=position.amount,
        pnl=pnl,
        pnl_pct=pnl_pct,
        commission_cost=(
            position.amount * position.entry_price * config.fee_rate
            + position.amount * exit_price * config.fee_rate
        ),
        slippage_cost=(
            position.amount * position.entry_price * config.slippage_rate
            + position.amount * exit_price * config.slippage_rate
        ),
        exit_reason=exit_reason,
    )

    logger.debug(f"Close {position.ticker} @ {exit_price:.0f}: PnL={pnl_pct:.2f}% ({exit_reason})")

    return trade, revenue


def execute_entry(
    ticker: str,
    row: pd.Series[float],
    current_date: date,
    cash: float,
    remaining_slots: int,
    config: BacktestConfig,
) -> tuple[Position | None, float]:
    """
    Execute entry if conditions are met.

    Args:
        ticker: Ticker symbol
        row: Current bar data
        current_date: Current date
        cash: Available cash
        remaining_slots: Number of available slots
        config: Backtest configuration

    Returns:
        Tuple of (Position or None, cost)
    """
    # Event-driven path: apply slippage here (not pre-computed like vectorized path)
    entry_price = float(row.get("entry_price", row.get("target", row["close"])))
    entry_price = entry_price * (1 + config.slippage_rate)

    allocation = cash / remaining_slots
    amount = (allocation / entry_price) * (1 - config.fee_rate)
    cost = amount * entry_price * (1 + config.fee_rate)

    if cost > cash:
        logger.debug(f"Insufficient cash for {ticker}: need {cost:.0f}, have {cash:.0f}")
        return None, 0.0

    position = Position(
        ticker=ticker,
        entry_date=current_date,
        entry_price=entry_price,
        amount=amount,
        highest_price=entry_price,
    )

    logger.debug(f"Open {ticker} @ {entry_price:.0f}: amount={amount:.4f}, cost={cost:.0f}")

    return position, cost


__all__ = [
    "check_exit_condition",
    "execute_exit",
    "execute_entry",
]
