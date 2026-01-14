"""
Trade simulation core for vectorized backtesting.

Re-exports from sub-modules for backward compatibility.
"""

from datetime import date

import numpy as np

from src.backtester.engine.trade_simulator_exec import (
    execute_exit,
    handle_normal_entry,
    handle_whipsaw,
)
from src.backtester.engine.trade_simulator_state import (
    SimulationState,
    initialize_simulation_state,
)
from src.backtester.models import BacktestConfig
from src.execution.orders.advanced_orders import AdvancedOrderManager
from src.utils.logger import get_logger

__all__ = [
    "SimulationState",
    "initialize_simulation_state",
    "process_stop_loss_take_profit",
    "process_exits",
    "calculate_daily_equity",
    "track_asset_returns",
    "finalize_open_positions",
    "handle_whipsaw",
    "handle_normal_entry",
]

logger = get_logger(__name__)


def process_stop_loss_take_profit(
    state: SimulationState,
    config: BacktestConfig,
    d_idx: int,
    current_date: date,
    sorted_dates: np.ndarray,
    tickers: list[str],
    highs: np.ndarray,
    lows: np.ndarray,
    exit_prices: np.ndarray,
    valid_data: np.ndarray,
    order_manager: AdvancedOrderManager,
) -> None:
    """Process stop-loss and take-profit orders.

    Uses intraday high/low prices for accurate stop-loss and take-profit detection:
    - Stop-loss: triggers if LOW price <= stop price (worst case intraday)
    - Take-profit: triggers if HIGH price >= target price (best case intraday)

    This prevents look-ahead bias that would occur if using only close price.
    """
    if config.stop_loss_pct is None and config.take_profit_pct is None:
        return

    n_tickers = len(tickers)
    in_position = state.position_amounts > 0

    for t_idx in range(n_tickers):
        if not in_position[t_idx] or not valid_data[t_idx]:
            continue

        entry_price = state.position_entry_prices[t_idx]
        if entry_price <= 0:
            continue

        high_price = highs[t_idx, d_idx]
        low_price = lows[t_idx, d_idx]

        should_exit, is_stop_loss, is_take_profit = _check_exit_conditions_intraday(
            entry_price, high_price, low_price, config
        )

        if should_exit:
            execute_exit(
                state,
                config,
                t_idx,
                d_idx,
                current_date,
                sorted_dates,
                tickers,
                exit_prices,
                order_manager,
                is_stop_loss,
                is_take_profit,
                "stop_loss" if is_stop_loss else "take_profit",
            )


def _check_exit_conditions_intraday(
    entry_price: float,
    high_price: float,
    low_price: float,
    config: BacktestConfig,
) -> tuple[bool, bool, bool]:
    """Check if exit conditions are met using intraday high/low.

    Args:
        entry_price: Position entry price
        high_price: Intraday high price
        low_price: Intraday low price
        config: Backtest configuration

    Returns:
        Tuple of (should_exit, is_stop_loss, is_take_profit)
    """
    # Stop-loss check: uses LOW price (worst case scenario)
    if config.stop_loss_pct is not None:
        stop_price = entry_price * (1 - config.stop_loss_pct)
        if low_price <= stop_price:
            return True, True, False

    # Take-profit check: uses HIGH price (best case scenario)
    if config.take_profit_pct is not None:
        target_price = entry_price * (1 + config.take_profit_pct)
        if high_price >= target_price:
            return True, False, True

    return False, False, False


def process_exits(
    state: SimulationState,
    config: BacktestConfig,
    d_idx: int,
    current_date: date,
    sorted_dates: np.ndarray,
    tickers: list[str],
    exit_signals: np.ndarray,
    exit_prices: np.ndarray,
    valid_data: np.ndarray,
    order_manager: AdvancedOrderManager,
) -> None:
    """Process exit signals."""
    in_position = state.position_amounts > 0
    should_exit = exit_signals[:, d_idx] & in_position & valid_data

    if not np.any(should_exit):
        return

    for t_idx in np.where(should_exit)[0]:
        execute_exit(
            state,
            config,
            t_idx,
            d_idx,
            current_date,
            sorted_dates,
            tickers,
            exit_prices,
            order_manager,
            False,
            False,
            "signal",
        )


def calculate_daily_equity(
    state: SimulationState,
    d_idx: int,
    n_tickers: int,
    closes: np.ndarray,
    valid_data: np.ndarray,
) -> None:
    """Calculate daily equity value."""
    positions_value = 0.0

    for t_idx in range(n_tickers):
        if state.position_amounts[t_idx] > 0:
            current_price = _get_current_price(state, t_idx, d_idx, closes, valid_data)
            positions_value += state.position_amounts[t_idx] * current_price

    state.equity_curve[d_idx] = state.cash + positions_value

    if (np.isnan(state.equity_curve[d_idx]) or state.equity_curve[d_idx] < 0) and d_idx > 0:
        state.equity_curve[d_idx] = state.equity_curve[d_idx - 1]


def _get_current_price(
    state: SimulationState, t_idx: int, d_idx: int, closes: np.ndarray, valid_data: np.ndarray
) -> float:
    """Get current price for position valuation."""
    if valid_data[t_idx] and not np.isnan(closes[t_idx, d_idx]):
        return float(closes[t_idx, d_idx])
    if d_idx > 0 and not np.isnan(closes[t_idx, d_idx - 1]):
        return float(closes[t_idx, d_idx - 1])
    return float(state.position_entry_prices[t_idx])


def track_asset_returns(
    state: SimulationState,
    d_idx: int,
    n_tickers: int,
    tickers: list[str],
    closes: np.ndarray,
    valid_data: np.ndarray,
) -> None:
    """Track individual asset returns for correlation analysis."""
    for t_idx in range(n_tickers):
        if valid_data[t_idx] and not np.isnan(closes[t_idx, d_idx]):
            current_close = closes[t_idx, d_idx]
            if not np.isnan(state.previous_closes[t_idx]):
                daily_return = (
                    current_close - state.previous_closes[t_idx]
                ) / state.previous_closes[t_idx]
                state.asset_returns[tickers[t_idx]].append(daily_return)
            state.previous_closes[t_idx] = current_close


def finalize_open_positions(
    state: SimulationState,
    sorted_dates: np.ndarray,
    tickers: list[str],
    n_tickers: int,
) -> None:
    """Add open positions to trades list at end of simulation."""
    for t_idx in range(n_tickers):
        if state.position_amounts[t_idx] > 0:
            state.trades_list.append(
                {
                    "ticker": tickers[t_idx],
                    "entry_date": sorted_dates[state.position_entry_dates[t_idx]],
                    "entry_price": state.position_entry_prices[t_idx],
                    "exit_date": None,
                    "exit_price": None,
                    "amount": state.position_amounts[t_idx],
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                    "is_whipsaw": False,
                    "commission_cost": 0.0,
                    "slippage_cost": 0.0,
                    "is_stop_loss": False,
                    "is_take_profit": False,
                    "exit_reason": "open",
                }
            )
