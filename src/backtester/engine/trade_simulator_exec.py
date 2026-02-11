"""
Trade execution helpers for trade simulation.

Contains exit execution and whipsaw handling logic.
"""

from datetime import date

import numpy as np

from src.backtester.engine.trade_costs import TradeCostCalculator
from src.backtester.engine.trade_simulator_state import SimulationState
from src.backtester.models import BacktestConfig
from src.orders.advanced_orders import AdvancedOrderManager


def execute_exit(
    state: SimulationState,
    config: BacktestConfig,
    t_idx: int,
    d_idx: int,
    current_date: date,
    sorted_dates: np.ndarray,
    tickers: list[str],
    exit_prices: np.ndarray,
    order_manager: AdvancedOrderManager,
    is_stop_loss: bool,
    is_take_profit: bool,
    exit_reason: str,
) -> None:
    """Execute a single exit."""
    exit_price = exit_prices[t_idx, d_idx]
    amount = state.position_amounts[t_idx]
    entry_price = state.position_entry_prices[t_idx]

    calculator = TradeCostCalculator(config.fee_rate, config.slippage_rate)
    costs = calculator.calculate_exit_costs(entry_price, exit_price, amount)

    state.cash += costs.revenue

    state.trades_list.append(
        {
            "ticker": tickers[t_idx],
            "entry_date": sorted_dates[state.position_entry_dates[t_idx]],
            "entry_price": entry_price,
            "exit_date": current_date,
            "exit_price": exit_price,
            "amount": amount,
            "pnl": costs.pnl,
            "pnl_pct": costs.pnl_pct,
            "is_whipsaw": False,
            "commission_cost": costs.commission,
            "slippage_cost": costs.slippage,
            "is_stop_loss": is_stop_loss,
            "is_take_profit": is_take_profit,
            "exit_reason": exit_reason,
        }
    )

    order_manager.cancel_all_orders(ticker=tickers[t_idx])
    state.position_amounts[t_idx] = 0
    state.position_entry_prices[t_idx] = 0
    state.position_entry_dates[t_idx] = -1


def handle_whipsaw(
    state: SimulationState,
    t_idx: int,
    d_idx: int,
    current_date: date,
    tickers: list[str],
    arrays: dict[str, np.ndarray],
    invest_amount: float,
    buy_price: float,
    fee_rate: float,
    slippage_rate: float,
) -> None:
    """Handle whipsaw (same-day entry and exit)."""
    sell_price = arrays["exit_prices"][t_idx, d_idx]

    calculator = TradeCostCalculator(fee_rate, slippage_rate)
    costs = calculator.calculate_whipsaw_costs(buy_price, sell_price, invest_amount)

    state.cash = state.cash - invest_amount + costs.revenue

    state.trades_list.append(
        {
            "ticker": tickers[t_idx],
            "entry_date": current_date,
            "entry_price": buy_price,
            "exit_date": current_date,
            "exit_price": sell_price,
            "amount": costs.net_amount,
            "pnl": costs.pnl,
            "pnl_pct": costs.pnl_pct,
            "is_whipsaw": True,
            "commission_cost": costs.commission,
            "slippage_cost": costs.slippage,
            "is_stop_loss": False,
            "is_take_profit": False,
            "exit_reason": "whipsaw",
        }
    )


def handle_normal_entry(
    state: SimulationState,
    t_idx: int,
    d_idx: int,
    invest_amount: float,
    buy_price: float,
    fee_rate: float,
) -> None:
    """Handle normal entry (position opened)."""
    calculator = TradeCostCalculator(fee_rate)
    amount = calculator.calculate_buy_amount(invest_amount, buy_price)

    state.position_amounts[t_idx] = amount
    state.position_entry_prices[t_idx] = buy_price
    state.position_entry_dates[t_idx] = d_idx
    state.cash -= invest_amount
