"""
Metrics calculation helpers.

Helper functions for performance metrics calculation.
Uses core metrics module for unified calculations.
"""

import numpy as np

from src.backtester.models import Trade
from src.metrics.core import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_total_return,
)

__all__ = [
    "calculate_return_metrics",
    "calculate_risk_metrics_from_equity",
    "calculate_trade_stats",
]


def calculate_return_metrics(
    equity_curve: np.ndarray,
    dates: np.ndarray,
    initial_capital: float,
) -> tuple[float, float]:
    """
    Calculate return-based metrics (total return, CAGR).

    Args:
        equity_curve: Daily equity values
        dates: Corresponding dates
        initial_capital: Initial capital

    Returns:
        Tuple of (total_return, cagr) in percentage
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0

    final = equity_curve[-1]
    total_return = calculate_total_return(initial_capital, final)

    total_days = (dates[-1] - dates[0]).days
    cagr = calculate_cagr(initial_capital, final, total_days)

    return total_return, cagr


def calculate_risk_metrics_from_equity(
    equity_curve: np.ndarray,
    cagr: float,
) -> tuple[float, float, float]:
    """
    Calculate risk metrics from equity curve.

    Args:
        equity_curve: Daily equity values
        cagr: CAGR in percentage

    Returns:
        Tuple of (mdd, calmar_ratio, sharpe_ratio)
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0, 0.0

    mdd = calculate_max_drawdown(equity_curve)
    calmar = calculate_calmar_ratio(cagr, mdd)

    returns = calculate_returns(equity_curve)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=252)

    return mdd, calmar, sharpe


def calculate_trade_stats(trades: list[Trade]) -> dict[str, float]:
    """
    Calculate trade statistics.

    Args:
        trades: List of trades

    Returns:
        Dictionary with trade stats
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_return": 0.0,
        }

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl < 0]

    total_trades = len(trades)
    winning_trades = len(winning)
    losing_trades = len(losing)
    win_rate = winning_trades / total_trades * 100

    if winning and losing:
        total_profit = sum(t.pnl for t in winning)
        total_loss = abs(sum(t.pnl for t in losing))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
    else:
        profit_factor = 0.0

    avg_trade_return = sum(t.pnl_pct for t in trades) / total_trades

    return {
        "total_trades": float(total_trades),
        "winning_trades": float(winning_trades),
        "losing_trades": float(losing_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_return,
    }
