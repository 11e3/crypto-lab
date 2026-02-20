"""
Metrics calculation helpers.

Helper functions for performance metrics calculation.

Metrics layer: Layer 2 — Trade Statistics.
Wraps src/utils/metrics_core.py (Layer 1) and adds pandas + Trade model support.
Called by: src/backtester/metrics.py (event-driven engine aggregator).
"""

import numpy as np

from src.backtester.models import Trade
from src.utils.metrics_core import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_daily_returns,
    calculate_mdd,
    calculate_sharpe_ratio,
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
    total_return = (final / initial_capital - 1) * 100

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

    mdd = calculate_mdd(equity_curve)
    calmar = calculate_calmar_ratio(cagr, mdd)
    returns = calculate_daily_returns(equity_curve)
    sharpe = calculate_sharpe_ratio(returns)

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

    _sentinel = 999.99  # Sentinel for "no losers" — avoids inf in JSON/display
    if winning and losing:
        total_profit = sum(t.pnl for t in winning)
        total_loss = abs(sum(t.pnl for t in losing))
        profit_factor = total_profit / total_loss if total_loss > 0 else _sentinel
    elif winning:
        profit_factor = _sentinel  # All winning trades
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
