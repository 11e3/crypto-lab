"""Performance metrics calculation for backtest reports.

Re-exports PerformanceMetrics for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtester.report_pkg.report_metrics_models import PerformanceMetrics
from src.backtester.report_pkg.report_metrics_trade import calculate_trade_statistics
from src.backtester.report_pkg.report_returns import (
    calculate_monthly_returns,
    calculate_yearly_returns,
)
from src.config import ANNUALIZATION_FACTOR
from src.utils.metrics_core import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_daily_returns,
    calculate_drawdown_series,
    calculate_mdd,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)

__all__ = [
    "PerformanceMetrics",
    "calculate_metrics",
    "calculate_sortino_ratio",
    "calculate_monthly_returns",
    "calculate_yearly_returns",
    "metrics_to_dataframe",
]


def calculate_metrics(
    equity_curve: np.ndarray,
    dates: np.ndarray,
    trades_df: pd.DataFrame,
    initial_capital: float = 1.0,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: Array of equity values
        dates: Array of dates
        trades_df: DataFrame with trade records
        initial_capital: Starting capital

    Returns:
        PerformanceMetrics object
    """
    start_date = dates[0]
    end_date = dates[-1]
    total_days = (end_date - start_date).days

    final_equity = equity_curve[-1]
    total_return_pct = (final_equity / initial_capital - 1) * 100

    cagr_pct = calculate_cagr(initial_capital, final_equity, total_days)
    daily_returns = calculate_daily_returns(equity_curve, prepend_zero=True)
    volatility_pct = np.std(daily_returns) * np.sqrt(ANNUALIZATION_FACTOR) * 100

    drawdown = calculate_drawdown_series(equity_curve)
    mdd_pct = calculate_mdd(equity_curve)

    sharpe_ratio = calculate_sharpe_ratio(daily_returns, ANNUALIZATION_FACTOR)
    sortino_ratio = calculate_sortino_ratio(daily_returns)
    calmar_ratio = calculate_calmar_ratio(cagr_pct, mdd_pct)

    trade_stats = calculate_trade_statistics(trades_df)

    return PerformanceMetrics(
        start_date=start_date,
        end_date=end_date,
        total_days=total_days,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        mdd_pct=mdd_pct,
        volatility_pct=volatility_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        total_trades=int(trade_stats["total_trades"]),
        winning_trades=int(trade_stats["winning_trades"]),
        losing_trades=int(trade_stats["losing_trades"]),
        win_rate_pct=float(trade_stats["win_rate_pct"]),
        profit_factor=float(trade_stats["profit_factor"]),
        avg_profit_pct=float(trade_stats["avg_profit_pct"]),
        avg_loss_pct=float(trade_stats["avg_loss_pct"]),
        avg_trade_pct=float(trade_stats["avg_trade_pct"]),
        equity_curve=equity_curve,
        drawdown_curve=drawdown * 100,
        dates=dates,
        daily_returns=daily_returns,
    )


def metrics_to_dataframe(metrics: PerformanceMetrics) -> pd.DataFrame:
    """Export PerformanceMetrics as DataFrame."""
    m = metrics
    return pd.DataFrame(
        {
            "Metric": [
                "Start Date",
                "End Date",
                "Total Days",
                "Total Return (%)",
                "CAGR (%)",
                "Max Drawdown (%)",
                "Volatility (%)",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Total Trades",
                "Winning Trades",
                "Losing Trades",
                "Win Rate (%)",
                "Profit Factor",
                "Avg Profit (%)",
                "Avg Loss (%)",
                "Avg Trade (%)",
            ],
            "Value": [
                str(m.start_date),
                str(m.end_date),
                m.total_days,
                round(m.total_return_pct, 2),
                round(m.cagr_pct, 2),
                round(m.mdd_pct, 2),
                round(m.volatility_pct, 2),
                round(m.sharpe_ratio, 2),
                round(m.sortino_ratio, 2),
                round(m.calmar_ratio, 2),
                m.total_trades,
                m.winning_trades,
                m.losing_trades,
                round(m.win_rate_pct, 2),
                round(m.profit_factor, 2),
                round(m.avg_profit_pct, 2),
                round(m.avg_loss_pct, 2),
                round(m.avg_trade_pct, 2),
            ],
        }
    )
