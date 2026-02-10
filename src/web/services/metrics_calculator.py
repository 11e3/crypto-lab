"""Extended metrics calculator service.

Calculate advanced backtesting metrics including Sortino, Calmar, VaR, CVaR,
upside/downside volatility, z-score, p-value, and more.

Uses focused metric calculators for each domain (SRP).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.metrics_core import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_daily_returns,
    calculate_mdd,
)
from src.web.services.metrics import (
    RatioMetrics,
    RiskMetrics,
    StatisticalMetrics,
    TradeMetrics,
)

__all__ = [
    "ExtendedMetrics",
    "calculate_extended_metrics",
    # Re-export metric calculators
    "RiskMetrics",
    "RatioMetrics",
    "StatisticalMetrics",
    "TradeMetrics",
]


@dataclass(frozen=True)
class ExtendedMetrics:
    """Extended backtesting metrics."""

    # Basic return metrics
    total_return_pct: float
    cagr_pct: float

    # Risk metrics
    max_drawdown_pct: float
    volatility_pct: float
    upside_volatility_pct: float
    downside_volatility_pct: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # VaR & CVaR
    var_95_pct: float
    var_99_pct: float
    cvar_95_pct: float
    cvar_99_pct: float

    # Statistical tests
    z_score: float
    p_value: float
    skewness: float
    kurtosis: float

    # Trade metrics
    num_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy: float

    # Period information
    trading_days: int
    years: float


def calculate_extended_metrics(
    equity: np.ndarray,
    trade_returns: list[float] | None = None,
    risk_free_rate: float = 0.02,
    dates: np.ndarray | None = None,
) -> ExtendedMetrics:
    """Calculate extended metrics.

    Args:
        equity: Portfolio value array
        trade_returns: Individual trade returns list (optional)
        risk_free_rate: Risk-free rate (annual, default: 2%)
        dates: Date array for accurate period calculation (optional)

    Returns:
        ExtendedMetrics dataclass
    """
    if len(equity) < 2:
        return _empty_metrics()

    # Period information
    trading_days = len(equity)
    if dates is not None and len(dates) >= 2:
        # Calculate years from actual date range
        import pandas as pd

        first_date = pd.to_datetime(dates[0])
        last_date = pd.to_datetime(dates[-1])
        years = (last_date - first_date).days / 365.0
    else:
        # Fallback to equity length / 365
        years = trading_days / 365.0

    # Daily returns
    returns = calculate_daily_returns(equity)

    # Return metrics
    initial_value = float(equity[0])
    final_value = float(equity[-1])
    total_return = (final_value / initial_value - 1) * 100 if initial_value > 0 else 0.0
    cagr = calculate_cagr(initial_value, final_value, years * 365)
    max_dd = calculate_mdd(equity)

    # Risk metrics
    volatility = RiskMetrics.calculate_volatility(returns)
    upside_vol = RiskMetrics.calculate_upside_volatility(returns)
    downside_vol = RiskMetrics.calculate_downside_volatility(returns)
    var_95 = RiskMetrics.calculate_var(returns, 0.95)
    var_99 = RiskMetrics.calculate_var(returns, 0.99)
    cvar_95 = RiskMetrics.calculate_cvar(returns, 0.95)
    cvar_99 = RiskMetrics.calculate_cvar(returns, 0.99)

    # Risk-adjusted returns
    sharpe = RatioMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = RatioMetrics.calculate_sortino_ratio(returns, risk_free_rate)
    calmar = calculate_calmar_ratio(cagr, max_dd)

    # Statistical metrics
    z_score, p_value = StatisticalMetrics.calculate_z_score_and_pvalue(returns)
    skewness = StatisticalMetrics.calculate_skewness(returns)
    kurtosis = StatisticalMetrics.calculate_kurtosis(returns)

    # Trade metrics
    trade_returns = trade_returns or []
    num_trades = len(trade_returns)
    win_rate, avg_win, avg_loss, profit_factor, expectancy = TradeMetrics.calculate(trade_returns)

    return ExtendedMetrics(
        total_return_pct=total_return,
        cagr_pct=cagr,
        max_drawdown_pct=max_dd,
        volatility_pct=volatility,
        upside_volatility_pct=upside_vol,
        downside_volatility_pct=downside_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        var_95_pct=var_95,
        var_99_pct=var_99,
        cvar_95_pct=cvar_95,
        cvar_99_pct=cvar_99,
        z_score=z_score,
        p_value=p_value,
        skewness=skewness,
        kurtosis=kurtosis,
        num_trades=num_trades,
        win_rate_pct=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        trading_days=trading_days,
        years=years,
    )


def _empty_metrics() -> ExtendedMetrics:
    """Return empty metrics for insufficient data."""
    return ExtendedMetrics(
        total_return_pct=0.0,
        cagr_pct=0.0,
        max_drawdown_pct=0.0,
        volatility_pct=0.0,
        upside_volatility_pct=0.0,
        downside_volatility_pct=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        var_95_pct=0.0,
        var_99_pct=0.0,
        cvar_95_pct=0.0,
        cvar_99_pct=0.0,
        z_score=0.0,
        p_value=1.0,
        skewness=0.0,
        kurtosis=0.0,
        num_trades=0,
        win_rate_pct=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        trading_days=0,
        years=0.0,
    )
