"""
Core metric calculation functions.

Single source of truth for MDD, Sharpe, CAGR, daily returns, Calmar, and Sortino.
All functions are pure math (numpy + scalars), no pandas or domain model dependencies.
"""

from __future__ import annotations

import numpy as np

from src.config import ANNUALIZATION_FACTOR, RISK_FREE_RATE

# Cap for extreme CAGR values to prevent overflow in downstream calculations
_MAX_CAGR_PCT = 99999.0

__all__ = [
    "calculate_calmar_ratio",
    "calculate_cagr",
    "calculate_daily_returns",
    "calculate_drawdown_series",
    "calculate_mdd",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
]


def calculate_drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """Calculate drawdown fraction series from equity curve.

    Args:
        equity_curve: Daily equity values.

    Returns:
        Array of drawdown fractions (0 to 1 range, 0 = no drawdown).
    """
    if len(equity_curve) < 2:
        return np.zeros_like(equity_curve)
    cummax = np.maximum.accumulate(equity_curve)
    drawdown: np.ndarray = (cummax - equity_curve) / cummax
    return drawdown


def calculate_mdd(equity_curve: np.ndarray) -> float:
    """Calculate Maximum Drawdown as a positive percentage.

    Args:
        equity_curve: Daily equity values.

    Returns:
        MDD as positive percentage (e.g. 15.3 means 15.3% drawdown).
        Returns 0.0 if fewer than 2 data points.
    """
    if len(equity_curve) < 2:
        return 0.0
    drawdown = calculate_drawdown_series(equity_curve)
    return float(np.nanmax(drawdown) * 100)


def calculate_daily_returns(
    equity_curve: np.ndarray,
    *,
    prepend_zero: bool = False,
) -> np.ndarray:
    """Calculate daily returns from equity curve.

    Args:
        equity_curve: Daily equity values.
        prepend_zero: If True, prepend a 0.0 return so output length matches input.

    Returns:
        Array of daily returns.
    """
    if len(equity_curve) < 2:
        if prepend_zero:
            return np.zeros(len(equity_curve))
        return np.array([], dtype=np.float64)
    returns: np.ndarray = np.diff(equity_curve) / equity_curve[:-1]
    if prepend_zero:
        returns = np.insert(returns, 0, 0.0)
    return returns


def calculate_sharpe_ratio(
    returns: np.ndarray,
    annualization_factor: float = ANNUALIZATION_FACTOR,
    *,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Calculate annualized Sharpe ratio.

    Uses excess returns (returns - daily risk-free rate) and sample std (ddof=1).

    Args:
        returns: Array of period returns.
        annualization_factor: Trading days per year (365 for crypto, 252 for stocks).
        risk_free_rate: Annual risk-free rate (converted to daily internally).

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if fewer than 2 data points or zero std.
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    excess_returns = returns - daily_rf
    std = float(np.std(excess_returns, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(excess_returns) / std * np.sqrt(annualization_factor))


def calculate_cagr(
    initial_value: float,
    final_value: float,
    total_days: int | float,
) -> float:
    """Calculate Compound Annual Growth Rate as percentage.

    Uses log-based formula for numerical stability with overflow guard.

    Args:
        initial_value: Starting portfolio value.
        final_value: Ending portfolio value.
        total_days: Number of calendar days.

    Returns:
        CAGR as percentage (e.g. 12.5 means 12.5%).
        Returns 0.0 if total_days <= 0 or initial_value <= 0.
        Returns -100.0 if final_value <= 0.
    """
    if total_days <= 0 or initial_value <= 0:
        return 0.0
    if final_value <= 0:
        return -100.0
    ratio = final_value / initial_value
    if ratio <= 0:
        return -100.0
    with np.errstate(over="ignore"):
        cagr_raw = (np.exp((365.0 / total_days) * np.log(ratio)) - 1) * 100
    if np.isinf(cagr_raw) or abs(cagr_raw) > _MAX_CAGR_PCT:
        return _MAX_CAGR_PCT if cagr_raw > 0 else -_MAX_CAGR_PCT
    return float(cagr_raw)


def calculate_calmar_ratio(cagr_pct: float, mdd_pct: float) -> float:
    """Calculate Calmar ratio (CAGR / MDD).

    Args:
        cagr_pct: CAGR as percentage.
        mdd_pct: MDD as positive percentage.

    Returns:
        Calmar ratio. Returns 0.0 if MDD <= 0.
    """
    if mdd_pct <= 0:
        return 0.0
    return cagr_pct / mdd_pct


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    annualization_factor: float = ANNUALIZATION_FACTOR,
) -> float:
    """Calculate Sortino ratio using downside semi-deviation.

    Uses excess returns (returns - daily risk-free rate), sample std (ddof=1),
    and semi-deviation (all returns clamped to min(excess, 0)).

    Args:
        returns: Array of period returns.
        risk_free_rate: Annual risk-free rate (converted to daily internally).
        annualization_factor: Annualization factor.

    Returns:
        Sortino ratio. Returns 0.0 if fewer than 2 data points or no downside deviation.
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    excess_returns = returns - daily_rf
    downside_returns = np.minimum(excess_returns, 0)
    downside_std = float(np.std(downside_returns, ddof=1))
    if downside_std <= 0:
        return 0.0
    return float(np.mean(excess_returns) / downside_std * np.sqrt(annualization_factor))
