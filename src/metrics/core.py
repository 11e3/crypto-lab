"""
Core metrics calculation functions.

Unified implementation of financial metrics used across the system.
All functions are pure, stateless, and operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np

# Default annualization factor (crypto markets: 365 days)
ANNUALIZATION_FACTOR = 365


def calculate_returns(equity: np.ndarray) -> np.ndarray:
    """
    Calculate daily returns from equity curve.

    Args:
        equity: Portfolio value array

    Returns:
        Daily returns array (length = len(equity) - 1)
    """
    if len(equity) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(equity) / equity[:-1]


def calculate_total_return(
    initial_value: float,
    final_value: float,
) -> float:
    """
    Calculate total return percentage.

    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value

    Returns:
        Total return in percentage
    """
    if initial_value <= 0:
        return 0.0
    return (final_value / initial_value - 1) * 100


def calculate_cagr(
    initial_value: float,
    final_value: float,
    days: int,
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        days: Number of days in period

    Returns:
        CAGR in percentage
    """
    if days <= 0 or initial_value <= 0 or final_value <= 0:
        return 0.0
    years = days / 365.0
    return ((final_value / initial_value) ** (1.0 / years) - 1) * 100


def calculate_max_drawdown(equity: np.ndarray) -> float:
    """
    Calculate Maximum Drawdown.

    Args:
        equity: Portfolio value array

    Returns:
        Maximum drawdown in percentage (positive value)
    """
    if len(equity) < 2:
        return 0.0

    cummax = np.maximum.accumulate(equity)
    drawdown = (cummax - equity) / cummax
    return float(np.nanmax(drawdown)) * 100


def calculate_volatility(
    returns: np.ndarray,
    annualize: bool = True,
    annualization_factor: int = ANNUALIZATION_FACTOR,
) -> float:
    """
    Calculate return volatility (standard deviation).

    Args:
        returns: Daily returns array
        annualize: Whether to annualize the volatility
        annualization_factor: Days per year for annualization

    Returns:
        Volatility in percentage
    """
    if len(returns) < 2:
        return 0.0

    vol = float(np.std(returns, ddof=1))
    if annualize:
        vol *= np.sqrt(annualization_factor)
    return vol * 100


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = ANNUALIZATION_FACTOR,
) -> float:
    """
    Calculate Sharpe Ratio.

    Args:
        returns: Daily returns array
        risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        annualization_factor: Days per year for annualization

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    excess_returns = returns - daily_rf

    std = np.std(excess_returns, ddof=1)
    if std == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / std
    return float(sharpe) * np.sqrt(annualization_factor)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = ANNUALIZATION_FACTOR,
) -> float:
    """
    Calculate Sortino Ratio (using downside deviation).

    Args:
        returns: Daily returns array
        risk_free_rate: Annual risk-free rate
        annualization_factor: Days per year for annualization

    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    excess_returns = returns - daily_rf

    # Downside returns only
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 2:
        return float("inf") if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0:
        return 0.0

    sortino = np.mean(excess_returns) / downside_std
    return float(sortino) * np.sqrt(annualization_factor)


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """
    Calculate Calmar Ratio (CAGR / MDD).

    Args:
        cagr: CAGR in percentage
        max_drawdown: Maximum drawdown in percentage

    Returns:
        Calmar Ratio
    """
    if max_drawdown == 0:
        return 0.0
    return cagr / max_drawdown


def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Value at Risk.

    Args:
        returns: Daily returns array
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        VaR in percentage (positive value representing potential loss)
    """
    if len(returns) < 2:
        return 0.0

    var = np.percentile(returns, (1 - confidence) * 100)
    return abs(float(var)) * 100


def calculate_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).

    Args:
        returns: Daily returns array
        confidence: Confidence level

    Returns:
        CVaR in percentage (positive value)
    """
    if len(returns) < 2:
        return 0.0

    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) == 0:
        return calculate_var(returns, confidence)

    return abs(float(np.mean(tail_returns))) * 100


def calculate_downside_volatility(
    returns: np.ndarray,
    mar: float = 0.0,
    annualize: bool = True,
    annualization_factor: int = ANNUALIZATION_FACTOR,
) -> float:
    """
    Calculate Downside Deviation (volatility of negative returns).

    Args:
        returns: Daily returns array
        mar: Minimum Acceptable Return
        annualize: Whether to annualize
        annualization_factor: Days per year

    Returns:
        Downside volatility in percentage
    """
    if len(returns) < 2:
        return 0.0

    downside_returns = returns[returns < mar]
    if len(downside_returns) < 2:
        return 0.0

    vol = float(np.std(downside_returns, ddof=1))
    if annualize:
        vol *= np.sqrt(annualization_factor)
    return vol * 100


def calculate_upside_volatility(
    returns: np.ndarray,
    annualize: bool = True,
    annualization_factor: int = ANNUALIZATION_FACTOR,
) -> float:
    """
    Calculate Upside Volatility (volatility of positive returns).

    Args:
        returns: Daily returns array
        annualize: Whether to annualize
        annualization_factor: Days per year

    Returns:
        Upside volatility in percentage
    """
    positive_returns = returns[returns > 0]
    if len(positive_returns) < 2:
        return 0.0

    vol = float(np.std(positive_returns, ddof=1))
    if annualize:
        vol *= np.sqrt(annualization_factor)
    return vol * 100
