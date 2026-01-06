"""
Portfolio-level risk metrics calculation.

Provides comprehensive risk analysis including:
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Portfolio volatility
- Correlation analysis
- Position concentration
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics."""

    # Value at Risk
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR
    cvar_99: float  # 99% CVaR

    # Portfolio volatility
    portfolio_volatility: float  # Annualized portfolio volatility

    # Correlation
    avg_correlation: float | None  # Average correlation between assets (None if not calculated)
    max_correlation: float | None  # Maximum correlation between any two assets
    min_correlation: float | None  # Minimum correlation between any two assets

    # Position concentration
    max_position_pct: float | None  # Maximum position size as % of portfolio (None if not calculated)
    position_concentration: float | None  # Herfindahl-Hirschman Index (HHI)

    # Beta (if benchmark provided)
    portfolio_beta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "portfolio_volatility": self.portfolio_volatility,
            "avg_correlation": self.avg_correlation,
            "max_correlation": self.max_correlation,
            "min_correlation": self.min_correlation,
            "max_position_pct": self.max_position_pct,
            "position_concentration": self.position_concentration,
            "portfolio_beta": self.portfolio_beta,
        }


def calculate_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Value at Risk (VaR) using historical simulation.

    VaR represents the maximum expected loss over a given time period
    at a specified confidence level.

    Args:
        returns: Array of portfolio returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR value (negative value representing loss)
    """
    if len(returns) == 0:
        return 0.0

    # VaR is the negative of the percentile
    percentile = (1 - confidence_level) * 100
    var = -np.percentile(returns, percentile)

    return var


def calculate_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) using historical simulation.

    CVaR (also known as Expected Shortfall) is the expected loss
    given that the loss exceeds VaR.

    Args:
        returns: Array of portfolio returns
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)

    Returns:
        CVaR value (negative value representing expected loss)
    """
    if len(returns) == 0:
        return 0.0

    var = calculate_var(returns, confidence_level)
    # CVaR is the mean of returns below VaR threshold
    threshold = -var
    tail_losses = returns[returns <= threshold]

    if len(tail_losses) > 0:
        cvar = -np.mean(tail_losses)
    else:
        cvar = var

    return cvar


def calculate_portfolio_volatility(
    returns: np.ndarray,
    annualization_factor: int = 365,
) -> float:
    """
    Calculate annualized portfolio volatility.

    Args:
        returns: Array of portfolio returns
        annualization_factor: Factor to annualize volatility (default: 365 for daily)

    Returns:
        Annualized volatility as decimal (e.g., 0.15 for 15%)
    """
    if len(returns) == 0:
        return 0.0

    return np.std(returns) * np.sqrt(annualization_factor)


def calculate_portfolio_correlation(
    asset_returns: dict[str, np.ndarray],
) -> tuple[float, float, float, pd.DataFrame]:
    """
    Calculate correlation metrics between assets in portfolio.

    Args:
        asset_returns: Dictionary mapping ticker to returns array

    Returns:
        Tuple of (avg_correlation, max_correlation, min_correlation, correlation_matrix)
    """
    if len(asset_returns) < 2:
        return 0.0, 0.0, 0.0, pd.DataFrame()

    # Align returns by index (assuming same length or using common dates)
    returns_df = pd.DataFrame(asset_returns)

    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()

    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    correlations = correlation_matrix.where(mask).stack()

    if len(correlations) > 0:
        avg_correlation = correlations.mean()
        max_correlation = correlations.max()
        min_correlation = correlations.min()
    else:
        avg_correlation = max_correlation = min_correlation = 0.0

    return avg_correlation, max_correlation, min_correlation, correlation_matrix


def calculate_position_concentration(
    position_values: dict[str, float],
    total_portfolio_value: float,
) -> tuple[float, float]:
    """
    Calculate position concentration metrics.

    Args:
        position_values: Dictionary mapping ticker to position value
        total_portfolio_value: Total portfolio value

    Returns:
        Tuple of (max_position_pct, hhi)
        - max_position_pct: Maximum position as % of portfolio
        - hhi: Herfindahl-Hirschman Index (0-1, higher = more concentrated)
    """
    if total_portfolio_value == 0 or len(position_values) == 0:
        return 0.0, 0.0

    # Calculate position percentages
    position_pcts = {
        ticker: value / total_portfolio_value
        for ticker, value in position_values.items()
    }

    # Maximum position percentage
    max_position_pct = max(position_pcts.values()) if position_pcts else 0.0

    # Herfindahl-Hirschman Index (HHI)
    # Sum of squared position percentages
    hhi = sum(pct**2 for pct in position_pcts.values())

    return max_position_pct, hhi


def calculate_portfolio_risk_metrics(
    equity_curve: np.ndarray,
    daily_returns: np.ndarray,
    asset_returns: dict[str, np.ndarray] | None = None,
    position_values: dict[str, float] | None = None,
    total_portfolio_value: float | None = None,
    benchmark_returns: np.ndarray | None = None,
    annualization_factor: int = 365,
) -> PortfolioRiskMetrics:
    """
    Calculate comprehensive portfolio risk metrics.

    Args:
        equity_curve: Array of portfolio equity values
        daily_returns: Array of daily portfolio returns
        asset_returns: Dictionary mapping ticker to returns (for correlation)
        position_values: Dictionary mapping ticker to current position values
        total_portfolio_value: Total portfolio value (for concentration)
        benchmark_returns: Benchmark returns (for beta calculation)
        annualization_factor: Factor to annualize metrics

    Returns:
        PortfolioRiskMetrics object
    """
    # VaR and CVaR
    var_95 = calculate_var(daily_returns, 0.95)
    var_99 = calculate_var(daily_returns, 0.99)
    cvar_95 = calculate_cvar(daily_returns, 0.95)
    cvar_99 = calculate_cvar(daily_returns, 0.99)

    # Portfolio volatility
    portfolio_volatility = calculate_portfolio_volatility(
        daily_returns, annualization_factor
    )

    # Correlation metrics
    if asset_returns and len(asset_returns) >= 2:
        avg_corr, max_corr, min_corr, _ = calculate_portfolio_correlation(asset_returns)
    else:
        avg_corr = max_corr = min_corr = None

    # Position concentration
    if position_values and total_portfolio_value and len(position_values) > 0:
        max_pos_pct, hhi = calculate_position_concentration(
            position_values, total_portfolio_value
        )
    else:
        max_pos_pct = hhi = None

    # Beta (portfolio sensitivity to benchmark)
    portfolio_beta = None
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        if len(daily_returns) == len(benchmark_returns):
            # Calculate beta: Cov(portfolio, benchmark) / Var(benchmark)
            covariance = np.cov(daily_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            if benchmark_variance > 0:
                portfolio_beta = covariance / benchmark_variance

    return PortfolioRiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        portfolio_volatility=portfolio_volatility,
        avg_correlation=avg_corr,
        max_correlation=max_corr,
        min_correlation=min_corr,
        max_position_pct=max_pos_pct,
        position_concentration=hhi,
        portfolio_beta=portfolio_beta,
    )
