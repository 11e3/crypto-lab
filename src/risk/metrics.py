"""
Portfolio-level risk metrics calculation.

Provides comprehensive risk analysis including:
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Portfolio volatility
- Correlation analysis
- Position concentration

Metrics layer: Layer 4 — Portfolio Risk.
Independent of the backtest engine layers.
Used by both vectorized and event-driven engines via src/backtester/metrics.py
and src/backtester/engine/metrics_calculator.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.risk.metrics_portfolio import (
    calculate_portfolio_correlation,
    calculate_position_concentration,
)
from src.risk.metrics_var import calculate_cvar, calculate_var

__all__ = [
    "PortfolioRiskMetrics",
    "calculate_var",
    "calculate_cvar",
    "calculate_portfolio_volatility",
    "calculate_portfolio_correlation",
    "calculate_position_concentration",
    "calculate_portfolio_risk_metrics",
]


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
    max_position_pct: (
        float | None
    )  # Maximum position size as % of portfolio (None if not calculated)
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

    return float(np.std(returns) * np.sqrt(annualization_factor))


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
    portfolio_volatility = calculate_portfolio_volatility(daily_returns, annualization_factor)

    # Correlation metrics - declare type first
    avg_corr: float | None
    max_corr: float | None
    min_corr: float | None
    if asset_returns and len(asset_returns) >= 2:
        avg_corr, max_corr, min_corr, _ = calculate_portfolio_correlation(asset_returns)
    else:
        avg_corr = None
        max_corr = None
        min_corr = None

    # Position concentration - declare type first
    max_pos_pct: float | None
    hhi: float | None
    if position_values and total_portfolio_value and len(position_values) > 0:
        max_pos_pct, hhi = calculate_position_concentration(position_values, total_portfolio_value)
    else:
        max_pos_pct = None
        hhi = None

    # Beta (portfolio sensitivity to benchmark)
    portfolio_beta: float | None = None
    if (
        benchmark_returns is not None
        and len(benchmark_returns) > 0
        and len(daily_returns) == len(benchmark_returns)
    ):
        # Calculate beta: Cov(portfolio, benchmark) / Var(benchmark)
        covariance: float = float(np.cov(daily_returns, benchmark_returns)[0, 1])
        benchmark_variance: float = float(np.var(benchmark_returns))
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
