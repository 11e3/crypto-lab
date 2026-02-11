"""
Portfolio optimization methods.

Implements:
- Modern Portfolio Theory (MPT): Mean-variance optimization
- Risk Parity: Equal risk contribution from each asset
- Kelly Criterion: Optimal position sizing based on win rate and payoff ratio
"""

from typing import Any

import pandas as pd

from src.risk.portfolio_methods import (
    calculate_kelly_criterion,
    optimize_kelly_portfolio,
    optimize_mpt,
    optimize_risk_parity,
)
from src.risk.portfolio_models import OptimizationMethod, PortfolioWeights


class PortfolioOptimizer:
    """
    Portfolio optimization using various methods.

    Supports Modern Portfolio Theory, Risk Parity, and Kelly Criterion.

    Example:
        >>> optimizer = PortfolioOptimizer()
        >>> weights = optimizer.optimize_mpt(returns_df)
        >>> print(weights.weights)
    """

    def __init__(self) -> None:
        """Initialize portfolio optimizer."""
        pass

    def optimize_mpt(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        target_return: float | None = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> PortfolioWeights:
        """Optimize portfolio using Modern Portfolio Theory."""
        return optimize_mpt(
            returns=returns,
            risk_free_rate=risk_free_rate,
            target_return=target_return,
            max_weight=max_weight,
            min_weight=min_weight,
        )

    def optimize_risk_parity(
        self,
        returns: pd.DataFrame,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> PortfolioWeights:
        """Optimize portfolio using Risk Parity (equal risk contribution)."""
        return optimize_risk_parity(
            returns=returns,
            max_weight=max_weight,
            min_weight=min_weight,
        )

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_kelly: float = 0.25,
    ) -> float:
        """Calculate Kelly Criterion for optimal position sizing."""
        return calculate_kelly_criterion(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_kelly=max_kelly,
        )

    def optimize_kelly_portfolio(
        self,
        trades: pd.DataFrame,
        available_cash: float,
        max_kelly: float = 0.25,
    ) -> dict[str, float]:
        """Calculate portfolio allocation using Kelly Criterion."""
        return optimize_kelly_portfolio(
            trades=trades,
            available_cash=available_cash,
            max_kelly=max_kelly,
        )


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str | OptimizationMethod = OptimizationMethod.MPT,
    **kwargs: Any,
) -> PortfolioWeights:
    """
    Convenience function to optimize portfolio.

    Args:
        returns: DataFrame with returns (columns = tickers, index = dates)
        method: Optimization method (OptimizationMethod enum or string)
        **kwargs: Additional arguments for optimization

    Returns:
        PortfolioWeights with optimized allocation
    """
    optimizer = PortfolioOptimizer()

    if method == OptimizationMethod.MPT:
        return optimizer.optimize_mpt(returns, **kwargs)
    elif method == OptimizationMethod.RISK_PARITY:
        return optimizer.optimize_risk_parity(returns, **kwargs)
    elif method == OptimizationMethod.KELLY:
        if "trades" not in kwargs:
            raise ValueError("Kelly method requires 'trades' DataFrame")
        trades = kwargs.pop("trades")
        available_cash = kwargs.pop("available_cash", 1.0)
        max_kelly = kwargs.pop("max_kelly", 0.25)
        allocations = optimizer.optimize_kelly_portfolio(trades, available_cash, max_kelly)
        total = sum(allocations.values())
        weights = {t: a / total if total > 0 else 0.0 for t, a in allocations.items()}
        return PortfolioWeights(weights=weights, method="kelly")
    else:
        raise ValueError(f"Unknown optimization method: {method}")


__all__ = [
    "OptimizationMethod",
    "PortfolioWeights",
    "PortfolioOptimizer",
    "optimize_portfolio",
]
