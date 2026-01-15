"""
Risk-adjusted ratio calculations.

Handles Sharpe, Sortino, and Calmar ratios (SRP).
"""

import numpy as np


class RatioMetrics:
    """Calculator for risk-adjusted ratios."""

    TRADING_DAYS_CRYPTO = 365

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate Sharpe Ratio."""
        if len(returns) < 2:
            return 0.0

        daily_rf = (1 + risk_free_rate) ** (1 / RatioMetrics.TRADING_DAYS_CRYPTO) - 1
        excess_returns = returns - daily_rf

        std = np.std(excess_returns, ddof=1)
        if std == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / std
        return float(sharpe * np.sqrt(RatioMetrics.TRADING_DAYS_CRYPTO))

    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate Sortino Ratio."""
        if len(returns) < 2:
            return 0.0

        daily_rf = (1 + risk_free_rate) ** (1 / RatioMetrics.TRADING_DAYS_CRYPTO) - 1
        excess_returns = returns - daily_rf

        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            return float("inf") if np.mean(excess_returns) > 0 else 0.0

        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_std
        return float(sortino * np.sqrt(RatioMetrics.TRADING_DAYS_CRYPTO))

    @staticmethod
    def calculate_calmar_ratio(cagr: float, max_dd: float) -> float:
        """Calculate Calmar Ratio."""
        if max_dd == 0:
            return 0.0
        return cagr / max_dd
