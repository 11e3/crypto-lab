"""
Statistical metrics calculations.

Handles z-score, p-value, skewness, kurtosis, and trade metrics (SRP).
"""

import numpy as np
from scipy import stats


class StatisticalMetrics:
    """Calculator for statistical metrics."""

    @staticmethod
    def calculate_z_score_and_pvalue(returns: np.ndarray) -> tuple[float, float]:
        """Calculate z-score and p-value.

        Returns:
            (z_score, p_value) tuple
        """
        if len(returns) < 2:
            return 0.0, 1.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0, 1.0

        # H0: mean return = 0
        z_score = mean_return / (std_return / np.sqrt(len(returns)))

        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return float(z_score), float(p_value)

    @staticmethod
    def calculate_skewness(returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        return float(stats.skew(returns))

    @staticmethod
    def calculate_kurtosis(returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 3:
            return 0.0
        return float(stats.kurtosis(returns))


class TradeMetrics:
    """Calculator for trade-specific metrics."""

    @staticmethod
    def calculate(
        trade_returns: list[float],
    ) -> tuple[float, float, float, float, float]:
        """Calculate trade metrics.

        Returns:
            (win_rate, avg_win, avg_loss, profit_factor, expectancy)
        """
        if not trade_returns:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r < 0]

        win_rate = len(wins) / len(trade_returns) * 100
        avg_win = sum(wins) / len(wins) * 100 if wins else 0.0
        avg_loss = sum(losses) / len(losses) * 100 if losses else 0.0

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        expectancy = sum(trade_returns) / len(trade_returns) * 100

        return win_rate, avg_win, avg_loss, profit_factor, expectancy
