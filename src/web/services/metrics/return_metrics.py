"""
Return-related metrics calculations.

Handles returns, CAGR, and maximum drawdown (SRP).
"""

import numpy as np


class ReturnMetrics:
    """Calculator for return-related metrics."""

    @staticmethod
    def calculate_returns(equity: np.ndarray) -> np.ndarray:
        """Calculate daily returns from equity curve."""
        if len(equity) < 2:
            return np.array([])
        return np.diff(equity) / equity[:-1]

    @staticmethod
    def calculate_total_return(
        initial_value: float,
        final_value: float,
    ) -> float:
        """Calculate total return percentage."""
        if initial_value <= 0:
            return 0.0
        return (final_value / initial_value - 1) * 100

    @staticmethod
    def calculate_cagr(
        initial_value: float,
        final_value: float,
        years: float,
    ) -> float:
        """Calculate Compound Annual Growth Rate."""
        if years <= 0 or initial_value <= 0:
            return 0.0
        return float(((final_value / initial_value) ** (1 / years) - 1) * 100)

    @staticmethod
    def calculate_max_drawdown(equity: np.ndarray) -> float:
        """Calculate maximum drawdown percentage."""
        if len(equity) < 2:
            return 0.0

        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return abs(float(np.min(drawdown))) * 100
