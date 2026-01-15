"""
Risk-related metrics calculations.

Handles volatility, VaR, and CVaR (SRP).
"""

import numpy as np


class RiskMetrics:
    """Calculator for risk-related metrics."""

    TRADING_DAYS_CRYPTO = 365  # Crypto markets are 24/7

    @staticmethod
    def calculate_volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(returns) < 2:
            return 0.0

        vol = float(np.std(returns, ddof=1))
        if annualize:
            vol *= np.sqrt(RiskMetrics.TRADING_DAYS_CRYPTO)
        return vol * 100

    @staticmethod
    def calculate_upside_volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate upside volatility (positive returns only)."""
        positive_returns = returns[returns > 0]
        if len(positive_returns) < 2:
            return 0.0

        vol = float(np.std(positive_returns, ddof=1))
        if annualize:
            vol *= np.sqrt(RiskMetrics.TRADING_DAYS_CRYPTO)
        return vol * 100

    @staticmethod
    def calculate_downside_volatility(
        returns: np.ndarray,
        mar: float = 0.0,
        annualize: bool = True,
    ) -> float:
        """Calculate downside volatility (Downside Deviation).

        Args:
            returns: Return array
            mar: Minimum Acceptable Return (default: 0)
            annualize: Whether to annualize
        """
        if len(returns) < 2:
            return 0.0

        downside_returns = returns[returns < mar]
        if len(downside_returns) < 2:
            return 0.0

        vol = float(np.std(downside_returns, ddof=1))
        if annualize:
            vol *= np.sqrt(RiskMetrics.TRADING_DAYS_CRYPTO)
        return vol * 100

    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 2:
            return 0.0

        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(float(var)) * 100

    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(returns) < 2:
            return 0.0

        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return RiskMetrics.calculate_var(returns, confidence)

        return abs(float(np.mean(tail_returns))) * 100
