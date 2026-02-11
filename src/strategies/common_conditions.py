"""
Shared indicator-based conditions used across multiple strategy modules.

These are strategy-agnostic conditions based on common indicators (SMA, RSI).
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition


class PriceAboveSMACondition(Condition):
    """Condition: Current close price is above SMA."""

    def __init__(self, sma_key: str = "sma", name: str = "PriceAboveSMA") -> None:
        super().__init__(name)
        self.sma_key = sma_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if close is above SMA."""
        sma = indicators.get(self.sma_key)

        if sma is None:
            return False

        return current.close > sma


class PriceBelowSMACondition(Condition):
    """Condition: Current close price is below SMA."""

    def __init__(self, sma_key: str = "sma", name: str = "PriceBelowSMA") -> None:
        super().__init__(name)
        self.sma_key = sma_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if close is below SMA."""
        sma = indicators.get(self.sma_key)

        if sma is None:
            return False

        return current.close < sma


class RSIOversoldCondition(Condition):
    """Condition: RSI is below oversold threshold."""

    def __init__(
        self,
        rsi_key: str = "rsi",
        oversold_threshold: float = 30.0,
        name: str = "RSIOversold",
    ) -> None:
        super().__init__(name)
        self.rsi_key = rsi_key
        self.oversold_threshold = oversold_threshold

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if RSI is oversold."""
        rsi = indicators.get(self.rsi_key)

        if rsi is None:
            return False

        return rsi < self.oversold_threshold


class RSIOverboughtCondition(Condition):
    """Condition: RSI is above overbought threshold."""

    def __init__(
        self,
        rsi_key: str = "rsi",
        overbought_threshold: float = 70.0,
        name: str = "RSIOverbought",
    ) -> None:
        super().__init__(name)
        self.rsi_key = rsi_key
        self.overbought_threshold = overbought_threshold

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if RSI is overbought."""
        rsi = indicators.get(self.rsi_key)

        if rsi is None:
            return False

        return rsi > self.overbought_threshold


__all__ = [
    "PriceAboveSMACondition",
    "PriceBelowSMACondition",
    "RSIOversoldCondition",
    "RSIOverboughtCondition",
]
