"""
RSI-based conditions for Mean Reversion strategy.

Re-exports shared RSI conditions and defines mean-reversion-specific ones.
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition
from src.strategies.common_conditions import (
    RSIOverboughtCondition,
    RSIOversoldCondition,
)

__all__ = [
    "RSIOversoldCondition",
    "RSIOverboughtCondition",
    "MeanReversionStrengthCondition",
]


class MeanReversionStrengthCondition(Condition):
    """
    Condition: Mean reversion strength based on deviation from mean.

    Filters for strong mean reversion opportunities.
    """

    def __init__(
        self,
        sma_key: str = "sma",
        min_deviation_pct: float = 0.02,
        name: str = "MeanReversionStrength",
    ) -> None:
        super().__init__(name)
        self.sma_key = sma_key
        self.min_deviation_pct = min_deviation_pct

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if price has deviated enough from mean."""
        sma = indicators.get(self.sma_key)

        if sma is None or sma <= 0:
            return False

        # Calculate deviation percentage
        deviation_pct = abs(current.close - sma) / sma
        return deviation_pct >= self.min_deviation_pct
