"""
Entry and exit conditions for Mean Reversion strategy.

These conditions identify when prices have deviated from their mean
and are likely to revert back.
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition
from src.strategies.common_conditions import (
    PriceAboveSMACondition,
    PriceBelowSMACondition,
    RSIOverboughtCondition,
    RSIOversoldCondition,
)

# Re-export mean-reversion-specific RSI condition
from src.strategies.mean_reversion.conditions_rsi import (
    MeanReversionStrengthCondition,
)

__all__ = [
    "BollingerLowerBandCondition",
    "BollingerUpperBandCondition",
    "PriceBelowSMACondition",
    "PriceAboveSMACondition",
    "RSIOversoldCondition",
    "RSIOverboughtCondition",
    "MeanReversionStrengthCondition",
]


class BollingerLowerBandCondition(Condition):
    """
    Entry condition: Price touches or falls below lower Bollinger Band.

    Indicates oversold condition where price is likely to revert upward.
    """

    def __init__(
        self,
        lower_band_key: str = "bb_lower",
        name: str = "BollingerLowerBand",
    ) -> None:
        """
        Initialize Bollinger lower band condition.

        Args:
            lower_band_key: Key for lower band value in indicators dict
            name: Condition name
        """
        super().__init__(name)
        self.lower_band_key = lower_band_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if price is at or below lower Bollinger Band."""
        lower_band = indicators.get(self.lower_band_key)

        if lower_band is None:
            return False

        # Price touches or falls below lower band
        return current.low <= lower_band


class BollingerUpperBandCondition(Condition):
    """
    Exit condition: Price touches or rises above upper Bollinger Band.

    Indicates overbought condition where price is likely to revert downward.
    """

    def __init__(
        self,
        upper_band_key: str = "bb_upper",
        name: str = "BollingerUpperBand",
    ) -> None:
        """
        Initialize Bollinger upper band condition.

        Args:
            upper_band_key: Key for upper band value in indicators dict
            name: Condition name
        """
        super().__init__(name)
        self.upper_band_key = upper_band_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if price is at or above upper Bollinger Band."""
        upper_band = indicators.get(self.upper_band_key)

        if upper_band is None:
            return False

        # Price touches or rises above upper band
        return current.high >= upper_band
