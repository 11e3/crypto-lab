"""
Entry and exit conditions for Mean Reversion strategy.

These conditions identify when prices have deviated from their mean
and are likely to revert back.
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition


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


class PriceBelowSMACondition(Condition):
    """
    Entry condition: Price is below SMA (oversold).

    Used in mean reversion to identify buying opportunities.
    """

    def __init__(self, sma_key: str = "sma", name: str = "PriceBelowSMA") -> None:
        """
        Initialize price below SMA condition.

        Args:
            sma_key: Key for SMA value in indicators dict
            name: Condition name
        """
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


class PriceAboveSMACondition(Condition):
    """
    Exit condition: Price rises above SMA.

    Indicates price has reverted back to mean.
    """

    def __init__(self, sma_key: str = "sma", name: str = "PriceAboveSMA") -> None:
        """
        Initialize price above SMA condition.

        Args:
            sma_key: Key for SMA value in indicators dict
            name: Condition name
        """
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


class RSIOversoldCondition(Condition):
    """
    Entry condition: RSI is oversold (below threshold).

    Confirms oversold condition for mean reversion entry.
    """

    def __init__(
        self,
        rsi_key: str = "rsi",
        oversold_threshold: float = 30.0,
        name: str = "RSIOversold",
    ) -> None:
        """
        Initialize RSI oversold condition.

        Args:
            rsi_key: Key for RSI value in indicators dict
            oversold_threshold: RSI threshold for oversold (default 30)
            name: Condition name
        """
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
    """
    Exit condition: RSI is overbought (above threshold).

    Indicates price has reverted and may reverse.
    """

    def __init__(
        self,
        rsi_key: str = "rsi",
        overbought_threshold: float = 70.0,
        name: str = "RSIOverbought",
    ) -> None:
        """
        Initialize RSI overbought condition.

        Args:
            rsi_key: Key for RSI value in indicators dict
            overbought_threshold: RSI threshold for overbought (default 70)
            name: Condition name
        """
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
        """
        Initialize mean reversion strength condition.

        Args:
            sma_key: Key for SMA value in indicators dict
            min_deviation_pct: Minimum deviation percentage from SMA (default 2%)
            name: Condition name
        """
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
