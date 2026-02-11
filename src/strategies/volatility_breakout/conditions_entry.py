"""
Entry conditions for Volatility Breakout strategy.

These conditions determine when to enter a position.
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition
from src.strategies.common_conditions import PriceAboveSMACondition  # noqa: F401


class BreakoutCondition(Condition):
    """
    Core volatility breakout condition.

    Triggers when price breaks above target price:
    Target = Open + (Prev_High - Prev_Low) * K

    Entry occurs when High >= Target
    """

    def __init__(self, name: str = "Breakout") -> None:
        """Initialize breakout condition."""
        super().__init__(name)

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if price has broken out above target."""
        target = indicators.get("target")

        if target is None:
            return False

        # Breakout: high price reaches or exceeds target
        return current.high >= target


class SMABreakoutCondition(Condition):
    """
    Condition: Target price must be above SMA.

    This ensures we only enter when the breakout target
    is above the moving average, filtering weak signals.
    """

    def __init__(self, sma_key: str = "sma", name: str = "SMABreakout") -> None:
        """
        Initialize SMA breakout condition.

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
        """Check if target is above SMA."""
        target = indicators.get("target")
        sma = indicators.get(self.sma_key)

        if target is None or sma is None:
            return False

        return target > sma


class TrendAlignmentCondition(Condition):
    """
    Condition: Target must align with longer-term trend.

    Target price should be above the trend SMA for uptrend confirmation.
    """

    def __init__(
        self,
        trend_sma_key: str = "sma_trend",
        name: str = "TrendAlignment",
    ) -> None:
        """Initialize trend alignment condition."""
        super().__init__(name)
        self.trend_sma_key = trend_sma_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if target is above trend SMA."""
        target = indicators.get("target")
        trend_sma = indicators.get(self.trend_sma_key)

        if target is None or trend_sma is None:
            return False

        return target > trend_sma


class VolatilityThresholdCondition(Condition):
    """
    Condition: Minimum volatility threshold.

    Filters out low-volatility periods where breakouts
    are less meaningful.
    """

    def __init__(
        self,
        min_range_pct: float = 0.01,
        name: str = "VolatilityThreshold",
    ) -> None:
        """
        Initialize volatility threshold condition.

        Args:
            min_range_pct: Minimum range as percentage of price (default 1%)
            name: Condition name
        """
        super().__init__(name)
        self.min_range_pct = min_range_pct

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if volatility is above threshold."""
        prev_range = indicators.get("prev_range")

        if prev_range is None:
            return False

        # Calculate range as percentage of open price
        range_pct = prev_range / current.open if current.open > 0 else 0

        return range_pct >= self.min_range_pct


class ConsecutiveUpCondition(Condition):
    """
    Condition: Price must have consecutive up days.

    Filters for momentum confirmation.
    """

    def __init__(self, days: int = 2, name: str = "ConsecutiveUp") -> None:
        """
        Initialize consecutive up condition.

        Args:
            days: Number of consecutive up days required
            name: Condition name
        """
        super().__init__(name)
        self.days = days

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check for consecutive up days."""
        if len(history) < self.days + 1:
            return False

        recent = history.tail(self.days + 1)
        closes = recent["close"].values

        # Check if each day closed higher than previous
        return all(closes[i] > closes[i - 1] for i in range(1, len(closes)))
