"""
Entry and exit conditions for Volatility Breakout strategy.

These conditions can be mixed and matched to create custom VBO variants.
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition


class BreakoutCondition(Condition):
    """
    Core volatility breakout condition.

    Triggers when price breaks above target price:
    Target = Open + (Prev_High - Prev_Low) * K

    Entry occurs when High >= Target
    """

    def __init__(self, name: str = "Breakout") -> None:
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


class PriceAboveSMACondition(Condition):
    """
    Condition: Current close price must be above SMA.

    Used to confirm trend direction before entry.
    """

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
    """
    Exit condition: Close price falls below SMA.

    Standard exit signal for VBO strategy.
    """

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


class WhipsawExitCondition(Condition):
    """
    Exit condition for same-day whipsaw detection.

    Triggers when price breaks out but then falls below SMA
    on the same candle, indicating a false breakout.
    """

    def __init__(
        self,
        sma_key: str = "sma",
        name: str = "WhipsawExit",
    ) -> None:
        super().__init__(name)
        self.sma_key = sma_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check for whipsaw condition."""
        target = indicators.get("target")
        sma = indicators.get(self.sma_key)

        if target is None or sma is None:
            return False

        # Whipsaw: broke out (high >= target) but closed below SMA
        breakout_occurred = current.high >= target
        closed_below_sma = current.close < sma

        return breakout_occurred and closed_below_sma


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


# Market conditions (formerly filters)
# These conditions can be used as entry conditions to filter trading opportunities


class TrendCondition(Condition):
    """
    Trend condition using moving average.

    Only allows trading when target price is above the trend SMA,
    indicating an uptrend environment.
    """

    def __init__(
        self,
        sma_key: str = "sma_trend",
        name: str = "TrendCondition",
    ) -> None:
        """
        Initialize trend condition.

        Args:
            sma_key: Key for trend SMA in indicators
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
        """Check if target is above trend SMA."""
        target = indicators.get("target")
        trend_sma = indicators.get(self.sma_key)

        if target is None or trend_sma is None:
            return False

        return target > trend_sma


class NoiseCondition(Condition):
    """
    Dynamic noise condition.

    Only allows trading when short-term noise is below long-term
    baseline, indicating a relatively calm market suitable for
    breakout strategies.
    """

    def __init__(
        self,
        short_noise_key: str = "short_noise",
        long_noise_key: str = "long_noise",
        name: str = "NoiseCondition",
    ) -> None:
        """
        Initialize noise condition.

        Args:
            short_noise_key: Key for short-term noise ratio
            long_noise_key: Key for long-term noise baseline
            name: Condition name
        """
        super().__init__(name)
        self.short_noise_key = short_noise_key
        self.long_noise_key = long_noise_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if current noise is below baseline."""
        short_noise = indicators.get(self.short_noise_key)
        long_noise = indicators.get(self.long_noise_key)

        if short_noise is None or long_noise is None:
            return False

        # Market is "calm" when short-term noise < long-term average
        return short_noise < long_noise


class NoiseThresholdCondition(Condition):
    """
    Absolute noise threshold condition.

    Filters out periods where noise ratio exceeds a fixed threshold.
    """

    def __init__(
        self,
        noise_key: str = "short_noise",
        max_noise: float = 0.7,
        name: str = "NoiseThresholdCondition",
    ) -> None:
        """
        Initialize noise threshold condition.

        Args:
            noise_key: Key for noise ratio
            max_noise: Maximum allowed noise ratio (0-1)
            name: Condition name
        """
        super().__init__(name)
        self.noise_key = noise_key
        self.max_noise = max_noise

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if noise is below threshold."""
        noise = indicators.get(self.noise_key)

        if noise is None:
            return False

        return noise <= self.max_noise


class VolatilityRangeCondition(Condition):
    """
    Volatility range condition.

    Only allows trading when volatility is within acceptable range.
    """

    def __init__(
        self,
        min_volatility_pct: float = 0.005,
        max_volatility_pct: float = 0.15,
        name: str = "VolatilityRangeCondition",
    ) -> None:
        """
        Initialize volatility range condition.

        Args:
            min_volatility_pct: Minimum range as % of price
            max_volatility_pct: Maximum range as % of price
            name: Condition name
        """
        super().__init__(name)
        self.min_volatility_pct = min_volatility_pct
        self.max_volatility_pct = max_volatility_pct

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if volatility is within range."""
        prev_range = indicators.get("prev_range")

        if prev_range is None or current.open <= 0:
            return False

        volatility_pct = prev_range / current.open

        return self.min_volatility_pct <= volatility_pct <= self.max_volatility_pct


class VolumeCondition(Condition):
    """
    Volume condition.

    Only allows trading when volume is above minimum threshold
    relative to recent average.
    """

    def __init__(
        self,
        min_volume_ratio: float = 0.5,
        lookback: int = 20,
        name: str = "VolumeCondition",
    ) -> None:
        """
        Initialize volume condition.

        Args:
            min_volume_ratio: Minimum volume as ratio of average
            lookback: Lookback period for average calculation
            name: Condition name
        """
        super().__init__(name)
        self.min_volume_ratio = min_volume_ratio
        self.lookback = lookback

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if volume is above threshold."""
        if len(history) < self.lookback:
            return True  # Not enough data, allow trading

        if "volume" not in history.columns:
            return True  # No volume data available

        avg_volume = history["volume"].tail(self.lookback).mean()

        if avg_volume <= 0:
            return True

        result: bool = bool(current.volume >= avg_volume * self.min_volume_ratio)
        return result


class DayOfWeekCondition(Condition):
    """
    Day of week condition.

    Allows trading only on specified days of the week.
    """

    def __init__(
        self,
        allowed_days: list[int] | None = None,
        name: str = "DayOfWeekCondition",
    ) -> None:
        """
        Initialize day of week condition.

        Args:
            allowed_days: List of allowed weekdays (0=Monday, 6=Sunday)
                         Default is Monday-Friday (0-4)
            name: Condition name
        """
        super().__init__(name)
        self.allowed_days = allowed_days or [0, 1, 2, 3, 4]

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if current day is allowed."""
        return current.date.weekday() in self.allowed_days


class MarketRegimeCondition(Condition):
    """
    Market regime condition based on price position relative to long-term MA.

    Only trades in bullish regimes (price above long-term MA).
    """

    def __init__(
        self,
        sma_key: str = "sma_trend",
        name: str = "MarketRegimeCondition",
    ) -> None:
        """
        Initialize market regime condition.

        Args:
            sma_key: Key for long-term SMA
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
        """Check if price is above long-term SMA."""
        sma = indicators.get(self.sma_key)

        if sma is None:
            return False

        return current.close > sma


# Backward compatibility aliases (deprecated, use Condition classes above)
# These aliases are provided for backward compatibility but will be removed in a future version.
# Please update your code to use the Condition classes directly.
import warnings  # noqa: E402
from typing import Any, TypeVar, cast  # noqa: E402

T = TypeVar("T")


def _create_deprecated_alias(condition_class: type[T], filter_name: str) -> type[T]:
    """Create a deprecated alias with warning."""

    class DeprecatedFilter(condition_class):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                f"{filter_name} is deprecated. Use {condition_class.__name__} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            super().__init__(*args, **kwargs)

    DeprecatedFilter.__name__ = filter_name
    DeprecatedFilter.__qualname__ = filter_name
    return cast(type[T], DeprecatedFilter)


# Create deprecated aliases with warnings
TrendFilter = _create_deprecated_alias(TrendCondition, "TrendFilter")
NoiseFilter = _create_deprecated_alias(NoiseCondition, "NoiseFilter")
NoiseThresholdFilter = _create_deprecated_alias(NoiseThresholdCondition, "NoiseThresholdFilter")
VolatilityFilter = _create_deprecated_alias(VolatilityRangeCondition, "VolatilityFilter")
VolumeFilter = _create_deprecated_alias(VolumeCondition, "VolumeFilter")
DayOfWeekFilter = _create_deprecated_alias(DayOfWeekCondition, "DayOfWeekFilter")
MarketRegimeFilter = _create_deprecated_alias(MarketRegimeCondition, "MarketRegimeFilter")
