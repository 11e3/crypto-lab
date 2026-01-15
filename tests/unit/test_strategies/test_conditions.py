"""
Unit tests for VBO strategy conditions.
"""

from datetime import date

import pandas as pd

from src.strategies.base import OHLCV
from src.strategies.volatility_breakout.conditions import (
    BreakoutCondition,
    ConsecutiveUpCondition,
    DayOfWeekCondition,
    MarketRegimeCondition,
    NoiseCondition,
    NoiseThresholdCondition,
    PriceAboveSMACondition,
    PriceBelowSMACondition,
    SMABreakoutCondition,
    TrendAlignmentCondition,
    TrendCondition,
    VolatilityRangeCondition,
    VolatilityThresholdCondition,
    VolumeCondition,
    WhipsawExitCondition,
)


class TestBreakoutCondition:
    """Tests for BreakoutCondition."""

    def test_breakout_condition_initialization(self) -> None:
        """Test BreakoutCondition initialization."""
        condition = BreakoutCondition()
        assert condition.name == "Breakout"

    def test_breakout_condition_with_target_above_high(self) -> None:
        """Test BreakoutCondition when high >= target."""
        condition = BreakoutCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"target": 108.0}

        assert condition.evaluate(current, history, indicators) is True

    def test_breakout_condition_with_target_below_high(self) -> None:
        """Test BreakoutCondition when high < target."""
        condition = BreakoutCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 105.0, 95.0, 103.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"target": 108.0}

        assert condition.evaluate(current, history, indicators) is False

    def test_breakout_condition_with_target_equal_high(self) -> None:
        """Test BreakoutCondition when high == target."""
        condition = BreakoutCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"target": 110.0}

        assert condition.evaluate(current, history, indicators) is True

    def test_breakout_condition_without_target(self) -> None:
        """Test BreakoutCondition when target is missing."""
        condition = BreakoutCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestSMABreakoutCondition:
    """Tests for SMABreakoutCondition."""

    def test_sma_breakout_condition_initialization(self) -> None:
        """Test SMABreakoutCondition initialization."""
        condition = SMABreakoutCondition()
        assert condition.name == "SMABreakout"
        assert condition.sma_key == "sma"

    def test_sma_breakout_condition_custom_key(self) -> None:
        """Test SMABreakoutCondition with custom SMA key."""
        condition = SMABreakoutCondition(sma_key="custom_sma")
        assert condition.sma_key == "custom_sma"

    def test_sma_breakout_condition_target_above_sma(self) -> None:
        """Test SMABreakoutCondition when target > sma."""
        condition = SMABreakoutCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"target": 110.0, "sma": 100.0}

        assert condition.evaluate(current, history, indicators) is True

    def test_sma_breakout_condition_target_below_sma(self) -> None:
        """Test SMABreakoutCondition when target <= sma."""
        condition = SMABreakoutCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"target": 95.0, "sma": 100.0}

        assert condition.evaluate(current, history, indicators) is False

    def test_sma_breakout_condition_missing_indicator(self) -> None:
        """Test SMABreakoutCondition when indicators are missing."""
        condition = SMABreakoutCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestPriceAboveSMACondition:
    """Tests for PriceAboveSMACondition."""

    def test_price_above_sma_condition_initialization(self) -> None:
        """Test PriceAboveSMACondition initialization."""
        condition = PriceAboveSMACondition()
        assert condition.name == "PriceAboveSMA"
        assert condition.sma_key == "sma"

    def test_price_above_sma_condition_close_above_sma(self) -> None:
        """Test PriceAboveSMACondition when close > sma."""
        condition = PriceAboveSMACondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"sma": 100.0}

        assert condition.evaluate(current, history, indicators) is True

    def test_price_above_sma_condition_close_below_sma(self) -> None:
        """Test PriceAboveSMACondition when close <= sma."""
        condition = PriceAboveSMACondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 95.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"sma": 100.0}

        assert condition.evaluate(current, history, indicators) is False

    def test_price_above_sma_condition_missing_sma(self) -> None:
        """Test PriceAboveSMACondition when sma is missing."""
        condition = PriceAboveSMACondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestPriceBelowSMACondition:
    """Tests for PriceBelowSMACondition."""

    def test_price_below_sma_condition_initialization(self) -> None:
        """Test PriceBelowSMACondition initialization."""
        condition = PriceBelowSMACondition()
        assert condition.name == "PriceBelowSMA"
        assert condition.sma_key == "sma"

    def test_price_below_sma_condition_close_below_sma(self) -> None:
        """Test PriceBelowSMACondition when close < sma."""
        condition = PriceBelowSMACondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 95.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"sma": 100.0}

        assert condition.evaluate(current, history, indicators) is True

    def test_price_below_sma_condition_close_above_sma(self) -> None:
        """Test PriceBelowSMACondition when close >= sma."""
        condition = PriceBelowSMACondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"sma": 100.0}

        assert condition.evaluate(current, history, indicators) is False

    def test_price_below_sma_condition_missing_sma(self) -> None:
        """Test PriceBelowSMACondition when sma is missing."""
        condition = PriceBelowSMACondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 95.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestWhipsawExitCondition:
    """Tests for WhipsawExitCondition."""

    def test_whipsaw_exit_condition_initialization(self) -> None:
        """Test WhipsawExitCondition initialization."""
        condition = WhipsawExitCondition()
        assert condition.name == "WhipsawExit"

    def test_whipsaw_exit_condition_breakout_and_close_below_sma(self) -> None:
        """Test WhipsawExitCondition when breakout occurred and closed below SMA."""
        condition = WhipsawExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 95.0, 1000.0)  # high=110, close=95
        history = pd.DataFrame()
        indicators = {
            "target": 108.0,
            "sma": 100.0,
        }  # high >= target (110>=108), close < sma (95<100)

        assert condition.evaluate(current, history, indicators) is True

    def test_whipsaw_exit_condition_breakout_but_close_above_sma(self) -> None:
        """Test WhipsawExitCondition when breakout occurred but close is above SMA."""
        condition = WhipsawExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)  # high=110, close=105
        history = pd.DataFrame()
        indicators = {
            "target": 108.0,
            "sma": 100.0,
        }  # high >= target (110>=108), close >= sma (105>=100)

        assert condition.evaluate(current, history, indicators) is False

    def test_whipsaw_exit_condition_no_breakout(self) -> None:
        """Test WhipsawExitCondition when no breakout occurred."""
        condition = WhipsawExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 105.0, 95.0, 95.0, 1000.0)  # high=105
        history = pd.DataFrame()
        indicators = {"target": 108.0, "sma": 100.0}  # high < target (105<108)

        assert condition.evaluate(current, history, indicators) is False

    def test_whipsaw_exit_condition_missing_indicator(self) -> None:
        """Test WhipsawExitCondition when indicators are missing."""
        condition = WhipsawExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 95.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False

    def test_whipsaw_exit_condition_empty_history(self) -> None:
        """Test WhipsawExitCondition with empty history."""
        condition = WhipsawExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 95.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestTrendAlignmentCondition:
    """Tests for TrendAlignmentCondition."""

    def test_trend_alignment_condition_initialization(self) -> None:
        """Test TrendAlignmentCondition initialization."""
        condition = TrendAlignmentCondition()
        assert condition.name == "TrendAlignment"

    def test_trend_alignment_condition_aligned(self) -> None:
        """Test TrendAlignmentCondition when target is above trend SMA."""
        condition = TrendAlignmentCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"target": 110.0, "sma_trend": 100.0}

        assert condition.evaluate(current, history, indicators) is True

    def test_trend_alignment_condition_not_aligned(self) -> None:
        """Test TrendAlignmentCondition when target is below trend SMA."""
        condition = TrendAlignmentCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 95.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"target": 95.0, "sma_trend": 100.0}

        assert condition.evaluate(current, history, indicators) is False

    def test_trend_alignment_condition_missing_indicator(self) -> None:
        """Test TrendAlignmentCondition when indicators are missing."""
        condition = TrendAlignmentCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestVolatilityThresholdCondition:
    """Tests for VolatilityThresholdCondition."""

    def test_volatility_threshold_condition_initialization(self) -> None:
        """Test VolatilityThresholdCondition initialization."""
        condition = VolatilityThresholdCondition(min_range_pct=0.01)
        assert condition.min_range_pct == 0.01

    def test_volatility_threshold_condition_above_threshold(self) -> None:
        """Test VolatilityThresholdCondition when volatility is above threshold."""
        condition = VolatilityThresholdCondition(min_range_pct=0.01)
        current = OHLCV(date(2024, 1, 1), 100.0, 105.0, 95.0, 102.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"prev_range": 2.0}  # 2.0 / 100.0 = 0.02 >= 0.01

        assert condition.evaluate(current, history, indicators) is True

    def test_volatility_threshold_condition_below_threshold(self) -> None:
        """Test VolatilityThresholdCondition when volatility is below threshold."""
        condition = VolatilityThresholdCondition(min_range_pct=0.01)
        current = OHLCV(date(2024, 1, 1), 100.0, 100.5, 99.5, 100.2, 1000.0)
        history = pd.DataFrame()
        indicators = {"prev_range": 0.5}  # 0.5 / 100.0 = 0.005 < 0.01

        assert condition.evaluate(current, history, indicators) is False

    def test_volatility_threshold_condition_missing_indicator(self) -> None:
        """Test VolatilityThresholdCondition when prev_range is missing."""
        condition = VolatilityThresholdCondition(min_range_pct=0.01)
        current = OHLCV(date(2024, 1, 1), 100.0, 105.0, 95.0, 102.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestConsecutiveUpCondition:
    """Tests for ConsecutiveUpCondition."""

    def test_consecutive_up_condition_initialization(self) -> None:
        """Test ConsecutiveUpCondition initialization."""
        condition = ConsecutiveUpCondition(days=3)
        assert condition.days == 3

    def test_consecutive_up_condition_meets_threshold(self) -> None:
        """Test ConsecutiveUpCondition when consecutive ups meet threshold."""
        condition = ConsecutiveUpCondition(days=3)
        current = OHLCV(date(2024, 1, 5), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame(
            {
                "close": [95.0, 98.0, 100.0, 102.0, 105.0],
            }
        )
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is True

    def test_consecutive_up_condition_below_threshold(self) -> None:
        """Test ConsecutiveUpCondition when consecutive ups are below threshold."""
        condition = ConsecutiveUpCondition(days=3)
        current = OHLCV(date(2024, 1, 4), 100.0, 110.0, 95.0, 102.0, 1000.0)
        history = pd.DataFrame(
            {
                "close": [100.0, 98.0, 100.0, 102.0],
            }
        )
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False

    def test_consecutive_up_condition_insufficient_data(self) -> None:
        """Test ConsecutiveUpCondition with insufficient data."""
        condition = ConsecutiveUpCondition(days=3)
        current = OHLCV(date(2024, 1, 2), 100.0, 110.0, 95.0, 102.0, 1000.0)
        history = pd.DataFrame({"close": [100.0, 102.0]})
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestTrendCondition:
    """Tests for TrendCondition."""

    def test_trend_condition_initialization(self) -> None:
        """Test TrendCondition initialization."""
        condition = TrendCondition()
        assert condition.name == "TrendCondition" or condition.name == "TrendFilter"
        assert condition.sma_key == "sma_trend"

    def test_trend_condition_target_above_trend_sma(self) -> None:
        """Test TrendCondition when target > trend SMA."""
        condition = TrendCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"target": 110.0, "sma_trend": 100.0}

        assert condition.evaluate(current, history, indicators) is True

    def test_trend_condition_target_below_trend_sma(self) -> None:
        """Test TrendCondition when target <= trend SMA."""
        condition = TrendCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"target": 95.0, "sma_trend": 100.0}

        assert condition.evaluate(current, history, indicators) is False

    def test_trend_condition_missing_indicator(self) -> None:
        """Test TrendCondition when indicators are missing."""
        condition = TrendCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestNoiseCondition:
    """Tests for NoiseCondition."""

    def test_noise_condition_initialization(self) -> None:
        """Test NoiseCondition initialization."""
        condition = NoiseCondition()
        assert condition.name == "NoiseCondition" or condition.name == "NoiseFilter"

    def test_noise_condition_short_below_long(self) -> None:
        """Test NoiseCondition when short_noise < long_noise."""
        condition = NoiseCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"short_noise": 0.5, "long_noise": 0.6}

        assert condition.evaluate(current, history, indicators) is True

    def test_noise_condition_short_above_long(self) -> None:
        """Test NoiseCondition when short_noise >= long_noise."""
        condition = NoiseCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"short_noise": 0.7, "long_noise": 0.6}

        assert condition.evaluate(current, history, indicators) is False

    def test_noise_condition_missing_indicator(self) -> None:
        """Test NoiseCondition when indicators are missing."""
        condition = NoiseCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestNoiseThresholdCondition:
    """Tests for NoiseThresholdCondition."""

    def test_noise_threshold_condition_initialization(self) -> None:
        """Test NoiseThresholdCondition initialization."""
        condition = NoiseThresholdCondition(max_noise=0.7)
        assert condition.max_noise == 0.7

    def test_noise_threshold_condition_below_threshold(self) -> None:
        """Test NoiseThresholdCondition when short_noise <= max_noise."""
        condition = NoiseThresholdCondition(max_noise=0.7)
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"short_noise": 0.5}

        assert condition.evaluate(current, history, indicators) is True

    def test_noise_threshold_condition_above_threshold(self) -> None:
        """Test NoiseThresholdCondition when short_noise > max_noise."""
        condition = NoiseThresholdCondition(max_noise=0.7)
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"short_noise": 0.8}

        assert condition.evaluate(current, history, indicators) is False

    def test_noise_threshold_condition_missing_indicator(self) -> None:
        """Test NoiseThresholdCondition when short_noise is missing."""
        condition = NoiseThresholdCondition(max_noise=0.7)
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestVolatilityRangeCondition:
    """Tests for VolatilityRangeCondition."""

    def test_volatility_range_condition_initialization(self) -> None:
        """Test VolatilityRangeCondition initialization."""
        condition = VolatilityRangeCondition(min_volatility_pct=0.005, max_volatility_pct=0.15)
        assert condition.min_volatility_pct == 0.005
        assert condition.max_volatility_pct == 0.15

    def test_volatility_range_condition_within_range(self) -> None:
        """Test VolatilityRangeCondition when volatility is within range."""
        condition = VolatilityRangeCondition(min_volatility_pct=0.005, max_volatility_pct=0.15)
        current = OHLCV(date(2024, 1, 2), 100.0, 108.0, 92.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"prev_range": 10.0}  # 10.0 / 100.0 = 0.10, within [0.005, 0.15]

        assert condition.evaluate(current, history, indicators) is True

    def test_volatility_range_condition_below_min(self) -> None:
        """Test VolatilityRangeCondition when volatility is below min."""
        condition = VolatilityRangeCondition(min_volatility_pct=0.005, max_volatility_pct=0.15)
        current = OHLCV(date(2024, 1, 2), 100.0, 100.5, 99.5, 100.2, 1000.0)
        history = pd.DataFrame()
        indicators = {"prev_range": 0.3}  # 0.3 / 100.0 = 0.003 < 0.005

        assert condition.evaluate(current, history, indicators) is False

    def test_volatility_range_condition_above_max(self) -> None:
        """Test VolatilityRangeCondition when volatility is above max."""
        condition = VolatilityRangeCondition(min_volatility_pct=0.005, max_volatility_pct=0.15)
        current = OHLCV(date(2024, 1, 2), 100.0, 120.0, 80.0, 110.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"prev_range": 20.0}  # 20.0 / 100.0 = 0.20 > 0.15

        assert condition.evaluate(current, history, indicators) is False

    def test_volatility_range_condition_insufficient_data(self) -> None:
        """Test VolatilityRangeCondition with insufficient data."""
        condition = VolatilityRangeCondition(min_volatility_pct=0.005, max_volatility_pct=0.15)
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestVolumeCondition:
    """Tests for VolumeCondition."""

    def test_volume_condition_initialization(self) -> None:
        """Test VolumeCondition initialization."""
        condition = VolumeCondition(min_volume_ratio=1.5)
        assert condition.min_volume_ratio == 1.5

    def test_volume_condition_above_threshold(self) -> None:
        """Test VolumeCondition when volume is above threshold."""
        condition = VolumeCondition(min_volume_ratio=1.5)
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 2000.0)
        history = pd.DataFrame({"volume": [1000.0, 1200.0, 1500.0]})
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is True

    def test_volume_condition_below_threshold(self) -> None:
        """Test VolumeCondition when volume is below threshold."""
        condition = VolumeCondition(min_volume_ratio=1.5, lookback=3)
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1200.0)
        # avg_volume = (1000 + 1200 + 1500) / 3 = 1233.33
        # current.volume = 1200 < 1233.33 * 1.5 = 1850
        history = pd.DataFrame({"volume": [1000.0, 1200.0, 1500.0]})
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False

    def test_volume_condition_insufficient_data(self) -> None:
        """Test VolumeCondition with insufficient data (returns True)."""
        condition = VolumeCondition(min_volume_ratio=1.5, lookback=20)
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame({"volume": [1000.0, 1200.0]})  # Less than lookback=20
        indicators: dict[str, float] = {}

        assert (
            condition.evaluate(current, history, indicators) is True
        )  # Returns True when insufficient data

    def test_volume_condition_no_volume_column(self) -> None:
        """Test VolumeCondition when volume column is missing (line 480)."""
        condition = VolumeCondition(min_volume_ratio=1.5, lookback=3)
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame({"close": [100.0, 105.0, 110.0]})  # No volume column
        indicators: dict[str, float] = {}

        assert (
            condition.evaluate(current, history, indicators) is True
        )  # Returns True when volume column missing

    def test_volume_condition_zero_avg_volume(self) -> None:
        """Test VolumeCondition when average volume is zero or negative (line 485)."""
        condition = VolumeCondition(min_volume_ratio=1.5, lookback=3)
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame({"volume": [0.0, 0.0, 0.0]})  # All zeros, avg_volume = 0
        indicators: dict[str, float] = {}

        assert (
            condition.evaluate(current, history, indicators) is True
        )  # Returns True when avg_volume <= 0


class TestDayOfWeekCondition:
    """Tests for DayOfWeekCondition."""

    def test_day_of_week_condition_initialization(self) -> None:
        """Test DayOfWeekCondition initialization."""
        condition = DayOfWeekCondition(allowed_days=[0, 1, 2, 3, 4])  # Monday-Friday
        assert condition.allowed_days == [0, 1, 2, 3, 4]

    def test_day_of_week_condition_monday(self) -> None:
        """Test DayOfWeekCondition on Monday."""
        condition = DayOfWeekCondition(allowed_days=[0])  # Monday
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)  # Monday
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is True

    def test_day_of_week_condition_sunday(self) -> None:
        """Test DayOfWeekCondition on Sunday (not allowed)."""
        condition = DayOfWeekCondition(allowed_days=[0, 1, 2, 3, 4])  # Monday-Friday
        current = OHLCV(date(2024, 1, 7), 100.0, 110.0, 95.0, 105.0, 1000.0)  # Sunday
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False


class TestMarketRegimeCondition:
    """Tests for MarketRegimeCondition."""

    def test_market_regime_condition_initialization(self) -> None:
        """Test MarketRegimeCondition initialization."""
        condition = MarketRegimeCondition(sma_key="sma_trend")
        assert condition.sma_key == "sma_trend"

    def test_market_regime_condition_price_above_sma(self) -> None:
        """Test MarketRegimeCondition when price is above SMA (bullish)."""
        condition = MarketRegimeCondition(sma_key="sma_trend")
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"sma_trend": 100.0}  # close=105 > sma_trend=100

        assert condition.evaluate(current, history, indicators) is True

    def test_market_regime_condition_price_below_sma(self) -> None:
        """Test MarketRegimeCondition when price is below SMA (bearish)."""
        condition = MarketRegimeCondition(sma_key="sma_trend")
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 95.0, 1000.0)
        history = pd.DataFrame()
        indicators = {"sma_trend": 100.0}  # close=95 < sma_trend=100

        assert condition.evaluate(current, history, indicators) is False

    def test_market_regime_condition_missing_indicator(self) -> None:
        """Test MarketRegimeCondition when SMA is missing."""
        condition = MarketRegimeCondition(sma_key="sma_trend")
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, history, indicators) is False
