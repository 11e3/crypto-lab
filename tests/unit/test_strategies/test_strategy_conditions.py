"""
Tests for strategy conditions modules - momentum, mean_reversion, and pair_trading.
"""

from datetime import date

import pandas as pd
import pytest

from src.strategies.base import OHLCV


@pytest.fixture
def sample_ohlcv() -> OHLCV:
    """Create sample OHLCV data."""
    return OHLCV(
        date=date(2024, 1, 1), open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0
    )


@pytest.fixture
def sample_history() -> pd.DataFrame:
    """Create sample price history."""
    dates = pd.date_range("2024-01-01", periods=50)
    return pd.DataFrame(
        {
            "open": [100 + i * 0.5 for i in range(50)],
            "high": [105 + i * 0.5 for i in range(50)],
            "low": [95 + i * 0.5 for i in range(50)],
            "close": [102 + i * 0.5 for i in range(50)],
            "volume": [1000 + i * 10 for i in range(50)],
        },
        index=dates,
    )


class TestMomentumConditions:
    """Test momentum strategy conditions."""

    def test_price_above_sma_entry(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test price above SMA entry condition."""
        from src.strategies.momentum.conditions import PriceAboveSMACondition

        condition = PriceAboveSMACondition()

        # Price above SMA - should enter
        indicators = {"sma": 100.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Price below SMA - should not enter
        indicators = {"sma": 105.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_price_below_sma_exit(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test price below SMA exit condition."""
        from src.strategies.momentum.conditions import PriceBelowSMACondition

        condition = PriceBelowSMACondition()

        # Price below SMA - should exit
        indicators = {"sma": 105.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Price above SMA - should not exit
        indicators = {"sma": 100.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_rsi_oversold_entry(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test RSI oversold entry condition."""
        from src.strategies.momentum.conditions import RSIOversoldCondition

        condition = RSIOversoldCondition(oversold_threshold=30.0)

        # RSI oversold - should enter
        indicators = {"rsi": 25.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # RSI neutral - should not enter
        indicators = {"rsi": 50.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_rsi_overbought_exit(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test RSI overbought exit condition."""
        from src.strategies.momentum.conditions import RSIOverboughtCondition

        condition = RSIOverboughtCondition(overbought_threshold=70.0)

        # RSI overbought - should exit
        indicators = {"rsi": 75.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # RSI neutral - should not exit
        indicators = {"rsi": 50.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_macd_bullish(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test MACD bullish condition."""
        from src.strategies.momentum.conditions import MACDBullishCondition

        condition = MACDBullishCondition()

        # MACD above signal - bullish
        indicators = {"macd": 0.5, "macd_signal": 0.2}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # MACD below signal - not bullish
        indicators = {"macd": 0.1, "macd_signal": 0.5}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_macd_bearish(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test MACD bearish condition."""
        from src.strategies.momentum.conditions import MACDBearishCondition

        condition = MACDBearishCondition()

        # MACD below signal - bearish
        indicators = {"macd": 0.1, "macd_signal": 0.5}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # MACD above signal - not bearish
        indicators = {"macd": 0.5, "macd_signal": 0.2}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_momentum_strength(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test momentum strength condition."""
        from src.strategies.momentum.conditions import MomentumStrengthCondition

        # With 10-day lookback, check change from 10 days ago
        # sample_history starts at 100.0, so 10 days ago it was 100 + (10*0.5) = 105
        # Current close is 102, so change is negative (-3%)
        condition = MomentumStrengthCondition(lookback=10, min_change_pct=0.05)
        result = condition.evaluate(sample_ohlcv, sample_history, {})
        assert not result or result is False

        # Let's use lookback of 5 with lower threshold
        # 5 days ago was 102 + (5*0.5) = 104.5
        # Change is (102 - 104.5) / 104.5 = -2.4%
        condition_short = MomentumStrengthCondition(lookback=5, min_change_pct=0.01)
        result_short = condition_short.evaluate(sample_ohlcv, sample_history, {})
        assert not result_short or result_short is False


class TestMeanReversionConditions:
    """Test mean reversion strategy conditions."""

    def test_bollinger_lower_band(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test Bollinger lower band entry condition."""
        from src.strategies.mean_reversion.conditions import BollingerLowerBandCondition

        condition = BollingerLowerBandCondition()

        # Price at lower band - entry
        indicators = {"bb_lower": 102.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Price well above lower band - no entry
        indicators = {"bb_lower": 90.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_bollinger_upper_band(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test Bollinger upper band exit condition."""
        from src.strategies.mean_reversion.conditions import BollingerUpperBandCondition

        condition = BollingerUpperBandCondition()

        # Price touches/above upper band - exit
        indicators = {"bb_upper": 100.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Price below upper band - no exit
        indicators = {"bb_upper": 110.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_price_above_sma_exit(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test price above SMA for mean reversion exit."""
        from src.strategies.mean_reversion.conditions import PriceAboveSMACondition

        condition = PriceAboveSMACondition()

        # Price above SMA - exit
        indicators = {"sma": 100.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Price below SMA - stay in
        indicators = {"sma": 105.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_price_below_sma_entry(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test price below SMA for mean reversion entry."""
        from src.strategies.mean_reversion.conditions import PriceBelowSMACondition

        condition = PriceBelowSMACondition()

        # Price below SMA - entry
        indicators = {"sma": 105.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Price above SMA - no entry
        indicators = {"sma": 100.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_rsi_oversold_entry_mean_reversion(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test RSI oversold for mean reversion entry."""
        from src.strategies.mean_reversion.conditions import RSIOversoldCondition

        condition = RSIOversoldCondition(oversold_threshold=30.0)

        # RSI oversold - entry signal
        indicators = {"rsi": 20.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # RSI normal - no signal
        indicators = {"rsi": 50.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_rsi_overbought_exit_mean_reversion(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test RSI overbought for mean reversion exit."""
        from src.strategies.mean_reversion.conditions import RSIOverboughtCondition

        condition = RSIOverboughtCondition(overbought_threshold=70.0)

        # RSI overbought - exit signal
        indicators = {"rsi": 80.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # RSI normal - no signal
        indicators = {"rsi": 50.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_mean_reversion_strength(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test mean reversion strength condition."""
        from src.strategies.mean_reversion.conditions import MeanReversionStrengthCondition

        condition = MeanReversionStrengthCondition(sma_key="sma", min_deviation_pct=0.02)

        # Price deviates 5% from SMA - strong signal
        indicators = {"sma": 97.14}  # 102.0 is 5% above 97.14
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Price 1% from SMA - weak signal
        indicators = {"sma": 101.0}  # 102.0 is ~1% above 101.0
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False


class TestPairTradingConditions:
    """Test pair trading strategy conditions."""

    def test_spread_zscore_oversold(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test spread Z-score oversold condition."""
        from src.strategies.pair_trading.conditions import SpreadZScoreCondition

        condition = SpreadZScoreCondition(z_score_key="z_score", entry_threshold=2.0)

        # Spread oversold (very negative)
        indicators = {"z_score": -3.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Spread neutral
        indicators = {"z_score": 0.5}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_spread_zscore_overbought(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test spread Z-score overbought condition."""
        from src.strategies.pair_trading.conditions import SpreadZScoreCondition

        condition = SpreadZScoreCondition(z_score_key="z_score", entry_threshold=2.0)

        # Spread overbought (very positive)
        indicators = {"z_score": 3.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Spread neutral
        indicators = {"z_score": 0.5}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_spread_mean_reversion(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test spread mean reversion condition."""
        from src.strategies.pair_trading.conditions import SpreadMeanReversionCondition

        condition = SpreadMeanReversionCondition()

        # Spread above mean
        indicators = {"spread": 2.0, "spread_mean": 1.0}
        result = condition.evaluate(sample_ohlcv, sample_history, indicators)
        assert isinstance(result, bool)

        # Spread below mean
        indicators = {"spread": 0.5, "spread_mean": 1.0}
        result = condition.evaluate(sample_ohlcv, sample_history, indicators)
        assert isinstance(result, bool)

    def test_spread_deviation(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test spread deviation condition."""
        from src.strategies.pair_trading.conditions import SpreadDeviationCondition

        condition = SpreadDeviationCondition(
            spread_key="spread", spread_mean_key="spread_mean", min_deviation_pct=0.02
        )

        # Large deviation (5%)
        indicators = {
            "spread": 3.0,
            "spread_mean": 2.86,  # 5% deviation
        }
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Small deviation (1%)
        indicators = {"spread": 1.01, "spread_mean": 1.0}
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False


class TestConditionsWithMissingIndicators:
    """Test conditions handle missing indicators gracefully."""

    def test_momentum_missing_sma(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test momentum condition with missing SMA."""
        from src.strategies.momentum.conditions import PriceAboveSMACondition

        condition = PriceAboveSMACondition()
        indicators = {}

        # Should return False when indicator missing
        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_momentum_missing_rsi(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test momentum condition with missing RSI."""
        from src.strategies.momentum.conditions import RSIOversoldCondition

        condition = RSIOversoldCondition()
        indicators = {}

        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_momentum_missing_macd(self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame) -> None:
        """Test momentum condition with missing MACD."""
        from src.strategies.momentum.conditions import MACDBullishCondition

        condition = MACDBullishCondition()
        indicators = {}

        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_mean_reversion_missing_bollinger(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test mean reversion condition with missing Bollinger bands."""
        from src.strategies.mean_reversion.conditions import BollingerLowerBandCondition

        condition = BollingerLowerBandCondition()
        indicators = {}

        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False

    def test_pair_trading_missing_zscore(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test pair trading condition with missing Z-score."""
        from src.strategies.pair_trading.conditions import SpreadZScoreCondition

        condition = SpreadZScoreCondition()
        indicators = {}

        assert condition.evaluate(sample_ohlcv, sample_history, indicators) is False


class TestConditionIntegration:
    """Integration tests combining multiple conditions."""

    def test_momentum_entry_signals(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test multiple momentum entry signals together."""
        from src.strategies.momentum.conditions import (
            MACDBullishCondition,
            PriceAboveSMACondition,
            RSIOversoldCondition,
        )

        indicators = {"sma": 100.0, "rsi": 25.0, "macd": 0.5, "macd_signal": 0.2}

        above_sma = PriceAboveSMACondition().evaluate(sample_ohlcv, sample_history, indicators)
        rsi_oversold = RSIOversoldCondition().evaluate(sample_ohlcv, sample_history, indicators)
        macd_bullish = MACDBullishCondition().evaluate(sample_ohlcv, sample_history, indicators)

        # All signals aligned for entry
        assert above_sma is True
        assert rsi_oversold is True
        assert macd_bullish is True

    def test_mean_reversion_entry_signals(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test multiple mean reversion entry signals together."""
        from src.strategies.mean_reversion.conditions import (
            BollingerLowerBandCondition,
            PriceBelowSMACondition,
            RSIOversoldCondition,
        )

        indicators = {"bb_lower": 105.0, "rsi": 20.0, "sma": 105.0}

        lower_band = BollingerLowerBandCondition().evaluate(
            sample_ohlcv, sample_history, indicators
        )
        rsi_oversold = RSIOversoldCondition().evaluate(sample_ohlcv, sample_history, indicators)
        below_sma = PriceBelowSMACondition().evaluate(sample_ohlcv, sample_history, indicators)

        # All signals aligned for entry
        assert lower_band is True
        assert rsi_oversold is True
        assert below_sma is True

    def test_pair_trading_spread_signals(
        self, sample_ohlcv: OHLCV, sample_history: pd.DataFrame
    ) -> None:
        """Test pair trading signals."""
        from src.strategies.pair_trading.conditions import SpreadZScoreCondition

        # Oversold pair
        oversold_condition = SpreadZScoreCondition(z_score_key="z_score", entry_threshold=2.0)
        indicators = {"z_score": -3.0}
        assert oversold_condition.evaluate(sample_ohlcv, sample_history, indicators) is True

        # Overbought pair
        overbought_condition = SpreadZScoreCondition(z_score_key="z_score", entry_threshold=2.0)
        indicators = {"z_score": 3.0}
        assert overbought_condition.evaluate(sample_ohlcv, sample_history, indicators) is True
