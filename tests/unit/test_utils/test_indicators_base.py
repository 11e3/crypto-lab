"""Tests for src/utils/indicators_base.py - base indicator functions."""

import numpy as np
import pandas as pd
import pytest

from src.utils.indicators_base import atr, ema, noise_ratio, sma


class TestSMA:
    """Tests for sma function."""

    def test_basic_sma(self) -> None:
        """Test basic SMA calculation."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(series, period=3)

        # SMA of [1,2,3] = 2, [2,3,4] = 3, [3,4,5] = 4
        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
        pd.testing.assert_series_equal(result, expected)

    def test_sma_exclude_current(self) -> None:
        """Test SMA with exclude_current=True."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = sma(series, period=3, exclude_current=True)

        # When exclude_current, we shift by 1 before rolling
        # So SMA at index 4 is mean of [2,3,4] = 3
        assert result.iloc[4] == pytest.approx(3.0)

    def test_sma_period_equals_length(self) -> None:
        """Test SMA when period equals series length."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = sma(series, period=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)


class TestEMA:
    """Tests for ema function."""

    def test_basic_ema(self) -> None:
        """Test basic EMA calculation."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(series, period=3)

        # EMA should be calculated
        assert len(result) == 5
        assert not result.isna().all()

    def test_ema_smoothing(self) -> None:
        """Test EMA smoothing property."""
        series = pd.Series([1.0, 5.0, 1.0, 5.0, 1.0])
        result = ema(series, period=3)

        # EMA should smooth out the oscillations
        assert result.std() < series.std()


class TestATR:
    """Tests for atr function."""

    def test_basic_atr(self) -> None:
        """Test basic ATR calculation."""
        high = pd.Series([110.0, 115.0, 120.0, 118.0, 122.0])
        low = pd.Series([100.0, 105.0, 110.0, 108.0, 112.0])
        close = pd.Series([105.0, 110.0, 115.0, 110.0, 118.0])

        result = atr(high, low, close, period=3)

        assert len(result) == 5
        # First few values should be NaN due to period
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # ATR should be positive
        assert result.iloc[3] > 0

    def test_atr_continuous_movement(self) -> None:
        """Test ATR with continuous price movement."""
        # Continuous price movement with gaps between close and next open
        high = pd.Series([102.0, 104.0, 106.0, 108.0, 110.0])
        low = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0])
        close = pd.Series([101.0, 103.0, 105.0, 107.0, 109.0])

        result = atr(high, low, close, period=3)

        # ATR should be positive and reasonable
        assert result.iloc[4] > 0
        assert result.iloc[4] < 10


class TestNoiseRatio:
    """Tests for noise_ratio function."""

    def test_basic_noise_ratio(self) -> None:
        """Test basic noise ratio calculation."""
        open_ = pd.Series([100.0, 105.0, 110.0])
        high = pd.Series([110.0, 115.0, 120.0])
        low = pd.Series([95.0, 100.0, 105.0])
        close = pd.Series([108.0, 112.0, 118.0])

        result = noise_ratio(open_, high, low, close)

        assert len(result) == 3
        # Noise ratio should be between 0 and 1
        assert all((result >= 0) | result.isna())
        assert all((result <= 1) | result.isna())

    def test_full_body_candle(self) -> None:
        """Test noise ratio for full body candle (open=low, close=high)."""
        open_ = pd.Series([100.0])
        high = pd.Series([110.0])
        low = pd.Series([100.0])
        close = pd.Series([110.0])

        result = noise_ratio(open_, high, low, close)

        # Full body = move/range = 10/10 = 1
        assert result.iloc[0] == pytest.approx(1.0)

    def test_doji_candle(self) -> None:
        """Test noise ratio for doji candle (open=close)."""
        open_ = pd.Series([105.0])
        high = pd.Series([110.0])
        low = pd.Series([100.0])
        close = pd.Series([105.0])

        result = noise_ratio(open_, high, low, close)

        # Doji = move/range = 0/10 = 0
        assert result.iloc[0] == pytest.approx(0.0)

    def test_zero_range_protection(self) -> None:
        """Test protection against division by zero."""
        open_ = pd.Series([100.0])
        high = pd.Series([100.0])
        low = pd.Series([100.0])
        close = pd.Series([100.0])

        result = noise_ratio(open_, high, low, close)

        # Should return NaN instead of error
        assert np.isnan(result.iloc[0])
