"""Tests for VBO indicators covering uncovered functions.

Targets: calculate_volatility_regime, calculate_adaptive_noise,
add_improved_indicators (from indicators_vbo.py) and
calculate_noise_ratio, calculate_adaptive_k_value (from indicators_vbo_adaptive.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.indicators_vbo import (
    _atr_local,
    _noise_ratio_local,
    _sma_local,
    add_improved_indicators,
    add_vbo_indicators,
    calculate_natr,
    calculate_volatility_regime,
    calculate_adaptive_noise,
)
from src.utils.indicators_vbo_adaptive import (
    calculate_adaptive_k_value,
    calculate_noise_ratio,
)


@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100, 10000, size=n).astype(float)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# =========================================================================
# calculate_volatility_regime
# =========================================================================


class TestCalculateVolatilityRegime:
    """Tests for calculate_volatility_regime."""

    def test_returns_series_of_regimes(self, ohlcv_df: pd.DataFrame) -> None:
        regime = calculate_volatility_regime(
            ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"]
        )
        assert isinstance(regime, pd.Series)
        assert len(regime) == len(ohlcv_df)
        # Regime values should be 0, 1, or 2
        assert set(regime.dropna().unique()).issubset({0, 1, 2})

    def test_short_data_still_works(self) -> None:
        """With short data, window adapts."""
        high = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        low = pd.Series([9.0, 10.0, 11.0, 12.0, 13.0])
        close = pd.Series([9.5, 10.5, 11.5, 12.5, 13.5])
        regime = calculate_volatility_regime(high, low, close, period=2, window=100)
        assert len(regime) == 5


# =========================================================================
# calculate_adaptive_noise
# =========================================================================


class TestCalculateAdaptiveNoise:
    """Tests for calculate_adaptive_noise."""

    def test_returns_two_series(self, ohlcv_df: pd.DataFrame) -> None:
        short_noise, long_noise = calculate_adaptive_noise(
            ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"]
        )
        assert isinstance(short_noise, pd.Series)
        assert isinstance(long_noise, pd.Series)
        assert len(short_noise) == len(ohlcv_df)
        assert len(long_noise) == len(ohlcv_df)

    def test_values_are_positive_after_warmup(self, ohlcv_df: pd.DataFrame) -> None:
        short_noise, long_noise = calculate_adaptive_noise(
            ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"]
        )
        # After warmup period, values should be positive
        valid_short = short_noise.dropna()
        valid_long = long_noise.dropna()
        assert (valid_short > 0).all()
        assert (valid_long > 0).all()


# =========================================================================
# calculate_noise_ratio
# =========================================================================


class TestCalculateNoiseRatio:
    """Tests for calculate_noise_ratio."""

    def test_returns_series(self, ohlcv_df: pd.DataFrame) -> None:
        ratio = calculate_noise_ratio(
            ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"]
        )
        assert isinstance(ratio, pd.Series)
        assert len(ratio) == len(ohlcv_df)

    def test_ratio_is_positive(self, ohlcv_df: pd.DataFrame) -> None:
        ratio = calculate_noise_ratio(
            ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"]
        )
        valid = ratio.dropna()
        assert (valid >= 0).all()


# =========================================================================
# calculate_adaptive_k_value
# =========================================================================


class TestCalculateAdaptiveKValue:
    """Tests for calculate_adaptive_k_value."""

    def test_returns_series(self, ohlcv_df: pd.DataFrame) -> None:
        k = calculate_adaptive_k_value(
            ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"]
        )
        assert isinstance(k, pd.Series)
        assert len(k) == len(ohlcv_df)

    def test_k_values_scale_with_base(self, ohlcv_df: pd.DataFrame) -> None:
        k_low = calculate_adaptive_k_value(
            ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"], base_k=0.3
        )
        k_high = calculate_adaptive_k_value(
            ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"], base_k=0.8
        )
        # Higher base_k should produce higher values on average
        assert k_high.mean() > k_low.mean()

    def test_regime_based_scaling(self, ohlcv_df: pd.DataFrame) -> None:
        """K value should be adjusted by regime (0.8x, 1.0x, 1.3x)."""
        k = calculate_adaptive_k_value(
            ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"], base_k=0.5
        )
        valid = k.dropna()
        # All values should be one of: 0.4 (0.5*0.8), 0.5 (0.5*1.0), 0.65 (0.5*1.3)
        expected_values = {0.4, 0.5, 0.65}
        for v in valid.unique():
            assert round(v, 2) in expected_values


# =========================================================================
# add_improved_indicators
# =========================================================================


class TestAddImprovedIndicators:
    """Tests for add_improved_indicators."""

    def test_adds_expected_columns(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_improved_indicators(ohlcv_df)
        expected_cols = [
            "atr", "natr", "volatility_regime",
            "short_noise_adaptive", "long_noise_adaptive",
            "noise_ratio", "k_value_adaptive",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_original(self, ohlcv_df: pd.DataFrame) -> None:
        original_cols = set(ohlcv_df.columns)
        add_improved_indicators(ohlcv_df)
        assert set(ohlcv_df.columns) == original_cols

    def test_custom_parameters(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_improved_indicators(
            ohlcv_df, short_period=3, long_period=6, atr_period=10, base_k=0.7
        )
        assert "k_value_adaptive" in result.columns
