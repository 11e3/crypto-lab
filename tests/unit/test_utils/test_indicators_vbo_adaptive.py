"""
Tests for adaptive VBO indicators.
"""

import pandas as pd
import pytest

from src.utils.indicators_vbo import (
    calculate_adaptive_noise,
    calculate_volatility_regime,
)


class TestCalculateVolatilityRegime:
    """Tests for calculate_volatility_regime function."""

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample high, low, close series."""
        high = pd.Series([102.0, 103.0, 104.0, 105.0, 106.0])
        low = pd.Series([99.0, 100.0, 101.0, 102.0, 103.0])
        close = pd.Series([101.0, 102.0, 103.0, 104.0, 105.0])
        return high, low, close

    def test_volatility_regime_returns_series(self, sample_data: tuple) -> None:
        """Test that function returns pandas Series."""
        high, low, close = sample_data
        result = calculate_volatility_regime(high, low, close)

        assert isinstance(result, pd.Series)
        assert len(result) == len(high)

    def test_volatility_regime_values_in_range(self, sample_data: tuple) -> None:
        """Test that regime values are 0, 1, or 2."""
        high, low, close = sample_data
        result = calculate_volatility_regime(high, low, close)

        unique_values = set(result.unique())
        assert unique_values.issubset({0, 1, 2})

    def test_volatility_regime_custom_period(self, sample_data: tuple) -> None:
        """Test with custom period parameter."""
        high, low, close = sample_data
        result = calculate_volatility_regime(high, low, close, period=7)

        assert isinstance(result, pd.Series)
        assert len(result) == len(high)

    def test_volatility_regime_custom_window(self, sample_data: tuple) -> None:
        """Test with custom window parameter."""
        high, low, close = sample_data
        result = calculate_volatility_regime(high, low, close, window=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(high)

    def test_volatility_regime_empty_series(self) -> None:
        """Test with empty series."""
        high = pd.Series([], dtype=float)
        low = pd.Series([], dtype=float)
        close = pd.Series([], dtype=float)

        result = calculate_volatility_regime(high, low, close)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_volatility_regime_single_value(self) -> None:
        """Test with single value series."""
        high = pd.Series([102.0])
        low = pd.Series([99.0])
        close = pd.Series([101.0])

        result = calculate_volatility_regime(high, low, close)

        assert isinstance(result, pd.Series)
        assert len(result) == 1

    def test_volatility_regime_large_window(self, sample_data: tuple) -> None:
        """Test with window larger than data length."""
        high, low, close = sample_data
        # Window > data length should be clamped
        result = calculate_volatility_regime(high, low, close, window=1000)

        assert isinstance(result, pd.Series)
        assert len(result) == len(high)


class TestCalculateAdaptiveNoise:
    """Tests for calculate_adaptive_noise function."""

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample high, low, close series."""
        high = pd.Series([102.0, 103.0, 104.0, 105.0, 106.0])
        low = pd.Series([99.0, 100.0, 101.0, 102.0, 103.0])
        close = pd.Series([101.0, 102.0, 103.0, 104.0, 105.0])
        return high, low, close

    def test_adaptive_noise_returns_tuple(self, sample_data: tuple) -> None:
        """Test that function returns tuple of two series."""
        high, low, close = sample_data
        result = calculate_adaptive_noise(high, low, close)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.Series)
        assert isinstance(result[1], pd.Series)

    def test_adaptive_noise_same_length(self, sample_data: tuple) -> None:
        """Test that output series have same length as input."""
        high, low, close = sample_data
        short_noise, long_noise = calculate_adaptive_noise(high, low, close)

        assert len(short_noise) == len(high)
        assert len(long_noise) == len(high)

    def test_adaptive_noise_custom_periods(self, sample_data: tuple) -> None:
        """Test with custom period parameters."""
        high, low, close = sample_data
        result = calculate_adaptive_noise(
            high, low, close, short_period=3, long_period=6, atr_period=10
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_adaptive_noise_empty_series(self) -> None:
        """Test with empty series."""
        high = pd.Series([], dtype=float)
        low = pd.Series([], dtype=float)
        close = pd.Series([], dtype=float)

        short_noise, long_noise = calculate_adaptive_noise(high, low, close)

        assert isinstance(short_noise, pd.Series)
        assert isinstance(long_noise, pd.Series)
        assert len(short_noise) == 0
        assert len(long_noise) == 0

    def test_adaptive_noise_single_value(self) -> None:
        """Test with single value series."""
        high = pd.Series([102.0])
        low = pd.Series([99.0])
        close = pd.Series([101.0])

        short_noise, long_noise = calculate_adaptive_noise(high, low, close)

        assert isinstance(short_noise, pd.Series)
        assert isinstance(long_noise, pd.Series)
        assert len(short_noise) == 1
        assert len(long_noise) == 1

    def test_adaptive_noise_returns_numeric(self, sample_data: tuple) -> None:
        """Test that noise values are numeric or NaN."""
        high, low, close = sample_data
        short_noise, long_noise = calculate_adaptive_noise(high, low, close)

        # Values should be numeric (float) or NaN during warmup
        assert short_noise.dtype in ["float64", "float32"]
        assert long_noise.dtype in ["float64", "float32"]
