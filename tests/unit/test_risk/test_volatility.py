"""Unit tests for volatility calculation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.risk.volatility import calculate_return_volatility


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample historical data with known volatility."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    return pd.DataFrame({"close": prices}, index=dates)


class TestCalculateReturnVolatility:
    """Tests for calculate_return_volatility."""

    def test_returns_float(self, sample_data: pd.DataFrame) -> None:
        """Should return a positive float for valid data."""
        result = calculate_return_volatility(sample_data, lookback_period=20)
        assert isinstance(result, float)
        assert result > 0

    def test_insufficient_data(self, sample_data: pd.DataFrame) -> None:
        """Should return None when data is shorter than lookback."""
        result = calculate_return_volatility(sample_data.head(5), lookback_period=20)
        assert result is None

    def test_zero_volatility(self) -> None:
        """Should return None when all prices are constant."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        flat_data = pd.DataFrame({"close": [100.0] * 30}, index=dates)
        result = calculate_return_volatility(flat_data, lookback_period=20)
        assert result is None

    def test_uses_lookback_period(self, sample_data: pd.DataFrame) -> None:
        """Should only use the most recent lookback_period rows."""
        vol_short = calculate_return_volatility(sample_data, lookback_period=10)
        vol_long = calculate_return_volatility(sample_data, lookback_period=40)
        assert vol_short is not None
        assert vol_long is not None
        # Different lookback periods should generally give different values
        assert vol_short != vol_long

    def test_custom_price_column(self) -> None:
        """Should work with a custom price column name."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        np.random.seed(42)
        data = pd.DataFrame(
            {"adjusted_close": 100 + np.cumsum(np.random.randn(30) * 0.5)},
            index=dates,
        )
        result = calculate_return_volatility(
            data, lookback_period=20, price_column="adjusted_close"
        )
        assert isinstance(result, float)
        assert result > 0

    def test_exact_lookback_boundary(self) -> None:
        """Should work when data length equals lookback period exactly."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        np.random.seed(42)
        data = pd.DataFrame({"close": 100 + np.cumsum(np.random.randn(20) * 0.5)}, index=dates)
        result = calculate_return_volatility(data, lookback_period=20)
        assert isinstance(result, float)
        assert result > 0

    def test_empty_dataframe(self) -> None:
        """Should return None for empty DataFrame."""
        empty = pd.DataFrame({"close": []})
        result = calculate_return_volatility(empty, lookback_period=20)
        assert result is None

    def test_single_row(self) -> None:
        """Should return None for single-row DataFrame."""
        data = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime(["2023-01-01"]))
        result = calculate_return_volatility(data, lookback_period=1)
        # Single row → pct_change produces NaN → std is NaN → None
        assert result is None
