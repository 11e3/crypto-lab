"""Tests for utils.indicators_momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.indicators_momentum import bollinger_bands, macd, rsi, stochastic


def _make_series(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = 100.0 + rng.normal(0, 1, n).cumsum()
    return pd.Series(values, name="close")


def _make_ohlc(n: int = 100, seed: int = 5) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)
    return pd.Series(high), pd.Series(low), pd.Series(close)


class TestRsi:
    def test_output_length_matches_input(self) -> None:
        series = _make_series(100)
        result = rsi(series, period=14)
        assert len(result) == len(series)

    def test_valid_values_in_range_0_to_100(self) -> None:
        series = _make_series(200)
        result = rsi(series, period=14)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_returns_pandas_series(self) -> None:
        series = _make_series(50)
        result = rsi(series, period=14)
        assert isinstance(result, pd.Series)

    def test_custom_period(self) -> None:
        series = _make_series(100)
        result_7 = rsi(series, period=7)
        result_21 = rsi(series, period=21)
        # Both should return series of same length
        assert len(result_7) == len(result_21) == len(series)


class TestBollingerBands:
    def test_returns_three_series(self) -> None:
        series = _make_series(100)
        upper, middle, lower = bollinger_bands(series, period=20)
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_upper_above_middle_above_lower(self) -> None:
        series = _make_series(100)
        upper, middle, lower = bollinger_bands(series, period=20)
        valid = upper.notna() & middle.notna() & lower.notna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_middle_equals_rolling_sma(self) -> None:
        series = _make_series(50)
        _, middle, _ = bollinger_bands(series, period=10)
        expected = series.rolling(window=10, min_periods=10).mean()
        pd.testing.assert_series_equal(middle, expected, check_names=False)

    def test_band_width_scales_with_std_dev(self) -> None:
        series = _make_series(100)
        upper1, middle1, lower1 = bollinger_bands(series, period=20, std_dev=1.0)
        upper2, middle2, lower2 = bollinger_bands(series, period=20, std_dev=2.0)
        valid = upper1.notna() & upper2.notna()
        # 2× std_dev bands should be wider
        width1 = (upper1 - lower1)[valid]
        width2 = (upper2 - lower2)[valid]
        assert (width2 >= width1).all()


class TestMacd:
    def test_returns_three_series(self) -> None:
        series = _make_series(200)
        macd_line, signal_line, histogram = macd(series)
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

    def test_histogram_equals_macd_minus_signal(self) -> None:
        series = _make_series(200)
        macd_line, signal_line, histogram = macd(series)
        expected = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected, check_names=False)

    def test_output_length_matches_input(self) -> None:
        series = _make_series(200)
        macd_line, signal_line, histogram = macd(series)
        assert len(macd_line) == len(series)
        assert len(signal_line) == len(series)
        assert len(histogram) == len(series)


class TestStochastic:
    def test_returns_two_series(self) -> None:
        high, low, close = _make_ohlc()
        k, d = stochastic(high, low, close)
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)

    def test_k_values_in_range_0_to_100(self) -> None:
        high, low, close = _make_ohlc(100)
        k, _ = stochastic(high, low, close, k_period=14)
        valid = k.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_output_length_matches_input(self) -> None:
        high, low, close = _make_ohlc(100)
        k, d = stochastic(high, low, close)
        assert len(k) == len(close)
        assert len(d) == len(close)

    def test_d_is_smoothed_k(self) -> None:
        high, low, close = _make_ohlc(100)
        k, d = stochastic(high, low, close, k_period=14, d_period=3)
        expected_d = k.rolling(window=3, min_periods=3).mean()
        pd.testing.assert_series_equal(d, expected_d, check_names=False)
