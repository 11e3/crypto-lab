"""Tests for backtester.report_pkg.report_returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtester.report_pkg.report_returns import (
    build_equity_dataframe,
    calculate_monthly_returns,
    calculate_yearly_returns,
)

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _make_equity(n: int = 400, seed: int = 9) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    equity = 1_000_000.0 * np.cumprod(1 + rng.normal(0.001, 0.005, n))
    dates = pd.date_range("2022-01-01", periods=n, freq="D").to_numpy()
    return equity, dates


class TestBuildEquityDataframe:
    def test_returns_dataframe(self) -> None:
        equity, dates = _make_equity(50)
        df = build_equity_dataframe(equity, dates)
        assert isinstance(df, pd.DataFrame)

    def test_has_equity_column(self) -> None:
        equity, dates = _make_equity(50)
        df = build_equity_dataframe(equity, dates)
        assert "equity" in df.columns

    def test_index_is_datetimeindex(self) -> None:
        equity, dates = _make_equity(50)
        df = build_equity_dataframe(equity, dates)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_length_matches_input(self) -> None:
        equity, dates = _make_equity(60)
        df = build_equity_dataframe(equity, dates)
        assert len(df) == 60


class TestCalculateMonthlyReturns:
    def test_returns_dataframe(self) -> None:
        equity, dates = _make_equity()
        result = calculate_monthly_returns(equity, dates)
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_valid_month_names(self) -> None:
        equity, dates = _make_equity()
        result = calculate_monthly_returns(equity, dates)
        for col in result.columns:
            assert col in _MONTH_NAMES

    def test_index_contains_integer_years(self) -> None:
        equity, dates = _make_equity()
        result = calculate_monthly_returns(equity, dates)
        for year in result.index:
            assert isinstance(int(year), int)

    def test_multi_year_data_has_multiple_rows(self) -> None:
        # 400 days spans 2+ years → at least 2 rows in pivot
        equity, dates = _make_equity(400)
        result = calculate_monthly_returns(equity, dates)
        assert len(result) >= 1


class TestCalculateYearlyReturns:
    def test_returns_series(self) -> None:
        equity, dates = _make_equity()
        result = calculate_yearly_returns(equity, dates)
        assert isinstance(result, pd.Series)

    def test_index_contains_integer_years(self) -> None:
        equity, dates = _make_equity()
        result = calculate_yearly_returns(equity, dates)
        for year in result.index:
            assert isinstance(int(year), int)

    def test_multi_year_data_has_multiple_entries(self) -> None:
        equity, dates = _make_equity(400)
        result = calculate_yearly_returns(equity, dates)
        assert len(result) >= 1
