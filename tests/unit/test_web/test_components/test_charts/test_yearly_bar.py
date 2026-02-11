"""Tests for yearly bar chart component — calculate_yearly_returns function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.web.components.charts.yearly_bar import calculate_yearly_returns


class TestCalculateYearlyReturns:
    """Tests for calculate_yearly_returns."""

    def test_empty_data(self) -> None:
        result = calculate_yearly_returns(np.array([]), np.array([]))
        assert result.empty
        assert list(result.columns) == ["year", "return_pct"]

    def test_single_year(self) -> None:
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        equity = np.linspace(100, 120, 365)  # 100 → 120 = +20%

        result = calculate_yearly_returns(np.array(dates), equity)
        assert len(result) == 1
        assert result.iloc[0]["year"] == 2023
        assert result.iloc[0]["return_pct"] == pytest.approx(20.0, abs=0.5)

    def test_multiple_years(self) -> None:
        dates = pd.date_range("2021-01-01", periods=730, freq="D")
        # Year 1: 100 → 120, Year 2: 120 → 150
        equity = np.concatenate(
            [
                np.linspace(100, 120, 365),
                np.linspace(120, 150, 365),
            ]
        )

        result = calculate_yearly_returns(np.array(dates), equity)
        assert len(result) == 2

    def test_negative_return_year(self) -> None:
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        equity = np.linspace(100, 80, 365)  # 100 → 80 = -20%

        result = calculate_yearly_returns(np.array(dates), equity)
        assert result.iloc[0]["return_pct"] < 0.0

    def test_mixed_returns(self) -> None:
        dates = pd.date_range("2021-01-01", periods=730, freq="D")
        equity = np.concatenate(
            [
                np.linspace(100, 120, 365),  # +20%
                np.linspace(120, 100, 365),  # -16.7%
            ]
        )

        result = calculate_yearly_returns(np.array(dates), equity)
        assert result.iloc[0]["return_pct"] > 0  # 2021 positive
        assert result.iloc[1]["return_pct"] < 0  # 2022 negative

    def test_returns_only_year_and_return_columns(self) -> None:
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        equity = np.linspace(100, 110, 100)

        result = calculate_yearly_returns(np.array(dates), equity)
        assert list(result.columns) == ["year", "return_pct"]
