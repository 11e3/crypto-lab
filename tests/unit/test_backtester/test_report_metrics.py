"""Tests for report_pkg.report_metrics module."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtester.report_pkg.report_metrics import (
    PerformanceMetrics,
    calculate_metrics,
    metrics_to_dataframe,
)


def _make_equity(n: int = 100, growth: float = 0.001) -> np.ndarray:
    """Build a simple growing equity curve."""
    return np.cumprod(np.full(n, 1 + growth)) * 1_000_000.0


def _make_dates(n: int = 100) -> np.ndarray:
    """Build an array of consecutive date objects."""
    start = date(2023, 1, 2)
    return np.array([date.fromordinal(start.toordinal() + i) for i in range(n)])


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=["pnl_pct", "entry_price", "exit_price"])


class TestCalculateMetrics:
    """Tests for calculate_metrics()."""

    def test_returns_performance_metrics_instance(self) -> None:
        equity = _make_equity()
        dates = _make_dates()
        result = calculate_metrics(equity, dates, _empty_trades(), initial_capital=1_000_000.0)
        assert isinstance(result, PerformanceMetrics)

    def test_positive_return_on_growing_curve(self) -> None:
        equity = _make_equity(n=100, growth=0.002)
        dates = _make_dates(100)
        result = calculate_metrics(equity, dates, _empty_trades(), initial_capital=equity[0])
        assert result.total_return_pct > 0

    def test_mdd_is_zero_for_monotone_curve(self) -> None:
        equity = np.linspace(1_000_000.0, 2_000_000.0, 100)
        dates = _make_dates(100)
        result = calculate_metrics(equity, dates, _empty_trades(), initial_capital=1_000_000.0)
        assert result.mdd_pct == pytest.approx(0.0, abs=1e-6)

    def test_sharpe_present(self) -> None:
        equity = _make_equity()
        dates = _make_dates()
        result = calculate_metrics(equity, dates, _empty_trades(), initial_capital=1_000_000.0)
        assert isinstance(result.sharpe_ratio, float)

    def test_zero_trades_gives_zero_win_rate(self) -> None:
        equity = _make_equity()
        dates = _make_dates()
        result = calculate_metrics(equity, dates, _empty_trades(), initial_capital=1_000_000.0)
        assert result.total_trades == 0
        assert result.win_rate_pct == 0.0


class TestMetricsToDataframe:
    """Tests for metrics_to_dataframe()."""

    def test_returns_dataframe(self) -> None:
        equity = _make_equity()
        dates = _make_dates()
        metrics = calculate_metrics(equity, dates, _empty_trades(), initial_capital=1_000_000.0)
        df = metrics_to_dataframe(metrics)
        assert isinstance(df, pd.DataFrame)

    def test_has_metric_and_value_columns(self) -> None:
        equity = _make_equity()
        dates = _make_dates()
        metrics = calculate_metrics(equity, dates, _empty_trades(), initial_capital=1_000_000.0)
        df = metrics_to_dataframe(metrics)
        assert "Metric" in df.columns
        assert "Value" in df.columns

    def test_row_count(self) -> None:
        equity = _make_equity()
        dates = _make_dates()
        metrics = calculate_metrics(equity, dates, _empty_trades(), initial_capital=1_000_000.0)
        df = metrics_to_dataframe(metrics)
        assert len(df) == 18
