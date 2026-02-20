"""Tests for backtester.report_pkg.report_metrics_models."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src.backtester.report_pkg.report_metrics_models import PerformanceMetrics


def _make_metrics(**overrides: object) -> PerformanceMetrics:
    defaults: dict[str, object] = {
        "start_date": date(2021, 1, 1),
        "end_date": date(2022, 1, 1),
        "total_days": 365,
        "total_return_pct": 25.0,
        "cagr_pct": 25.0,
        "mdd_pct": 10.0,
        "volatility_pct": 15.0,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "calmar_ratio": 2.5,
        "total_trades": 50,
        "winning_trades": 30,
        "losing_trades": 20,
        "win_rate_pct": 60.0,
        "profit_factor": 1.8,
        "avg_profit_pct": 2.0,
        "avg_loss_pct": -1.5,
        "avg_trade_pct": 0.5,
        "equity_curve": np.array([1_000_000.0, 1_100_000.0, 1_250_000.0]),
        "drawdown_curve": np.array([0.0, 0.0, 0.0]),
        "dates": np.array([date(2021, 1, 1), date(2021, 7, 1), date(2022, 1, 1)]),
        "daily_returns": np.array([0.0, 0.1, 0.14]),
    }
    defaults.update(overrides)
    return PerformanceMetrics(**defaults)  # type: ignore[arg-type]


class TestPerformanceMetrics:
    def test_creation_succeeds(self) -> None:
        m = _make_metrics()
        assert isinstance(m, PerformanceMetrics)

    def test_float_fields_are_float(self) -> None:
        m = _make_metrics()
        assert isinstance(m.total_return_pct, float)
        assert isinstance(m.sharpe_ratio, float)
        assert isinstance(m.sortino_ratio, float)
        assert isinstance(m.calmar_ratio, float)
        assert isinstance(m.win_rate_pct, float)

    def test_int_fields_are_int(self) -> None:
        m = _make_metrics()
        assert isinstance(m.total_trades, int)
        assert isinstance(m.winning_trades, int)
        assert isinstance(m.losing_trades, int)
        assert isinstance(m.total_days, int)

    def test_date_fields(self) -> None:
        m = _make_metrics()
        assert m.start_date == date(2021, 1, 1)
        assert m.end_date == date(2022, 1, 1)
        assert m.total_days == 365

    def test_array_fields_are_ndarray(self) -> None:
        m = _make_metrics()
        assert isinstance(m.equity_curve, np.ndarray)
        assert isinstance(m.drawdown_curve, np.ndarray)
        assert isinstance(m.daily_returns, np.ndarray)

    def test_win_rate_consistent_with_trade_counts(self) -> None:
        m = _make_metrics(winning_trades=6, losing_trades=4, win_rate_pct=60.0)
        expected = m.winning_trades / (m.winning_trades + m.losing_trades) * 100
        assert m.win_rate_pct == pytest.approx(expected)

    def test_field_values_stored_correctly(self) -> None:
        m = _make_metrics(sharpe_ratio=2.34, mdd_pct=15.5)
        assert m.sharpe_ratio == pytest.approx(2.34)
        assert m.mdd_pct == pytest.approx(15.5)


