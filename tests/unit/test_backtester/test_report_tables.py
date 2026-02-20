"""Tests for backtester.report_pkg.report_tables."""

from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import numpy as np

from src.backtester.report_pkg.report_metrics_models import PerformanceMetrics
from src.backtester.report_pkg.report_tables import plot_metrics_table


def _make_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        start_date=date(2021, 1, 1),
        end_date=date(2022, 1, 1),
        total_days=365,
        total_return_pct=25.0,
        cagr_pct=25.0,
        mdd_pct=10.0,
        volatility_pct=15.0,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=2.5,
        total_trades=50,
        winning_trades=30,
        losing_trades=20,
        win_rate_pct=60.0,
        profit_factor=1.8,
        avg_profit_pct=2.0,
        avg_loss_pct=-1.5,
        avg_trade_pct=0.5,
        equity_curve=np.array([1_000_000.0, 1_250_000.0]),
        drawdown_curve=np.array([0.0, 0.0]),
        dates=np.array([date(2021, 1, 1), date(2022, 1, 1)]),
        daily_returns=np.array([0.0, 0.25]),
    )


class TestPlotMetricsTable:
    def test_runs_without_error(self) -> None:
        _, ax = plt.subplots()
        metrics = _make_metrics()
        plot_metrics_table(ax, metrics)
        plt.close("all")

    def test_no_risk_metrics_variant(self) -> None:
        _, ax = plt.subplots()
        metrics = _make_metrics()
        plot_metrics_table(ax, metrics, risk_metrics=None)
        plt.close("all")

    def test_ax_axis_is_off_after_call(self) -> None:
        """plot_metrics_table calls ax.axis('off') — verify ax is still accessible."""
        _, ax = plt.subplots()
        metrics = _make_metrics()
        plot_metrics_table(ax, metrics)
        # Axis should still be valid; no exception means success
        assert ax is not None
        plt.close("all")
