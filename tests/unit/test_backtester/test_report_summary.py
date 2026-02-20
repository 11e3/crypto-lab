"""Tests for backtester.report_pkg.report_summary."""

from __future__ import annotations

from datetime import date

import numpy as np

from src.backtester.report_pkg.report_metrics_models import PerformanceMetrics
from src.backtester.report_pkg.report_summary import print_performance_summary


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


class TestPrintPerformanceSummary:
    def test_runs_without_error(self) -> None:
        metrics = _make_metrics()
        # Output goes to logger — just verify no exception raised
        print_performance_summary(metrics, risk_metrics=None, strategy_name="TestStrategy")

    def test_accepts_different_strategy_names(self) -> None:
        metrics = _make_metrics()
        for name in ["VBO", "MomentumStrategy", "BTC_Only"]:
            print_performance_summary(metrics, risk_metrics=None, strategy_name=name)

    def test_risk_metrics_none_does_not_raise(self) -> None:
        metrics = _make_metrics()
        print_performance_summary(metrics, risk_metrics=None, strategy_name="NoRisk")
