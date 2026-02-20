"""Tests for backtester.report_pkg.report_charts."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.backtester.report_pkg.report import BacktestReport
from src.backtester.report_pkg.report_charts import (
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_heatmap,
)


def _make_report(n: int = 400) -> BacktestReport:
    """Build a BacktestReport with synthetic equity data."""
    rng = np.random.default_rng(42)
    equity = 1_000_000.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    start = date(2022, 1, 1)
    dates = np.array([start + timedelta(days=i) for i in range(n)])
    return BacktestReport(
        equity_curve=equity,
        dates=dates,
        trades=pd.DataFrame(),
        strategy_name="TestStrategy",
        initial_capital=1_000_000.0,
    )


class TestPlotEquityCurve:
    def test_returns_figure_when_no_ax(self) -> None:
        import matplotlib.pyplot as plt

        report = _make_report()
        fig = plot_equity_curve(report)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_returns_none_when_ax_provided(self) -> None:
        import matplotlib.pyplot as plt

        report = _make_report()
        _, ax = plt.subplots()
        result = plot_equity_curve(report, ax=ax)
        assert result is None
        plt.close("all")

    def test_no_drawdown_variant(self) -> None:
        import matplotlib.pyplot as plt

        report = _make_report()
        fig = plot_equity_curve(report, show_drawdown=False)
        assert isinstance(fig, Figure)
        plt.close("all")


class TestPlotDrawdown:
    def test_returns_figure_when_no_ax(self) -> None:
        import matplotlib.pyplot as plt

        report = _make_report()
        fig = plot_drawdown(report)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_returns_none_when_ax_provided(self) -> None:
        import matplotlib.pyplot as plt

        report = _make_report()
        _, ax = plt.subplots()
        result = plot_drawdown(report, ax=ax)
        assert result is None
        plt.close("all")


class TestPlotMonthlyHeatmap:
    def test_returns_figure_when_no_ax(self) -> None:
        import matplotlib.pyplot as plt

        report = _make_report()
        fig = plot_monthly_heatmap(report)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_returns_none_when_ax_provided(self) -> None:
        import matplotlib.pyplot as plt

        report = _make_report()
        _, ax = plt.subplots()
        result = plot_monthly_heatmap(report, ax=ax)
        assert result is None
        plt.close("all")
