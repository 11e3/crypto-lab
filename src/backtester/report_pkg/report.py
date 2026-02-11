"""Backtest result reporting and visualization module."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.backtester.report_pkg.report_metrics import (
    PerformanceMetrics,
    calculate_metrics,
    calculate_monthly_returns,
    calculate_sortino_ratio,
    calculate_yearly_returns,
)
from src.backtester.report_pkg.report_summary import print_performance_summary
from src.risk.metrics import PortfolioRiskMetrics
from src.utils.logger import get_logger

__all__ = [
    "PerformanceMetrics",
    "calculate_metrics",
    "calculate_monthly_returns",
    "calculate_yearly_returns",
    "calculate_sortino_ratio",
    "BacktestReport",
    "generate_report",
]

plt.style.use("seaborn-v0_8-whitegrid")
logger = get_logger(__name__)


class BacktestReport:
    """Generates comprehensive backtest reports with visualizations."""

    def __init__(
        self,
        equity_curve: np.ndarray,
        dates: np.ndarray,
        trades: list[Any] | pd.DataFrame,
        strategy_name: str = "Strategy",
        initial_capital: float = 1.0,
    ) -> None:
        """Initialize report generator."""
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.risk_metrics: PortfolioRiskMetrics | None = None
        self.equity_curve = np.array(equity_curve)
        self.dates = np.array(dates)
        self.trades_df = self._build_trades_df(trades)
        self.metrics = calculate_metrics(
            self.equity_curve, self.dates, self.trades_df, initial_capital
        )

    def _build_trades_df(self, trades: list[Any] | pd.DataFrame) -> pd.DataFrame:
        """Build trades DataFrame from input."""
        if isinstance(trades, pd.DataFrame):
            return trades
        if trades:
            return pd.DataFrame(
                [
                    {
                        "ticker": t.ticker,
                        "entry_date": t.entry_date,
                        "entry_price": t.entry_price,
                        "exit_date": t.exit_date,
                        "exit_price": t.exit_price,
                        "amount": t.amount,
                        "pnl": t.pnl,
                        "pnl_pct": t.pnl_pct,
                        "is_whipsaw": t.is_whipsaw,
                    }
                    for t in trades
                ]
            )
        return pd.DataFrame()

    def print_summary(self) -> None:
        """Print performance summary to console."""
        print_performance_summary(self.metrics, self.risk_metrics, self.strategy_name)

    def plot_equity_curve(
        self, ax: Axes | None = None, show_drawdown: bool = True
    ) -> Figure | None:
        """Plot equity curve with optional drawdown."""
        from src.backtester.report_pkg.report_charts import plot_equity_curve

        return plot_equity_curve(self, ax=ax, show_drawdown=show_drawdown)

    def plot_drawdown(self, ax: Axes | None = None) -> Figure | None:
        """Plot drawdown curve."""
        from src.backtester.report_pkg.report_charts import plot_drawdown

        return plot_drawdown(self, ax=ax)

    def plot_monthly_heatmap(self, ax: Axes | None = None) -> Figure | None:
        """Plot monthly returns heatmap."""
        from src.backtester.report_pkg.report_charts import plot_monthly_heatmap

        return plot_monthly_heatmap(self, ax=ax)

    def plot_full_report(self, save_path: Path | str | None = None, show: bool = True) -> Figure:
        """Generate full visual report with all charts."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 2], hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        self.plot_equity_curve(ax=ax1, show_drawdown=True)

        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_drawdown(ax=ax2)

        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_metrics_table(ax=ax3)

        ax4 = fig.add_subplot(gs[2, :])
        self.plot_monthly_heatmap(ax=ax4)

        fig.suptitle(
            f"Backtest Report: {self.strategy_name}", fontsize=14, fontweight="bold", y=0.98
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()  # pragma: no cover
        return fig

    def _plot_metrics_table(self, ax: Axes) -> None:
        """Plot metrics as a table."""
        from src.backtester.report_pkg.report_charts import plot_metrics_table

        plot_metrics_table(ax, self.metrics, self.risk_metrics)

    def to_dataframe(self) -> pd.DataFrame:
        """Export metrics as DataFrame."""
        from src.backtester.report_pkg.report_metrics import metrics_to_dataframe

        return metrics_to_dataframe(self.metrics)


def generate_report(
    result: Any,
    strategy_name: str | None = None,
    save_path: Path | str | None = None,
    show: bool = True,
    format: str = "png",
    strategy_obj: Any = None,
    config: Any = None,
    tickers: list[str] | None = None,
) -> BacktestReport:
    """Convenience function to generate report from BacktestResult."""
    report = BacktestReport(
        equity_curve=result.equity_curve,
        dates=result.dates,
        trades=result.trades,
        strategy_name=strategy_name or result.strategy_name,
        initial_capital=result.config.initial_capital if result.config else 1.0,
    )

    if hasattr(result, "risk_metrics") and result.risk_metrics:
        report.risk_metrics = result.risk_metrics

    report.print_summary()

    if save_path:
        _save_report(report, save_path, format, show, strategy_obj, config, tickers)
    elif show:
        report.plot_full_report(save_path=None, show=show)

    return report


def _save_report(
    report: BacktestReport,
    save_path: Path | str,
    format: str,
    show: bool,
    strategy_obj: Any,
    config: Any,
    tickers: list[str] | None,
) -> None:
    """Save report to file."""
    save_path = Path(save_path) if isinstance(save_path, str) else save_path

    if save_path.suffix.lower() == ".html":
        format = "html"
    elif save_path.suffix.lower() == ".png":
        format = "png"

    if format == "html":
        from src.backtester.html.html_report import generate_html_report

        generate_html_report(
            report, save_path, strategy_obj=strategy_obj, config=config, tickers=tickers
        )
    else:
        report.plot_full_report(save_path=save_path, show=show)
