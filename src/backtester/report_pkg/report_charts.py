"""
Report chart plotting functions.

Contains matplotlib-based chart plotting for backtest reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Re-export plot_metrics_table for backward compatibility
from src.backtester.report_pkg.report_tables import plot_metrics_table

if TYPE_CHECKING:
    from src.backtester.report_pkg.report import BacktestReport

__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_monthly_heatmap",
    "plot_metrics_table",
]


def plot_equity_curve(
    report: BacktestReport,
    ax: Axes | None = None,
    show_drawdown: bool = True,
) -> Figure | None:
    """
    Plot equity curve with optional drawdown.

    Args:
        report: BacktestReport instance
        ax: Matplotlib axes (creates new figure if None)
        show_drawdown: Whether to show drawdown on secondary axis

    Returns:
        Figure if ax was None, else None
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot equity curve
    ax.plot(report.dates, report.equity_curve, "b-", linewidth=1.5, label="Equity")
    ax.fill_between(
        report.dates,
        report.initial_capital,
        report.equity_curve,  # type: ignore[arg-type]
        alpha=0.3,
        color="blue",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Equity", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.set_title(f"{report.strategy_name} - Equity Curve")

    # Add drawdown on secondary axis
    if show_drawdown:
        ax2 = ax.twinx()
        ax2.fill_between(
            report.dates,
            0,
            -report.metrics.drawdown_curve,  # type: ignore[arg-type]
            alpha=0.3,
            color="red",
            label="Drawdown",
        )
        ax2.set_ylabel("Drawdown (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim(bottom=-report.metrics.mdd_pct * 1.2, top=5)

    ax.grid(True, alpha=0.3)

    if fig:
        plt.tight_layout()

    return fig


def plot_drawdown(
    report: BacktestReport,
    ax: Axes | None = None,
) -> Figure | None:
    """
    Plot drawdown curve.

    Args:
        report: BacktestReport instance
        ax: Matplotlib axes

    Returns:
        Figure if ax was None
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(
        report.dates,
        0,
        -report.metrics.drawdown_curve,  # type: ignore[arg-type]
        color="red",
        alpha=0.5,
    )
    ax.plot(report.dates, -report.metrics.drawdown_curve, "r-", linewidth=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"{report.strategy_name} - Drawdown")
    ax.grid(True, alpha=0.3)

    # Add MDD annotation
    mdd_idx = int(np.argmax(report.metrics.drawdown_curve))
    ax.annotate(
        f"MDD: {-report.metrics.mdd_pct:.1f}%",
        xy=(report.dates[mdd_idx], -report.metrics.mdd_pct),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color="red",
    )

    if fig:
        plt.tight_layout()

    return fig


def plot_monthly_heatmap(
    report: BacktestReport,
    ax: Axes | None = None,
) -> Figure | None:
    """
    Plot monthly returns heatmap.

    Args:
        report: BacktestReport instance
        ax: Matplotlib axes

    Returns:
        Figure if ax was None
    """
    from src.backtester.report_pkg.report import calculate_monthly_returns, calculate_yearly_returns

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    monthly = calculate_monthly_returns(report.equity_curve, report.dates)

    # Add yearly totals column
    yearly = calculate_yearly_returns(report.equity_curve, report.dates)
    monthly["Year"] = yearly

    # Create heatmap
    im = ax.imshow(monthly.values, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=20)

    # Set ticks
    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels(list(monthly.columns))  # type: ignore[operator]
    ax.set_yticks(range(len(monthly.index)))  # type: ignore[operator]
    ax.set_yticklabels(list(monthly.index))  # type: ignore[operator]

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Return (%)")

    # Add value annotations
    for i in range(len(monthly.index)):
        for j in range(len(monthly.columns)):
            val = monthly.iloc[i, j]
            # Type guard: ensure val is numeric
            if isinstance(val, (int, float, np.number)) and not np.isnan(val):
                color = "white" if abs(float(val)) > 10 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

    ax.set_title(f"{report.strategy_name} - Monthly Returns Heatmap (%)")

    if fig:
        plt.tight_layout()

    return fig
