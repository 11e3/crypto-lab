"""
Report table plotting functions.

Contains matplotlib-based table plotting for backtest reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.axes import Axes

if TYPE_CHECKING:
    from src.backtester.report_pkg.report import PerformanceMetrics
    from src.risk.metrics import PortfolioRiskMetrics

__all__ = ["plot_metrics_table"]


def plot_metrics_table(
    ax: Axes,
    metrics: PerformanceMetrics,
    risk_metrics: PortfolioRiskMetrics | None = None,
) -> None:
    """
    Plot metrics as a table.

    Args:
        ax: Matplotlib axes
        metrics: Performance metrics
        risk_metrics: Optional risk metrics
    """
    ax.axis("off")

    m = metrics
    data = [
        ["Total Return", f"{m.total_return_pct:,.2f}%"],
        ["CAGR", f"{m.cagr_pct:.2f}%"],
        ["Max Drawdown", f"-{m.mdd_pct:.2f}%"],
        ["Sharpe Ratio", f"{m.sharpe_ratio:.2f}"],
        ["Sortino Ratio", f"{m.sortino_ratio:.2f}"],
        ["Calmar Ratio", f"{m.calmar_ratio:.2f}"],
        ["Total Trades", f"{m.total_trades:,}"],
        ["Win Rate", f"{m.win_rate_pct:.2f}%"],
        ["Profit Factor", f"{m.profit_factor:.2f}"],
        ["Avg Profit", f"{m.avg_profit_pct:.2f}%"],
        ["Avg Loss", f"{m.avg_loss_pct:.2f}%"],
    ]

    # Add risk metrics if available
    if risk_metrics:
        data.extend(
            [
                ["", ""],  # Separator
                ["VaR (95%)", f"{risk_metrics.var_95 * 100:.2f}%"],
                ["CVaR (95%)", f"{risk_metrics.cvar_95 * 100:.2f}%"],
                ["Portfolio Vol", f"{risk_metrics.portfolio_volatility * 100:.2f}%"],
            ]
        )
        if risk_metrics.avg_correlation != 0.0 and risk_metrics.avg_correlation is not None:
            data.append(["Avg Correlation", f"{risk_metrics.avg_correlation:.3f}"])

    table = ax.table(
        cellText=data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#E6E6E6")
