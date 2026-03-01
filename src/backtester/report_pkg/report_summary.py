"""Report summary printer for backtest results."""

from typing import TYPE_CHECKING

from src.backtester.report_pkg.report_metrics import PerformanceMetrics
from src.risk.metrics import PortfolioRiskMetrics
from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def print_performance_summary(
    metrics: PerformanceMetrics,
    risk_metrics: PortfolioRiskMetrics | None,
    strategy_name: str,
) -> None:
    """
    Print performance summary to console using structured logging.

    Args:
        metrics: Performance metrics
        risk_metrics: Optional risk metrics
        strategy_name: Name of the strategy
    """
    m = metrics

    output_lines = [
        f"\n{'=' * 60}",
        f"  BACKTEST REPORT: {strategy_name}",
        f"{'=' * 60}",
        "\n[Period]",
        f"   Start Date:     {m.start_date}",
        f"   End Date:       {m.end_date}",
        f"   Total Days:     {m.total_days:,}",
        "\n[Returns]",
        f"   Total Return:   {m.total_return_pct:,.2f}%",
        f"   CAGR:           {m.cagr_pct:.2f}%",
        "\n[Risk]",
        f"   Max Drawdown:   -{m.mdd_pct:.2f}%",
        f"   Volatility:     {m.volatility_pct:.2f}%",
        "\n[Risk-Adjusted Returns]",
        f"   Sharpe Ratio:   {m.sharpe_ratio:.2f}",
        f"   Sortino Ratio:  {m.sortino_ratio:.2f}",
        f"   Calmar Ratio:   {m.calmar_ratio:.2f}",
        "\n[Trade Statistics]",
        f"   Total Trades:   {m.total_trades:,}",
        f"   Winning:        {m.winning_trades:,}",
        f"   Losing:         {m.losing_trades:,}",
        f"   Win Rate:       {m.win_rate_pct:.2f}%",
        f"   Profit Factor:  {m.profit_factor:.2f}",
        "\n[Per-Trade]",
        f"   Avg Profit:     {m.avg_profit_pct:.2f}%",
        f"   Avg Loss:       {m.avg_loss_pct:.2f}%",
        f"   Avg Trade:      {m.avg_trade_pct:.2f}%",
    ]

    # Add risk metrics if available
    if risk_metrics:
        output_lines.extend(_format_risk_metrics(risk_metrics))

    output_lines.append(f"\n{'=' * 60}\n")

    for line in output_lines:
        logger.info(line)


def _format_risk_metrics(risk_metrics: PortfolioRiskMetrics) -> list[str]:
    """Format risk metrics as output lines."""
    lines = [
        "\n[Portfolio Risk Metrics]",
        f"   VaR (95%):      {risk_metrics.var_95 * 100:.2f}%",
        f"   CVaR (95%):     {risk_metrics.cvar_95 * 100:.2f}%",
        f"   VaR (99%):      {risk_metrics.var_99 * 100:.2f}%",
        f"   CVaR (99%):    {risk_metrics.cvar_99 * 100:.2f}%",
        f"   Portfolio Vol:  {risk_metrics.portfolio_volatility * 100:.2f}%",
    ]

    if risk_metrics.avg_correlation is not None:
        lines.extend(
            [
                f"   Avg Correlation: {risk_metrics.avg_correlation:.3f}",
                f"   Max Correlation: {risk_metrics.max_correlation:.3f}",
                f"   Min Correlation: {risk_metrics.min_correlation:.3f}",
            ]
        )

    if risk_metrics.max_position_pct is not None and risk_metrics.max_position_pct > 0:
        lines.extend(
            [
                f"   Max Position %:  {risk_metrics.max_position_pct * 100:.2f}%",
                f"   Position HHI:     {risk_metrics.position_concentration:.3f}",
            ]
        )

    if risk_metrics.portfolio_beta is not None:
        lines.append(f"   Portfolio Beta:  {risk_metrics.portfolio_beta:.2f}")

    return lines
