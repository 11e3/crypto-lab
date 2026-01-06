"""
Backtest result reporting and visualization module.

Generates comprehensive reports with:
- Performance metrics
- Equity curve charts
- Drawdown charts
- Monthly/yearly returns heatmaps
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ANNUALIZATION_FACTOR, RISK_FREE_RATE
from src.utils.logger import get_logger

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Period
    start_date: date
    end_date: date
    total_days: int

    # Returns
    total_return_pct: float
    cagr_pct: float

    # Risk
    mdd_pct: float
    volatility_pct: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float

    # Per-trade
    avg_profit_pct: float
    avg_loss_pct: float
    avg_trade_pct: float

    # Equity data
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    dates: np.ndarray
    daily_returns: np.ndarray


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    annualization: float = ANNUALIZATION_FACTOR,
) -> float:
    """
    Calculate Sortino ratio (downside deviation).

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (daily)
        annualization: Annualization factor

    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate
    downside_returns = np.minimum(excess_returns, 0)
    downside_std = np.std(downside_returns)

    if downside_std <= 0:
        return 0.0

    mean_excess = float(np.mean(excess_returns))
    sqrt_annualization = float(np.sqrt(annualization))
    result: float = (mean_excess / downside_std) * sqrt_annualization
    return result


def calculate_metrics(
    equity_curve: np.ndarray,
    dates: np.ndarray,
    trades_df: pd.DataFrame,
    initial_capital: float = 1.0,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: Array of equity values
        dates: Array of dates
        trades_df: DataFrame with trade records
        initial_capital: Starting capital

    Returns:
        PerformanceMetrics object
    """
    # Basic period info
    start_date = dates[0]
    end_date = dates[-1]
    total_days = (end_date - start_date).days

    # Returns
    final_equity = equity_curve[-1]
    total_return_pct = (final_equity / initial_capital - 1) * 100

    # CAGR (matching legacy calculation)
    if total_days <= 0:
        cagr_pct = 0.0
    elif initial_capital <= 0:
        cagr_pct = 0.0
    elif final_equity <= 0:
        cagr_pct = -100.0
    else:
        cagr_pct = ((final_equity / initial_capital) ** (365.0 / total_days) - 1) * 100

    # Daily returns
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)  # First day has 0 return

    # Volatility (annualized)
    volatility_pct = np.std(daily_returns) * np.sqrt(ANNUALIZATION_FACTOR) * 100

    # MDD and drawdown curve
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (cummax - equity_curve) / cummax
    mdd_pct = np.nanmax(drawdown) * 100

    # Sharpe ratio
    if np.std(daily_returns) > 0:
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(
            ANNUALIZATION_FACTOR
        )
    else:
        sharpe_ratio = 0.0

    # Sortino ratio
    sortino_ratio = calculate_sortino_ratio(daily_returns)

    # Calmar ratio
    calmar_ratio = cagr_pct / mdd_pct if mdd_pct > 0 else 0.0

    # Trade statistics
    if len(trades_df) > 0:
        closed_trades = trades_df[trades_df["exit_date"].notna()]
        total_trades = len(closed_trades)

        if total_trades > 0:
            winning = closed_trades[closed_trades["pnl"] > 0]
            losing = closed_trades[closed_trades["pnl"] <= 0]

            winning_trades = len(winning)
            losing_trades = len(losing)
            win_rate_pct = (winning_trades / total_trades) * 100

            # Profit factor
            total_profit = winning["pnl"].sum() if len(winning) > 0 else 0
            total_loss = abs(losing["pnl"].sum()) if len(losing) > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

            # Average profit/loss
            avg_profit_pct = winning["pnl_pct"].mean() if len(winning) > 0 else 0.0
            avg_loss_pct = losing["pnl_pct"].mean() if len(losing) > 0 else 0.0
            avg_trade_pct = closed_trades["pnl_pct"].mean()
        else:
            winning_trades = losing_trades = 0
            win_rate_pct = profit_factor = 0.0
            avg_profit_pct = avg_loss_pct = avg_trade_pct = 0.0
    else:
        total_trades = winning_trades = losing_trades = 0
        win_rate_pct = profit_factor = 0.0
        avg_profit_pct = avg_loss_pct = avg_trade_pct = 0.0

    return PerformanceMetrics(
        start_date=start_date,
        end_date=end_date,
        total_days=total_days,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        mdd_pct=mdd_pct,
        volatility_pct=volatility_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        avg_profit_pct=avg_profit_pct,
        avg_loss_pct=avg_loss_pct,
        avg_trade_pct=avg_trade_pct,
        equity_curve=equity_curve,
        drawdown_curve=drawdown * 100,  # Convert to percentage
        dates=dates,
        daily_returns=daily_returns,
    )


def calculate_monthly_returns(
    equity_curve: np.ndarray,
    dates: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate monthly returns for heatmap.

    Args:
        equity_curve: Array of equity values
        dates: Array of dates

    Returns:
        DataFrame with monthly returns (rows=years, cols=months)
    """
    # Create DataFrame with dates and equity
    df = pd.DataFrame({"date": dates, "equity": equity_curve})
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Resample to monthly, taking last value of each month
    monthly = df["equity"].resample("ME").last()

    # Calculate monthly returns
    monthly_returns = monthly.pct_change() * 100

    # Create pivot table (year x month)
    monthly_df = pd.DataFrame(
        {
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values,
        }
    )

    # Pivot to create heatmap data
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ][: len(pivot.columns)]

    return pivot


def calculate_yearly_returns(
    equity_curve: np.ndarray,
    dates: np.ndarray,
) -> pd.Series:
    """
    Calculate yearly returns.

    Args:
        equity_curve: Array of equity values
        dates: Array of dates

    Returns:
        Series with yearly returns
    """
    df = pd.DataFrame({"date": dates, "equity": equity_curve})
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Resample to yearly, taking last value
    yearly = df["equity"].resample("YE").last()
    yearly_returns = yearly.pct_change() * 100

    # First year: calculate from initial
    if len(yearly) > 0:
        first_year_return = (yearly.iloc[0] / equity_curve[0] - 1) * 100
        yearly_returns.iloc[0] = first_year_return

    yearly_returns.index = yearly_returns.index.year

    return yearly_returns


class BacktestReport:
    """
    Generates comprehensive backtest reports with visualizations.
    """

    def __init__(
        self,
        equity_curve: np.ndarray,
        dates: np.ndarray,
        trades: list | pd.DataFrame,
        strategy_name: str = "Strategy",
        initial_capital: float = 1.0,
    ) -> None:
        """
        Initialize report generator.

        Args:
            equity_curve: Array of equity values
            dates: Array of dates
            trades: List of Trade objects or DataFrame
            strategy_name: Name of the strategy
            initial_capital: Starting capital
        """
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital

        # Convert to numpy arrays if needed
        self.equity_curve = np.array(equity_curve)
        self.dates = np.array(dates)

        # Convert trades to DataFrame
        if isinstance(trades, pd.DataFrame):
            self.trades_df = trades
        elif trades:
            self.trades_df = pd.DataFrame(
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
        else:
            self.trades_df = pd.DataFrame()

        # Calculate metrics
        self.metrics = calculate_metrics(
            self.equity_curve, self.dates, self.trades_df, initial_capital
        )

    def print_summary(self) -> None:
        """Print performance summary to console using structured logging."""
        m = self.metrics

        # Use sys.stdout for console output (structured logging alternative)
        # This maintains the original console output behavior while avoiding print()
        output_lines = [
            f"\n{'=' * 60}",
            f"  BACKTEST REPORT: {self.strategy_name}",
            f"{'=' * 60}",
            "\n[Period]",
            f"   Start Date:     {m.start_date}",
            f"   End Date:       {m.end_date}",
            f"   Total Days:     {m.total_days:,}",
            "\n[Returns]",
            f"   Total Return:   {m.total_return_pct:,.2f}%",
            f"   CAGR:           {m.cagr_pct:.2f}%",
            "\n[Risk]",
            f"   Max Drawdown:   {m.mdd_pct:.2f}%",
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
        if self.risk_metrics:
            output_lines.extend([
                "\n[Portfolio Risk Metrics]",
                f"   VaR (95%):      {self.risk_metrics.var_95*100:.2f}%",
                f"   CVaR (95%):     {self.risk_metrics.cvar_95*100:.2f}%",
                f"   VaR (99%):      {self.risk_metrics.var_99*100:.2f}%",
                f"   CVaR (99%):    {self.risk_metrics.cvar_99*100:.2f}%",
                f"   Portfolio Vol:  {self.risk_metrics.portfolio_volatility*100:.2f}%",
            ])
            if self.risk_metrics.avg_correlation != 0.0:
                output_lines.extend([
                    f"   Avg Correlation: {self.risk_metrics.avg_correlation:.3f}",
                    f"   Max Correlation: {self.risk_metrics.max_correlation:.3f}",
                    f"   Min Correlation: {self.risk_metrics.min_correlation:.3f}",
                ])
            if self.risk_metrics.max_position_pct > 0:
                output_lines.extend([
                    f"   Max Position %:  {self.risk_metrics.max_position_pct*100:.2f}%",
                    f"   Position HHI:     {self.risk_metrics.position_concentration:.3f}",
                ])
            if self.risk_metrics.portfolio_beta is not None:
                output_lines.append(f"   Portfolio Beta:  {self.risk_metrics.portfolio_beta:.2f}")
        
        output_lines.append(f"\n{'=' * 60}\n")

        # Use logger.info for structured logging (outputs to console via logging handler)
        for line in output_lines:
            logger.info(line)

    def plot_equity_curve(
        self,
        ax: plt.Axes | None = None,
        show_drawdown: bool = True,
    ) -> plt.Figure | None:
        """
        Plot equity curve with optional drawdown.

        Args:
            ax: Matplotlib axes (creates new figure if None)
            show_drawdown: Whether to show drawdown on secondary axis

        Returns:
            Figure if ax was None, else None
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Plot equity curve
        ax.plot(self.dates, self.equity_curve, "b-", linewidth=1.5, label="Equity")
        ax.fill_between(
            self.dates,
            self.initial_capital,
            self.equity_curve,
            alpha=0.3,
            color="blue",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Equity", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax.set_title(f"{self.strategy_name} - Equity Curve")

        # Add drawdown on secondary axis
        if show_drawdown:
            ax2 = ax.twinx()
            ax2.fill_between(
                self.dates,
                0,
                -self.metrics.drawdown_curve,
                alpha=0.3,
                color="red",
                label="Drawdown",
            )
            ax2.set_ylabel("Drawdown (%)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(bottom=-self.metrics.mdd_pct * 1.2, top=5)

        ax.grid(True, alpha=0.3)

        if fig:
            plt.tight_layout()

        return fig

    def plot_drawdown(self, ax: plt.Axes | None = None) -> plt.Figure | None:
        """
        Plot drawdown curve.

        Args:
            ax: Matplotlib axes

        Returns:
            Figure if ax was None
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        ax.fill_between(
            self.dates,
            0,
            -self.metrics.drawdown_curve,
            color="red",
            alpha=0.5,
        )
        ax.plot(self.dates, -self.metrics.drawdown_curve, "r-", linewidth=0.5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(f"{self.strategy_name} - Drawdown")
        ax.grid(True, alpha=0.3)

        # Add MDD annotation
        mdd_idx = np.argmax(self.metrics.drawdown_curve)
        ax.annotate(
            f"MDD: {-self.metrics.mdd_pct:.1f}%",
            xy=(self.dates[mdd_idx], -self.metrics.mdd_pct),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            color="red",
        )

        if fig:
            plt.tight_layout()

        return fig

    def plot_monthly_heatmap(self, ax: plt.Axes | None = None) -> plt.Figure | None:
        """
        Plot monthly returns heatmap.

        Args:
            ax: Matplotlib axes

        Returns:
            Figure if ax was None
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))

        monthly = calculate_monthly_returns(self.equity_curve, self.dates)

        # Add yearly totals column
        yearly = calculate_yearly_returns(self.equity_curve, self.dates)
        monthly["Year"] = yearly

        # Create heatmap
        im = ax.imshow(monthly.values, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=20)

        # Set ticks
        ax.set_xticks(range(len(monthly.columns)))
        ax.set_xticklabels(monthly.columns)
        ax.set_yticks(range(len(monthly.index)))
        ax.set_yticklabels(monthly.index)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Return (%)")

        # Add value annotations
        for i in range(len(monthly.index)):
            for j in range(len(monthly.columns)):
                val = monthly.iloc[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 10 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

        ax.set_title(f"{self.strategy_name} - Monthly Returns Heatmap (%)")

        if fig:
            plt.tight_layout()

        return fig

    def plot_full_report(
        self,
        save_path: Path | str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Generate full visual report with all charts.

        Args:
            save_path: Path to save the figure
            show: Whether to display the figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 2], hspace=0.3, wspace=0.3)

        # Equity curve (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_equity_curve(ax=ax1, show_drawdown=True)

        # Drawdown (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_drawdown(ax=ax2)

        # Metrics table (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_metrics_table(ax=ax3)

        # Monthly heatmap (bottom, full width)
        ax4 = fig.add_subplot(gs[2, :])
        self.plot_monthly_heatmap(ax=ax4)

        fig.suptitle(
            f"Backtest Report: {self.strategy_name}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()  # pragma: no cover (interactive display, difficult to test)

        return fig

    def _plot_metrics_table(self, ax: plt.Axes) -> None:
        """Plot metrics as a table."""
        ax.axis("off")

        m = self.metrics
        data = [
            ["Total Return", f"{m.total_return_pct:,.2f}%"],
            ["CAGR", f"{m.cagr_pct:.2f}%"],
            ["Max Drawdown", f"{m.mdd_pct:.2f}%"],
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
        if self.risk_metrics:
            data.extend([
                ["", ""],  # Separator
                ["VaR (95%)", f"{self.risk_metrics.var_95*100:.2f}%"],
                ["CVaR (95%)", f"{self.risk_metrics.cvar_95*100:.2f}%"],
                ["Portfolio Vol", f"{self.risk_metrics.portfolio_volatility*100:.2f}%"],
            ])
            if self.risk_metrics.avg_correlation != 0.0:
                data.append(["Avg Correlation", f"{self.risk_metrics.avg_correlation:.3f}"])

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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export metrics as DataFrame.

        Returns:
            DataFrame with all metrics
        """
        m = self.metrics
        return pd.DataFrame(
            {
                "Metric": [
                    "Start Date",
                    "End Date",
                    "Total Days",
                    "Total Return (%)",
                    "CAGR (%)",
                    "Max Drawdown (%)",
                    "Volatility (%)",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Total Trades",
                    "Winning Trades",
                    "Losing Trades",
                    "Win Rate (%)",
                    "Profit Factor",
                    "Avg Profit (%)",
                    "Avg Loss (%)",
                    "Avg Trade (%)",
                ],
                "Value": [
                    str(m.start_date),
                    str(m.end_date),
                    m.total_days,
                    round(m.total_return_pct, 2),
                    round(m.cagr_pct, 2),
                    round(m.mdd_pct, 2),
                    round(m.volatility_pct, 2),
                    round(m.sharpe_ratio, 2),
                    round(m.sortino_ratio, 2),
                    round(m.calmar_ratio, 2),
                    m.total_trades,
                    m.winning_trades,
                    m.losing_trades,
                    round(m.win_rate_pct, 2),
                    round(m.profit_factor, 2),
                    round(m.avg_profit_pct, 2),
                    round(m.avg_loss_pct, 2),
                    round(m.avg_trade_pct, 2),
                ],
            }
        )


def generate_report(
    result,  # BacktestResult
    strategy_name: str | None = None,
    save_path: Path | str | None = None,
    show: bool = True,
    format: str = "png",  # "png" or "html"
    strategy_obj=None,  # Strategy instance for parameter extraction
    config=None,  # BacktestConfig instance
    tickers: list[str] | None = None,  # List of tickers used in backtest
) -> BacktestReport:
    """
    Convenience function to generate report from BacktestResult.

    Args:
        result: BacktestResult from backtest engine
        strategy_name: Override strategy name
        save_path: Path to save the figure/report
        show: Whether to display the figure (PNG only)
        format: Output format ("png" or "html")

    Returns:
        BacktestReport instance
    """
    report = BacktestReport(
        equity_curve=result.equity_curve,
        dates=result.dates,
        trades=result.trades,
        strategy_name=strategy_name or result.strategy_name,
        initial_capital=result.config.initial_capital if result.config else 1.0,
    )
    
    # Add risk metrics to report if available
    if hasattr(result, 'risk_metrics') and result.risk_metrics:
        report.risk_metrics = result.risk_metrics

    report.print_summary()

    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        
        # Determine format from file extension if not specified
        if format == "png" and save_path.suffix.lower() == ".html":
            format = "html"
        elif format == "html" and save_path.suffix.lower() == ".png":
            format = "png"
        elif save_path.suffix.lower() == ".html":
            format = "html"
        elif save_path.suffix.lower() == ".png":
            format = "png"
        
        if format == "html":
            from src.backtester.html_report import generate_html_report
            generate_html_report(report, save_path, strategy_obj=strategy_obj, config=config, tickers=tickers)
        else:
            report.plot_full_report(save_path=save_path, show=show)
    elif show:
        report.plot_full_report(save_path=None, show=show)

    return report
