"""
Performance Analysis Example

This example demonstrates how to analyze and compare strategy performance.
It shows various performance metrics and how to interpret them.
"""

from pathlib import Path

from src.backtester import BacktestConfig, run_backtest
from src.backtester.report import PerformanceMetrics
from src.strategies.volatility_breakout import VanillaVBO
from src.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def analyze_metrics(metrics: PerformanceMetrics) -> None:
    """Analyze and display performance metrics with interpretation."""
    print("\n" + "=" * 60)
    print("Performance Analysis")
    print("=" * 60)

    # Return Metrics
    print("\n📈 Return Metrics")
    print("-" * 60)
    print(f"Total Return:     {metrics.total_return * 100:,.2f}%")
    print(f"CAGR:             {metrics.cagr * 100:.2f}%")
    print(f"Annual Return:    {metrics.annual_return * 100:.2f}%")

    # Risk Metrics
    print("\n⚠️  Risk Metrics")
    print("-" * 60)
    print(f"Max Drawdown:     {metrics.max_drawdown * 100:.2f}%")
    print(f"Volatility:       {metrics.volatility * 100:.2f}%")
    print(f"Downside Dev:     {metrics.downside_deviation * 100:.2f}%")

    # Risk-Adjusted Returns
    print("\n📊 Risk-Adjusted Returns")
    print("-" * 60)
    print(f"Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:    {metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio:     {metrics.calmar_ratio:.2f}")

    # Trading Statistics
    print("\n📉 Trading Statistics")
    print("-" * 60)
    print(f"Total Trades:      {metrics.total_trades}")
    print(f"Win Rate:          {metrics.win_rate * 100:.2f}%")
    print(f"Loss Rate:        {(1 - metrics.win_rate) * 100:.2f}%")
    print(f"Profit Factor:    {metrics.profit_factor:.2f}")
    print(f"Avg Win:          {metrics.avg_win:,.0f} KRW")
    print(f"Avg Loss:         {metrics.avg_loss:,.0f} KRW")

    # Interpretation
    print("\n💡 Interpretation")
    print("-" * 60)

    # Sharpe Ratio interpretation
    if metrics.sharpe_ratio > 2:
        sharpe_interpretation = "Excellent (Very good risk-adjusted returns)"
    elif metrics.sharpe_ratio > 1:
        sharpe_interpretation = "Good (Acceptable risk-adjusted returns)"
    elif metrics.sharpe_ratio > 0:
        sharpe_interpretation = "Fair (Positive but low risk-adjusted returns)"
    else:
        sharpe_interpretation = "Poor (Negative risk-adjusted returns)"

    print(f"Sharpe Ratio: {sharpe_interpretation}")

    # Win Rate interpretation
    if metrics.win_rate > 0.5:
        win_rate_interpretation = "Good (More wins than losses)"
    elif metrics.win_rate > 0.4:
        win_rate_interpretation = "Fair (Acceptable win rate)"
    else:
        win_rate_interpretation = "Low (Many losses, but may be offset by large wins)"

    print(f"Win Rate: {win_rate_interpretation}")

    # Profit Factor interpretation
    if metrics.profit_factor > 2:
        pf_interpretation = "Excellent (Very profitable)"
    elif metrics.profit_factor > 1.5:
        pf_interpretation = "Good (Profitable)"
    elif metrics.profit_factor > 1:
        pf_interpretation = "Fair (Slightly profitable)"
    else:
        pf_interpretation = "Poor (Not profitable)"

    print(f"Profit Factor: {pf_interpretation}")

    # Max Drawdown interpretation
    if metrics.max_drawdown < 0.1:
        mdd_interpretation = "Excellent (Very low drawdown)"
    elif metrics.max_drawdown < 0.2:
        mdd_interpretation = "Good (Acceptable drawdown)"
    elif metrics.max_drawdown < 0.3:
        mdd_interpretation = "Fair (Moderate drawdown)"
    else:
        mdd_interpretation = "High (Significant drawdown risk)"

    print(f"Max Drawdown: {mdd_interpretation}")


def main() -> None:
    """Run performance analysis example."""
    print("=" * 60)
    print("Performance Analysis Example")
    print("=" * 60)
    print()

    # Step 1: Create strategy
    print("Step 1: Creating strategy...")
    strategy = VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        short_noise_period=4,
        long_noise_period=8,
    )
    print("✓ Strategy created")
    print()

    # Step 2: Configure backtest
    print("Step 2: Configuring backtest...")
    config = BacktestConfig(
        initial_capital=1_000_000.0,
        fee_rate=0.0005,
        slippage_rate=0.0005,
        max_slots=4,
        use_cache=True,
    )
    print("✓ Configuration ready")
    print()

    # Step 3: Run backtest
    print("Step 3: Running backtest...")
    print("  This may take a few moments...")
    tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-TRX"]

    result = run_backtest(
        strategy=strategy,
        tickers=tickers,
        interval="day",
        config=config,
    )
    print("✓ Backtest completed!")
    print()

    # Step 4: Analyze metrics
    print("Step 4: Analyzing performance metrics...")
    analyze_metrics(result.metrics)

    # Step 5: Additional analysis
    print("\n" + "=" * 60)
    print("Additional Analysis")
    print("=" * 60)

    # Trade distribution
    if result.trades:
        winning_trades = [t for t in result.trades if t.profit > 0]
        losing_trades = [t for t in result.trades if t.profit <= 0]

        print(f"\nTrade Distribution:")
        print(f"  Winning trades: {len(winning_trades)}")
        print(f"  Losing trades:  {len(losing_trades)}")

        if winning_trades:
            max_win = max(t.profit for t in winning_trades)
            print(f"  Largest win:    {max_win:,.0f} KRW")

        if losing_trades:
            max_loss = min(t.profit for t in losing_trades)
            print(f"  Largest loss:   {max_loss:,.0f} KRW")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
