"""
Basic Backtest Example

This example demonstrates how to run a simple backtest with default settings.
It shows the basic workflow: strategy creation, configuration, execution, and results.
"""

from pathlib import Path

from src.backtester import BacktestConfig, generate_report, run_backtest
from src.strategies.volatility_breakout import VanillaVBO
from src.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main() -> None:
    """Run a basic backtest example."""
    print("=" * 60)
    print("Basic Backtest Example")
    print("=" * 60)
    print()

    # Step 1: Create a strategy
    print("Step 1: Creating strategy...")
    strategy = VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        short_noise_period=4,
        long_noise_period=8,
    )
    print(f"✓ Strategy created: {strategy.name}")
    print(f"  Entry conditions: {len(strategy.entry_conditions.conditions)}")
    print(f"  Exit conditions: {len(strategy.exit_conditions.conditions)}")
    print()

    # Step 2: Configure backtest
    print("Step 2: Configuring backtest...")
    config = BacktestConfig(
        initial_capital=1_000_000.0,  # 1 million KRW
        fee_rate=0.0005,  # 0.05% trading fee
        slippage_rate=0.0005,  # 0.05% slippage
        max_slots=4,  # Maximum 4 concurrent positions
        use_cache=True,  # Use cached data if available
    )
    print(f"✓ Initial capital: {config.initial_capital:,.0f} KRW")
    print(f"✓ Fee rate: {config.fee_rate * 100:.2f}%")
    print(f"✓ Max positions: {config.max_slots}")
    print()

    # Step 3: Run backtest
    print("Step 3: Running backtest...")
    print("  This may take a few moments...")
    tickers = ["KRW-BTC", "KRW-ETH"]

    result = run_backtest(
        strategy=strategy,
        tickers=tickers,
        interval="day",
        config=config,
    )
    print("✓ Backtest completed!")
    print()

    # Step 4: Display results
    print("Step 4: Results Summary")
    print("-" * 60)
    print(result.summary())
    print()

    # Step 5: Generate report
    print("Step 5: Generating HTML report...")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    report_path = reports_dir / "basic_backtest_report.html"
    generate_report(
        result,
        save_path=report_path,
        show=False,
    )
    print(f"✓ Report saved to: {report_path}")
    print()

    # Step 6: Key metrics
    print("Step 6: Key Performance Metrics")
    print("-" * 60)
    metrics = result.metrics
    print(f"Total Return:     {metrics.total_return * 100:,.2f}%")
    print(f"CAGR:             {metrics.cagr * 100:.2f}%")
    print(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {metrics.max_drawdown * 100:.2f}%")
    print(f"Win Rate:         {metrics.win_rate * 100:.2f}%")
    print(f"Total Trades:     {metrics.total_trades}")
    print(f"Profit Factor:    {metrics.profit_factor:.2f}")
    print()

    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
