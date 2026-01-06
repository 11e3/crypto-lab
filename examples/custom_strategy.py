"""
Custom Strategy Example

This example demonstrates how to create and test a custom trading strategy.
It shows how to combine different conditions and filters to build a custom strategy.
"""

from pathlib import Path

from src.backtester import BacktestConfig, generate_report, run_backtest
from src.strategies.base import Strategy
from src.strategies.volatility_breakout.conditions import (
    BreakoutCondition,
    NoiseCondition,
    SMABreakoutCondition,
    TrendCondition,
    WhipsawExitCondition,
)
from src.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


class CustomVBOStrategy(Strategy):
    """
    Custom VBO strategy with modified parameters.

    This strategy uses:
    - Longer SMA period for trend detection (10 instead of 8)
    - Stricter noise threshold (0.3 instead of default)
    - Custom breakout condition
    """

    def __init__(
        self,
        sma_period: int = 5,
        trend_sma_period: int = 10,
        short_noise_period: int = 5,
        long_noise_period: int = 10,
        noise_threshold: float = 0.3,
    ) -> None:
        """Initialize custom VBO strategy."""
        super().__init__(name="CustomVBO")

        # Entry conditions
        entry_conditions = [
            SMABreakoutCondition(sma_period=sma_period),
            TrendCondition(trend_sma_period=trend_sma_period),
            NoiseCondition(
                short_period=short_noise_period,
                long_period=long_noise_period,
                threshold=noise_threshold,
            ),
        ]

        # Exit conditions
        exit_conditions = [
            BreakoutCondition(),
            WhipsawExitCondition(),
        ]

        self.entry_conditions.add_conditions(entry_conditions)
        self.exit_conditions.add_conditions(exit_conditions)


def main() -> None:
    """Run custom strategy example."""
    print("=" * 60)
    print("Custom Strategy Example")
    print("=" * 60)
    print()

    # Step 1: Create custom strategy
    print("Step 1: Creating custom strategy...")
    strategy = CustomVBOStrategy(
        sma_period=5,
        trend_sma_period=10,
        short_noise_period=5,
        long_noise_period=10,
        noise_threshold=0.3,
    )
    print(f"✓ Strategy created: {strategy.name}")
    print(f"  Entry conditions: {[c.name for c in strategy.entry_conditions.conditions]}")
    print(f"  Exit conditions: {[c.name for c in strategy.exit_conditions.conditions]}")
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
    print("Step 3: Running backtest with custom strategy...")
    tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]

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

    report_path = reports_dir / "custom_strategy_report.html"
    generate_report(
        result,
        save_path=report_path,
        show=False,
    )
    print(f"✓ Report saved to: {report_path}")
    print()

    # Step 6: Compare with default
    print("Step 6: Strategy Comparison Tips")
    print("-" * 60)
    print("To compare with default strategy:")
    print("1. Run basic_backtest.py with same tickers")
    print("2. Compare metrics in both reports")
    print("3. Adjust parameters based on results")
    print()

    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
