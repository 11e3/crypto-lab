"""
Strategy Comparison Example

This example demonstrates how to compare multiple strategies side-by-side.
It shows how to evaluate different strategy configurations and choose the best one.
"""

from pathlib import Path
from typing import Dict, List

from src.backtester import BacktestConfig, BacktestResult, run_backtest
from src.strategies.volatility_breakout import MinimalVBO, StrictVBO, VanillaVBO
from src.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def compare_strategies(
    results: Dict[str, BacktestResult],
) -> None:
    """Compare multiple strategy results."""
    print("\n" + "=" * 80)
    print("Strategy Comparison")
    print("=" * 80)

    # Create comparison table
    print("\n{:<20} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Strategy",
        "Total Return",
        "CAGR",
        "Sharpe",
        "Max DD",
        "Win Rate",
        "Trades",
    ))
    print("-" * 80)

    for name, result in results.items():
        m = result.metrics
        print("{:<20} {:>11.2f}% {:>9.2f}% {:>9.2f} {:>9.2f}% {:>9.2f}% {:>10}".format(
            name,
            m.total_return * 100,
            m.cagr * 100,
            m.sharpe_ratio,
            m.max_drawdown * 100,
            m.win_rate * 100,
            m.total_trades,
        ))

    # Find best strategy by different metrics
    print("\n" + "=" * 80)
    print("Best Strategy by Metric")
    print("=" * 80)

    # Best by Total Return
    best_return = max(results.items(), key=lambda x: x[1].metrics.total_return)
    print(f"Highest Total Return: {best_return[0]} ({best_return[1].metrics.total_return * 100:.2f}%)")

    # Best by Sharpe Ratio
    best_sharpe = max(results.items(), key=lambda x: x[1].metrics.sharpe_ratio)
    print(f"Best Sharpe Ratio:    {best_sharpe[0]} ({best_sharpe[1].metrics.sharpe_ratio:.2f})")

    # Best by Max Drawdown (lowest)
    best_dd = min(results.items(), key=lambda x: x[1].metrics.max_drawdown)
    print(f"Lowest Max Drawdown:  {best_dd[0]} ({best_dd[1].metrics.max_drawdown * 100:.2f}%)")

    # Best by Win Rate
    best_winrate = max(results.items(), key=lambda x: x[1].metrics.win_rate)
    print(f"Highest Win Rate:     {best_winrate[0]} ({best_winrate[1].metrics.win_rate * 100:.2f}%)")

    # Best by Calmar Ratio
    best_calmar = max(results.items(), key=lambda x: x[1].metrics.calmar_ratio)
    print(f"Best Calmar Ratio:    {best_calmar[0]} ({best_calmar[1].metrics.calmar_ratio:.2f})")


def main() -> None:
    """Run strategy comparison example."""
    print("=" * 80)
    print("Strategy Comparison Example")
    print("=" * 80)
    print()

    # Common configuration
    config = BacktestConfig(
        initial_capital=1_000_000.0,
        fee_rate=0.0005,
        slippage_rate=0.0005,
        max_slots=4,
        use_cache=True,
    )

    tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
    results: Dict[str, BacktestResult] = {}

    # Strategy 1: Minimal VBO (baseline)
    print("Running Minimal VBO strategy...")
    strategy1 = MinimalVBO()
    result1 = run_backtest(
        strategy=strategy1,
        tickers=tickers,
        interval="day",
        config=config,
    )
    results["Minimal VBO"] = result1
    print("✓ Completed")

    # Strategy 2: Vanilla VBO (standard)
    print("\nRunning Vanilla VBO strategy...")
    strategy2 = VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        short_noise_period=4,
        long_noise_period=8,
    )
    result2 = run_backtest(
        strategy=strategy2,
        tickers=tickers,
        interval="day",
        config=config,
    )
    results["Vanilla VBO"] = result2
    print("✓ Completed")

    # Strategy 3: Strict VBO (conservative)
    print("\nRunning Strict VBO strategy...")
    strategy3 = StrictVBO()
    result3 = run_backtest(
        strategy=strategy3,
        tickers=tickers,
        interval="day",
        config=config,
    )
    results["Strict VBO"] = result3
    print("✓ Completed")

    # Compare strategies
    print("\n" + "=" * 80)
    print("Comparing strategies...")
    compare_strategies(results)

    # Recommendations
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)
    print("""
1. **For Maximum Returns**: Choose the strategy with highest Total Return
   (but consider the risk!)

2. **For Risk-Adjusted Returns**: Choose the strategy with best Sharpe Ratio
   (balances return and risk)

3. **For Low Risk**: Choose the strategy with lowest Max Drawdown
   (preserves capital during downturns)

4. **For Consistency**: Choose the strategy with highest Win Rate
   (more frequent wins, but check profit factor)

5. **For Long-term Growth**: Choose the strategy with best Calmar Ratio
   (CAGR relative to max drawdown)
    """)

    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
