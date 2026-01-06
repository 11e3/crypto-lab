"""
Performance Benchmark Example

This example demonstrates how to benchmark backtesting performance.
It measures execution time, memory usage, and scalability.
"""

import time
from pathlib import Path
from typing import List

from src.backtester import BacktestConfig, run_backtest
from src.strategies.volatility_breakout import VanillaVBO
from src.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def benchmark_backtest(
    tickers: List[str],
    interval: str = "day",
    runs: int = 3,
) -> None:
    """Benchmark backtest performance."""
    print("=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    print()

    strategy = VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        short_noise_period=4,
        long_noise_period=8,
    )

    config = BacktestConfig(
        initial_capital=1_000_000.0,
        fee_rate=0.0005,
        slippage_rate=0.0005,
        max_slots=4,
        use_cache=True,
    )

    print(f"Configuration:")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Interval: {interval}")
    print(f"  Runs: {runs}")
    print()

    execution_times: List[float] = []

    for i in range(runs):
        print(f"Run {i + 1}/{runs}...", end=" ", flush=True)

        start_time = time.time()
        result = run_backtest(
            strategy=strategy,
            tickers=tickers,
            interval=interval,
            config=config,
        )
        end_time = time.time()

        execution_time = end_time - start_time
        execution_times.append(execution_time)

        print(f"✓ ({execution_time:.2f}s)")

    # Calculate statistics
    avg_time = sum(execution_times) / len(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)

    print()
    print("=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Average execution time: {avg_time:.2f}s")
    print(f"Minimum execution time: {min_time:.2f}s")
    print(f"Maximum execution time: {max_time:.2f}s")
    print(f"Runs: {runs}")
    print()

    # Performance per day
    if result.trades:
        # Estimate days from trades
        first_trade = min(result.trades, key=lambda t: t.entry_time)
        last_trade = max(result.trades, key=lambda t: t.exit_time)
        days = (last_trade.exit_time - first_trade.entry_time).days
        if days > 0:
            time_per_day = avg_time / days * 1000  # Convert to ms
            print(f"Estimated time per day: {time_per_day:.2f}ms")
            print()

    # Scalability test
    print("=" * 60)
    print("Scalability Test")
    print("=" * 60)
    print("Testing with different numbers of tickers...")
    print()

    ticker_counts = [1, 2, 4, 8]
    scalability_results: List[tuple[int, float]] = []

    for count in ticker_counts:
        test_tickers = tickers[:count]
        print(f"Testing with {count} ticker(s): {', '.join(test_tickers)}...", end=" ", flush=True)

        start_time = time.time()
        run_backtest(
            strategy=strategy,
            tickers=test_tickers,
            interval=interval,
            config=config,
        )
        end_time = time.time()

        execution_time = end_time - start_time
        scalability_results.append((count, execution_time))
        print(f"✓ ({execution_time:.2f}s)")

    print()
    print("Scalability Results:")
    print("-" * 60)
    print(f"{'Tickers':<10} {'Time (s)':<12} {'Time/Ticker (s)':<15}")
    print("-" * 60)
    for count, exec_time in scalability_results:
        time_per_ticker = exec_time / count
        print(f"{count:<10} {exec_time:<12.2f} {time_per_ticker:<15.2f}")

    print()
    print("=" * 60)
    print("Benchmark completed!")
    print("=" * 60)


def main() -> None:
    """Run performance benchmark."""
    # Test with different ticker sets
    ticker_sets = [
        (["KRW-BTC"], "Single ticker"),
        (["KRW-BTC", "KRW-ETH"], "Two tickers"),
        (["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-TRX"], "Four tickers"),
    ]

    for tickers, description in ticker_sets:
        print("\n" + "=" * 60)
        print(description)
        print("=" * 60)
        benchmark_backtest(tickers=tickers, interval="day", runs=3)
        print()


if __name__ == "__main__":
    main()
