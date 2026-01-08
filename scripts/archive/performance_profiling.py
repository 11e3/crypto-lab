#!/usr/bin/env python
"""
Performance Profiling Script for Crypto Quant System.

Analyzes:
1. Pandas operation performance
2. Numpy vectorization opportunities
3. Memory optimization recommendations
4. Groupby aggregation performance

Usage:
    python scripts/performance_profiling.py
"""

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def timer(name: str):
    """Context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"  [TIMER] {name}: {elapsed:.4f}s")


def analyze_pandas_operations():
    """Analyze common Pandas operations performance."""
    print("\n" + "=" * 80)
    print("PANDAS OPERATIONS PERFORMANCE ANALYSIS")
    print("=" * 80)

    n_rows = 10000
    n_cols = 10

    print(f"\nTest DataFrame: {n_rows:,} rows x {n_cols} columns")
    df = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        columns=[f"col_{i}" for i in range(n_cols)],
        index=pd.date_range(start="2023-01-01", periods=n_rows, freq="D"),
    )
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n")

    # Operations
    print("Operation Timings:")
    print("-" * 80)

    with timer("Rolling mean (window=20)"):
        df.rolling(20).mean()

    with timer("Correlation matrix"):
        df.corr()

    with timer("Sorting by first column"):
        df.sort_values("col_0")

    with timer("Column-wise sum"):
        df.sum()


def analyze_vectorization():
    """Identify vectorization opportunities."""
    print("\n" + "=" * 80)
    print("VECTORIZATION OPPORTUNITIES")
    print("=" * 80)

    prices = np.random.randn(10000).cumsum() + 100

    print(f"\nTest data: {len(prices):,} price points\n")

    # Test 1: Returns calculation
    print("1. Returns Calculation")
    print("-" * 80)
    with timer("Vectorized numpy.diff()"):
        returns = np.diff(prices) / prices[:-1]

    print("Optimization: Use numpy.diff for vectorized operations")
    print(f"Result shape: {returns.shape}\n")

    # Test 2: Cumulative maximum
    print("2. Cumulative Maximum (for Drawdown)")
    print("-" * 80)
    with timer("Vectorized numpy.maximum.accumulate()"):
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax

    print("Optimization: Use accumulate for running statistics")
    print(f"Max Drawdown: {drawdown.min():.2%}\n")

    # Test 3: Rolling operations
    print("3. Rolling Average (window=20)")
    print("-" * 80)
    window = 20
    with timer("Numpy convolve"):
        rolling_avg = np.convolve(prices, np.ones(window) / window, mode="valid")

    print("Optimization: Use numpy.convolve for rolling operations")
    print(f"Result length: {len(rolling_avg)}\n")


def analyze_memory_optimization():
    """Analyze memory optimization opportunities."""
    print("\n" + "=" * 80)
    print("MEMORY OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)

    n_rows = 50000

    print(f"\nTest DataFrame: {n_rows:,} rows\n")

    # Create test data
    df = pd.DataFrame(
        {
            "ticker": np.random.choice(["BTC", "ETH", "XRP", "DOT"], n_rows),
            "price": np.random.randn(n_rows).cumsum() + 100,
            "volume": np.random.randint(1000000, 10000000, n_rows),
            "open": np.random.randn(n_rows).cumsum() + 100,
            "high": np.random.randn(n_rows).cumsum() + 105,
            "low": np.random.randn(n_rows).cumsum() + 95,
        }
    )

    original_memory = df.memory_usage(deep=True).sum() / 1e6
    print(f"Original memory usage: {original_memory:.2f} MB")
    print(f"Dtypes:\n{df.dtypes}\n")

    # Optimize
    df_opt = df.copy()
    df_opt["ticker"] = df_opt["ticker"].astype("category")
    df_opt["volume"] = df_opt["volume"].astype("int32")
    df_opt["open"] = df_opt["open"].astype("float32")
    df_opt["high"] = df_opt["high"].astype("float32")
    df_opt["low"] = df_opt["low"].astype("float32")

    opt_memory = df_opt.memory_usage(deep=True).sum() / 1e6
    savings = (1 - opt_memory / original_memory) * 100

    print(f"After optimization: {opt_memory:.2f} MB")
    print(f"Memory savings: {savings:.1f}%")
    print(f"\nOptimized dtypes:\n{df_opt.dtypes}")


def analyze_groupby_operations():
    """Analyze groupby aggregation performance."""
    print("\n" + "=" * 80)
    print("GROUPBY AGGREGATION ANALYSIS")
    print("=" * 80)

    n_rows = 50000
    tickers = np.random.choice(["BTC", "ETH", "XRP", "DOT"], n_rows)

    df = pd.DataFrame(
        {
            "ticker": tickers,
            "close": np.random.randn(n_rows) + 100,
            "volume": np.random.randint(1000000, 10000000, n_rows),
        }
    )

    print(f"\nTest DataFrame: {n_rows:,} rows, {len(df['ticker'].unique())} tickers\n")

    print("Groupby Operations:")
    print("-" * 80)

    with timer("Multiple aggregations"):
        result = df.groupby("ticker").agg(
            {
                "close": ["mean", "std", "min", "max"],
                "volume": ["sum", "mean"],
            }
        )

    print(f"Result shape: {result.shape}")
    print("Optimization: Cache grouped data if reusing, avoid repeated groupby\n")

    with timer("Transform (forward fill missing dates)"):
        result = df.groupby("ticker")["close"].transform("mean")

    print("Optimization: Use groupby.transform for aligned results")


def show_recommendations():
    """Display optimization recommendations."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)

    recommendations = [
        (
            "Use category dtype for tickers",
            "Saves 70-90% memory compared to string objects",
            "Example: df['ticker'] = df['ticker'].astype('category')",
        ),
        (
            "Use float32 for OHLCV prices",
            "Halves memory for price columns with minimal precision loss",
            "Example: df['close'] = df['close'].astype('float32')",
        ),
        (
            "Use int32 for volume",
            "Saves 50% memory compared to int64",
            "Example: df['volume'] = df['volume'].astype('int32')",
        ),
        (
            "Vectorize rolling window calculations",
            "Use numpy.convolve or stride tricks instead of Pandas rolling",
            "Performance: 5-10x faster for large windows",
        ),
        (
            "Cache technical indicators",
            "Avoid recalculating SMA, RSI, etc. across backtests",
            "Optimization: Implement indicator cache with versioning",
        ),
        (
            "Use numpy for signal generation",
            "Vectorize boolean conditions instead of iterating",
            "Example: signals = prices[condition] instead of loop",
        ),
        (
            "Parallelize portfolio evaluation",
            "Use multiprocessing for independent strategy backtests",
            "Tool: concurrent.futures.ProcessPoolExecutor",
        ),
        (
            "Consider Polars for data loading",
            "5-10x faster CSV/Parquet I/O than Pandas",
            "Migration: Polars API is similar to Pandas",
        ),
    ]

    for i, (title, benefit, example) in enumerate(recommendations, 1):
        print(f"\n[{i}] {title}")
        print(f"    Benefit: {benefit}")
        print(f"    {example}")


def main():
    """Run all profiling analyses."""
    print("\n" + "=" * 80)
    print("CRYPTO QUANT SYSTEM - PERFORMANCE PROFILING")
    print("=" * 80)

    try:
        analyze_pandas_operations()
        analyze_vectorization()
        analyze_memory_optimization()
        analyze_groupby_operations()
        show_recommendations()

        print("\n" + "=" * 80)
        print("PROFILING COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Profile production backtest code with cProfile")
        print("2. Apply top 3-5 optimizations from recommendations")
        print("3. Benchmark improvements with realistic data volumes")
        print("4. Consider Polars migration if I/O is bottleneck")
        print("5. Use Numba JIT if signal generation loops are slow")

    except Exception as e:
        logger.error(f"Profiling error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
