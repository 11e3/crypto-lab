"""
Data loading utilities for vectorized backtesting.

Handles file I/O, caching, and data preparation.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.backtester.engine.data_loader_base import (
    apply_strategy_signals,
    load_parquet_data,
)
from src.data.cache.cache import get_cache
from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.utils.memory import optimize_dtypes

__all__ = ["load_parquet_data", "get_cache_params", "load_ticker_data"]

logger = get_logger(__name__)


def get_cache_params(strategy: Strategy) -> dict[str, Any]:
    """
    Extract cache parameters from strategy.

    Args:
        strategy: Trading strategy

    Returns:
        Dictionary of cache parameters
    """
    params: dict[str, Any] = {"strategy_name": strategy.name}

    # Extract VBO-specific parameters if available
    for attr in ["sma_period", "trend_sma_period", "short_noise_period", "long_noise_period"]:
        if hasattr(strategy, attr):
            params[attr] = getattr(strategy, attr)

    # Include entry/exit conditions in cache key
    if hasattr(strategy, "entry_conditions"):
        params["entry_conditions"] = [c.name for c in strategy.entry_conditions.conditions]
    if hasattr(strategy, "exit_conditions"):
        params["exit_conditions"] = [c.name for c in strategy.exit_conditions.conditions]

    return params


def load_ticker_data(
    ticker: str,
    filepath: Path,
    strategy: Strategy,
    cache_params: dict[str, Any],
    use_cache: bool = True,
    position_sizing: str = "equal",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load and prepare data for a single ticker.

    Args:
        ticker: Ticker symbol
        filepath: Path to data file
        strategy: Trading strategy
        cache_params: Cache parameters
        use_cache: Whether to use caching
        position_sizing: Position sizing method

    Returns:
        Tuple of (processed_df, historical_df or None)
    """
    interval = filepath.stem.split("_")[1] if "_" in filepath.stem else "unknown"
    raw_mtime = filepath.stat().st_mtime if filepath.exists() else None

    cache = get_cache() if use_cache else None
    cached_df = None
    historical_df: pd.DataFrame | None = None

    if cache is not None:
        cached_df = cache.get(ticker, interval, cache_params, raw_mtime)

    if cached_df is not None:
        df = cached_df
        logger.debug(f"Loaded {ticker} from cache")
    else:
        df = load_parquet_data(filepath)
        df = optimize_dtypes(df)

        if position_sizing != "equal":
            historical_df = df.copy()

        df = apply_strategy_signals(df, strategy)

        if cache is not None:
            cache.set(ticker, interval, cache_params, df, raw_mtime)
            logger.debug(f"Saved {ticker} to cache")

    df["ticker"] = ticker
    df = optimize_dtypes(df)

    if historical_df is None and position_sizing != "equal":
        historical_df = df.copy()

    return df, historical_df
