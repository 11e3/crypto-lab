"""
Shared data loading utilities for backtest engines.

Provides common functions used by both vectorized and event-driven loaders:
- load_parquet_data: Read and normalize parquet files
- apply_strategy_signals: Apply indicators and generate signals
- validate_required_columns: Validate required DataFrame columns
"""

from pathlib import Path

import pandas as pd

from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "load_parquet_data",
    "apply_strategy_signals",
    "validate_required_columns",
]


def load_parquet_data(filepath: Path) -> pd.DataFrame:
    """
    Load OHLCV data from parquet file.

    Args:
        filepath: Path to parquet file

    Returns:
        DataFrame with DatetimeIndex and lowercased columns

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is corrupted or invalid
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        df = pd.read_parquet(filepath)
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:  # parquet/pyarrow raises diverse error types
        raise ValueError(f"Error loading data from {filepath}: {e}") from e


def apply_strategy_signals(df: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
    """
    Apply strategy indicators and generate entry/exit signals.

    Args:
        df: Raw OHLCV DataFrame
        strategy: Trading strategy instance

    Returns:
        DataFrame with indicators and entry_signal/exit_signal columns
    """
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)
    return df


def validate_required_columns(df: pd.DataFrame, required: list[str], ticker: str = "") -> bool:
    """
    Validate that required columns are present in DataFrame.

    Args:
        df: DataFrame to validate
        required: List of required column names
        ticker: Ticker name for logging context

    Returns:
        True if all columns present, False otherwise
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        prefix = f"{ticker}: " if ticker else ""
        logger.error(f"{prefix}Missing columns {missing}")
        return False
    return True
