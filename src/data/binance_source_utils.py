"""OHLCV data update utilities for BinanceDataSource."""

from datetime import datetime

import pandas as pd

from src.utils.logger import get_logger

__all__ = ["calculate_binance_update_count", "merge_ohlcv_data"]

logger = get_logger(__name__)

# Binance interval to minutes mapping
_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
    "1M": 43200,
}


def _make_naive(ts: datetime) -> datetime:
    """Strip timezone info for consistent comparison."""
    if ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


def calculate_binance_update_count(
    latest_timestamp: datetime,
    interval: str,
) -> int:
    """Calculate number of candles to fetch for incremental update.

    Args:
        latest_timestamp: Latest timestamp in existing data
        interval: Binance interval (e.g., '1d', '4h', '1m')

    Returns:
        Number of candles to fetch (capped at 1000)
    """
    now = datetime.now()
    latest = _make_naive(latest_timestamp)

    minutes_per_candle = _INTERVAL_MINUTES.get(interval)
    if minutes_per_candle is None:
        return 1000

    minutes_since = (now - latest).total_seconds() / 60
    return min(int(minutes_since / minutes_per_candle) + 10, 1000)


def merge_ohlcv_data(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    latest_timestamp: datetime,
) -> tuple[pd.DataFrame, int]:
    """Merge existing and new OHLCV data.

    Args:
        existing_df: Existing DataFrame
        new_df: New DataFrame to merge
        latest_timestamp: Latest timestamp to filter new data

    Returns:
        Tuple of (merged DataFrame, number of new rows added)
    """
    # Filter to new data (>= to avoid dropping exact timestamp matches)
    new_df = new_df[new_df.index >= latest_timestamp]

    if len(new_df) == 0:
        return existing_df, 0

    # Merge with existing data
    updated_df = pd.concat([existing_df, new_df])
    updated_df = updated_df[~updated_df.index.duplicated(keep="last")]
    updated_df = updated_df.sort_index()

    return updated_df, len(new_df)
