"""
OHLCV data update utilities for UpbitDataSource.
"""

from datetime import datetime

import pandas as pd

from src.utils.logger import get_logger

__all__ = ["calculate_update_count", "merge_ohlcv_data"]

logger = get_logger(__name__)


def _make_naive(ts: datetime) -> datetime:
    """Strip timezone info for consistent comparison."""
    if ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


def calculate_update_count(
    latest_timestamp: datetime,
    interval: str,
) -> int:
    """
    Calculate number of candles to fetch for incremental update.

    Args:
        latest_timestamp: Latest timestamp in existing data
        interval: Data interval (e.g., 'day', 'minute240')

    Returns:
        Number of candles to fetch
    """
    now = datetime.now()
    latest = _make_naive(latest_timestamp)

    if interval == "day":
        days_since = (now - latest).days
        return min(days_since + 10, 200)
    elif interval.startswith("minute"):
        try:
            minutes = int(interval.replace("minute", ""))
            minutes_since = (now - latest).total_seconds() / 60
            return min(int(minutes_since / minutes) + 10, 200)
        except ValueError:
            return 200
    else:
        return 200


def merge_ohlcv_data(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    latest_timestamp: datetime,
) -> tuple[pd.DataFrame, int]:
    """
    Merge existing and new OHLCV data.

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
