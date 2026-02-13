"""
Cache get/set operations for IndicatorCache.
"""

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cache.cache_eviction import enforce_cache_limits
from src.data.cache.cache_metadata import generate_cache_key, get_cache_path
from src.utils.logger import get_logger

__all__ = ["cache_get", "cache_set"]

logger = get_logger(__name__)


def cache_get(
    cache_dir: Path,
    metadata: dict[str, dict[str, Any]],
    access_times: OrderedDict[str, float],
    ticker: str,
    interval: str,
    params: dict[str, Any],
    raw_data_mtime: float | None = None,
) -> pd.DataFrame | None:
    """
    Retrieve cached indicators if valid.

    Args:
        cache_dir: Cache directory
        metadata: Cache metadata
        access_times: Access times for LRU
        ticker: Asset ticker
        interval: Data interval
        params: Indicator parameters
        raw_data_mtime: Modification time of raw data file

    Returns:
        Cached DataFrame or None if cache miss
    """
    cache_key = generate_cache_key(ticker, interval, params)
    cache_path = get_cache_path(cache_dir, cache_key)

    if not cache_path.exists():
        logger.debug(f"Cache miss (file not found): {cache_key}")
        return None

    if cache_key not in metadata:
        logger.debug(f"Cache miss (no metadata): {cache_key}")
        return None

    meta = metadata[cache_key]

    if meta.get("params") != params:
        logger.debug(f"Cache miss (params changed): {cache_key}")
        return None

    if raw_data_mtime is not None:
        cached_mtime = meta.get("raw_data_mtime", 0)
        if raw_data_mtime > cached_mtime:
            logger.debug(f"Cache miss (raw data updated): {cache_key}")
            return None

    current_time = time.time()
    if cache_key in access_times:
        del access_times[cache_key]
    access_times[cache_key] = current_time

    try:
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        logger.debug(f"Cache hit: {cache_key}")
        return df
    except Exception as e:
        logger.warning(f"Corrupted cache {cache_key}, removing: {e}")
        cache_path.unlink(missing_ok=True)
        metadata.pop(cache_key, None)
        access_times.pop(cache_key, None)
        return None


def cache_set(
    cache_dir: Path,
    metadata: dict[str, dict[str, Any]],
    access_times: OrderedDict[str, float],
    ticker: str,
    interval: str,
    params: dict[str, Any],
    df: pd.DataFrame,
    raw_data_mtime: float | None,
    use_compression: bool,
    max_entries: int,
    max_size_mb: float,
    ttl_days: int,
) -> None:
    """
    Store calculated indicators in cache.

    Args:
        cache_dir: Cache directory
        metadata: Cache metadata
        access_times: Access times for LRU
        ticker: Asset ticker
        interval: Data interval
        params: Indicator parameters
        df: DataFrame with indicators
        raw_data_mtime: Modification time of raw data file
        use_compression: Use compression for parquet
        max_entries: Max cache entries
        max_size_mb: Max cache size in MB
        ttl_days: TTL in days
    """
    cache_key = generate_cache_key(ticker, interval, params)
    cache_path = get_cache_path(cache_dir, cache_key)

    if use_compression:
        df.to_parquet(cache_path, engine="pyarrow", compression="snappy")
    else:
        df.to_parquet(cache_path, engine="pyarrow", compression=None)

    current_time = time.time()
    metadata[cache_key] = {
        "ticker": ticker,
        "interval": interval,
        "params": params,
        "raw_data_mtime": raw_data_mtime or 0,
        "rows": len(df),
        "created_at": current_time,
    }

    access_times[cache_key] = current_time

    enforce_cache_limits(
        metadata,
        access_times,
        cache_dir,
        max_entries,
        max_size_mb,
        ttl_days,
    )

    logger.debug(f"Cache saved: {cache_key} ({len(df)} rows)")
