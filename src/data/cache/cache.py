"""
Indicator cache system for backtesting optimization.

Caches calculated indicators to data/processed/ to avoid
recalculating on every backtest run.

Improvements:
- LRU eviction policy
- Cache size limits
- Automatic cleanup of old entries
- Compression support
"""

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CACHE_METADATA_FILENAME, PROCESSED_DATA_DIR
from src.data.cache.cache_metadata import load_metadata, save_metadata
from src.data.cache.cache_ops import cache_get, cache_set
from src.data.cache.cache_stats import cleanup_cache, clear_cache, get_cache_stats, invalidate_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default cache limits
DEFAULT_MAX_CACHE_SIZE_MB = 1024  # 1GB default
DEFAULT_MAX_CACHE_ENTRIES = 1000
DEFAULT_CACHE_TTL_DAYS = 30  # 30 days TTL


class IndicatorCache:
    """
    Manages caching of indicator calculations.

    Stores processed DataFrames with indicators in parquet format,
    with metadata to track calculation parameters.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_size_mb: float = DEFAULT_MAX_CACHE_SIZE_MB,
        max_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        ttl_days: int = DEFAULT_CACHE_TTL_DAYS,
        use_compression: bool = True,
    ) -> None:
        """
        Initialize the indicator cache.

        Args:
            cache_dir: Directory for cached files. Defaults to data/processed/
            max_size_mb: Maximum cache size in MB (default: 1024)
            max_entries: Maximum number of cache entries (default: 1000)
            ttl_days: Time-to-live for cache entries in days (default: 30)
            use_compression: Use compression for parquet files (default: True)
        """
        self.cache_dir = cache_dir or PROCESSED_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / CACHE_METADATA_FILENAME
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self.ttl_days = ttl_days
        self.use_compression = use_compression
        # Track access times for LRU eviction
        self._access_times: OrderedDict[str, float] = OrderedDict()
        self._metadata = load_metadata(self.metadata_file, self._access_times)

    def _save_metadata(self) -> None:
        """Save cache metadata to JSON file."""
        save_metadata(self._metadata, self._access_times, self.metadata_file)

    def get(
        self,
        ticker: str,
        interval: str,
        params: dict[str, Any],
        raw_data_mtime: float | None = None,
    ) -> pd.DataFrame | None:
        """Retrieve cached indicators if valid."""
        return cache_get(
            self.cache_dir,
            self._metadata,
            self._access_times,
            ticker,
            interval,
            params,
            raw_data_mtime,
        )

    def set(
        self,
        ticker: str,
        interval: str,
        params: dict[str, Any],
        df: pd.DataFrame,
        raw_data_mtime: float | None = None,
    ) -> None:
        """Store calculated indicators in cache."""
        cache_set(
            self.cache_dir,
            self._metadata,
            self._access_times,
            ticker,
            interval,
            params,
            df,
            raw_data_mtime,
            self.use_compression,
            self.max_entries,
            self.max_size_mb,
            self.ttl_days,
        )
        self._save_metadata()

    def invalidate(
        self,
        ticker: str | None = None,
        interval: str | None = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            ticker: Invalidate only this ticker (or all if None)
            interval: Invalidate only this interval (or all if None)

        Returns:
            Number of entries invalidated
        """
        return invalidate_cache(
            self._metadata,
            self.cache_dir,
            ticker,
            interval,
            self._save_metadata,
        )

    def clear(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of entries cleared
        """
        return clear_cache(
            self._metadata,
            self.cache_dir,
            self._save_metadata,
        )

    def cleanup(self, max_age_days: int | None = None) -> dict[str, int | float]:
        """
        Clean up cache by removing old or unused entries.

        Args:
            max_age_days: Remove entries older than this many days (uses TTL if None)

        Returns:
            Dictionary with cleanup statistics
        """
        if max_age_days is None:
            max_age_days = self.ttl_days

        return cleanup_cache(
            self._metadata,
            self._access_times,
            self.cache_dir,
            max_age_days,
            self._save_metadata,
        )

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return get_cache_stats(
            self._metadata,
            self.cache_dir,
            self.max_size_mb,
            self.max_entries,
        )


# Global cache instance with thread-safe initialization
_cache: IndicatorCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> IndicatorCache:
    """Get global cache instance (thread-safe)."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = IndicatorCache()
    return _cache
