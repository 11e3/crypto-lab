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

import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CACHE_METADATA_FILENAME, PROCESSED_DATA_DIR
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
        self._metadata = self._load_metadata()
        # Track access times for LRU eviction
        self._access_times: OrderedDict[str, float] = OrderedDict()
        self._load_access_times()

    def _load_metadata(self) -> dict[str, Any]:
        """Load cache metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data: Any = json.load(f)
                    if isinstance(data, dict):
                        # Separate metadata and access times
                        if "_access_times" in data:
                            access_times_data = data.pop("_access_times")
                            if isinstance(access_times_data, dict):
                                self._access_times = OrderedDict(
                                    sorted(
                                        access_times_data.items(),
                                        key=lambda x: x[1]  # Sort by access time
                                    )
                                )
                        return data
                    return {}
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _load_access_times(self) -> None:
        """Load access times from metadata."""
        # Access times are loaded in _load_metadata
        pass

    def _save_metadata(self) -> None:
        """Save cache metadata to JSON file."""
        # Include access times in metadata
        data_to_save = self._metadata.copy()
        data_to_save["_access_times"] = dict(self._access_times)
        with open(self.metadata_file, "w") as f:
            json.dump(data_to_save, f, indent=2)

    def _generate_cache_key(
        self,
        ticker: str,
        interval: str,
        params: dict[str, Any],
    ) -> str:
        """
        Generate unique cache key from ticker, interval, and parameters.

        Args:
            ticker: Asset ticker
            interval: Data interval
            params: Indicator parameters

        Returns:
            Cache key string
        """
        # Create deterministic hash of parameters
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{ticker}_{interval}_{params_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.parquet"

    def get(
        self,
        ticker: str,
        interval: str,
        params: dict[str, Any],
        raw_data_mtime: float | None = None,
    ) -> pd.DataFrame | None:
        """
        Retrieve cached indicators if valid.

        Args:
            ticker: Asset ticker
            interval: Data interval
            params: Indicator parameters used for calculation
            raw_data_mtime: Modification time of raw data file (for invalidation)

        Returns:
            Cached DataFrame with indicators, or None if cache miss
        """
        cache_key = self._generate_cache_key(ticker, interval, params)
        cache_path = self._get_cache_path(cache_key)

        # Check if cache file exists
        if not cache_path.exists():
            logger.debug(f"Cache miss (file not found): {cache_key}")
            return None

        # Check metadata
        if cache_key not in self._metadata:
            logger.debug(f"Cache miss (no metadata): {cache_key}")
            return None

        meta = self._metadata[cache_key]

        # Verify parameters match
        if meta.get("params") != params:
            logger.debug(f"Cache miss (params changed): {cache_key}")
            return None

        # Check if raw data was updated
        if raw_data_mtime is not None:
            cached_mtime = meta.get("raw_data_mtime", 0)
            if raw_data_mtime > cached_mtime:
                logger.debug(f"Cache miss (raw data updated): {cache_key}")
                return None

        # Update access time for LRU
        current_time = time.time()
        if cache_key in self._access_times:
            # Move to end (most recently used)
            del self._access_times[cache_key]
        self._access_times[cache_key] = current_time

        # Load cached data
        try:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)
            logger.debug(f"Cache hit: {cache_key}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            # Remove from access times if load failed
            self._access_times.pop(cache_key, None)
            return None

    def set(
        self,
        ticker: str,
        interval: str,
        params: dict[str, Any],
        df: pd.DataFrame,
        raw_data_mtime: float | None = None,
    ) -> None:
        """
        Store calculated indicators in cache.

        Args:
            ticker: Asset ticker
            interval: Data interval
            params: Indicator parameters used for calculation
            df: DataFrame with calculated indicators
            raw_data_mtime: Modification time of raw data file
        """
        cache_key = self._generate_cache_key(ticker, interval, params)
        cache_path = self._get_cache_path(cache_key)

        # Save DataFrame with optional compression
        compression = "snappy" if self.use_compression else None
        df.to_parquet(cache_path, engine="pyarrow", compression=compression)

        # Update metadata
        current_time = time.time()
        self._metadata[cache_key] = {
            "ticker": ticker,
            "interval": interval,
            "params": params,
            "raw_data_mtime": raw_data_mtime or 0,
            "rows": len(df),
            "created_at": current_time,
        }
        
        # Update access time
        self._access_times[cache_key] = current_time

        # Check cache limits and evict if necessary
        self._enforce_cache_limits()

        self._save_metadata()

        logger.debug(f"Cache saved: {cache_key} ({len(df)} rows)")

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
        keys_to_remove = []

        for key, meta in self._metadata.items():
            if ticker and meta.get("ticker") != ticker:
                continue
            if interval and meta.get("interval") != interval:
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            # Remove file
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            # Remove metadata
            del self._metadata[key]

        if keys_to_remove:
            self._save_metadata()

        logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
        return len(keys_to_remove)

    def clear(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of entries cleared
        """
        count = len(self._metadata)

        # Remove all parquet files
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()

        # Clear metadata
        self._metadata = {}
        self._save_metadata()

        logger.info(f"Cleared {count} cache entries")
        return count

    def _enforce_cache_limits(self) -> None:
        """
        Enforce cache size and entry limits using LRU eviction.
        
        Note: In parallel environments, file deletion may fail due to locks.
        This is handled gracefully by catching exceptions in _evict_entry.
        """
        # Clean up expired entries first
        try:
            self._cleanup_expired()
        except (PermissionError, OSError) as e:
            # In parallel backtesting, cache cleanup may fail due to file locks
            # This is acceptable - cache will be cleaned up later
            logger.debug(f"Cache cleanup skipped due to file lock: {e}")
            return

        # Check entry limit
        max_attempts = len(self._metadata)  # Prevent infinite loop
        attempts = 0
        while len(self._metadata) > self.max_entries and attempts < max_attempts:
            attempts += 1
            # Remove least recently used
            if not self._access_times:
                break
            lru_key = next(iter(self._access_times))
            try:
                self._evict_entry(lru_key)
            except (PermissionError, OSError):
                # Skip if eviction fails (file may be locked)
                # Remove from access_times to avoid infinite loop
                self._access_times.pop(lru_key, None)
                continue

        # Check size limit
        current_size_mb = self._get_total_size_mb()
        attempts = 0
        max_attempts = len(self._metadata)
        while current_size_mb > self.max_size_mb and self._metadata and attempts < max_attempts:
            attempts += 1
            if not self._access_times:
                break
            lru_key = next(iter(self._access_times))
            try:
                evicted_size = self._evict_entry(lru_key)
                current_size_mb -= evicted_size / (1024 * 1024)
            except (PermissionError, OSError):
                # Skip if eviction fails (file may be locked)
                self._access_times.pop(lru_key, None)
                continue

    def _cleanup_expired(self) -> None:
        """Remove cache entries that have expired (TTL)."""
        if self.ttl_days <= 0:
            return  # TTL disabled

        current_time = time.time()
        ttl_seconds = self.ttl_days * 24 * 60 * 60
        expired_keys = []

        # Use list() to avoid dict size change during iteration
        for key, meta in list(self._metadata.items()):
            created_at = meta.get("created_at", 0)
            if current_time - created_at > ttl_seconds:
                expired_keys.append(key)

        cleaned_count = 0
        for key in expired_keys:
            try:
                self._evict_entry(key)
                cleaned_count += 1
            except (PermissionError, OSError):
                # Skip if eviction fails (file may be locked in parallel backtesting)
                continue

        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} expired cache entries")

    def _evict_entry(self, cache_key: str) -> int:
        """
        Evict a cache entry.

        Args:
            cache_key: Cache key to evict

        Returns:
            Size of evicted file in bytes
        """
        size = 0
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                size = cache_path.stat().st_size
                cache_path.unlink()
            except FileNotFoundError:
                pass  # Already deleted
            except PermissionError:
                # File may be in use by another process (parallel backtesting)
                # Log warning but don't fail - will be cleaned up later
                logger.debug(
                    f"Could not delete cache file {cache_path.name}: "
                    "file may be in use by another process"
                )
            except OSError as e:
                # Other OS errors (e.g., file locked)
                logger.debug(f"Could not delete cache file {cache_path.name}: {e}")

        # Remove from metadata even if file deletion failed
        # This prevents repeated attempts on the same entry
        self._metadata.pop(cache_key, None)
        self._access_times.pop(cache_key, None)

        return size

    def _get_total_size_mb(self) -> float:
        """Calculate total cache size in MB."""
        total_size = sum(
            self._get_cache_path(k).stat().st_size
            for k in self._metadata
            if self._get_cache_path(k).exists()
        )
        return total_size / (1024 * 1024)

    def cleanup(self, max_age_days: int | None = None) -> dict[str, int]:
        """
        Clean up cache by removing old or unused entries.

        Args:
            max_age_days: Remove entries older than this many days (uses TTL if None)

        Returns:
            Dictionary with cleanup statistics
        """
        if max_age_days is None:
            max_age_days = self.ttl_days

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        expired_keys = []
        for key, meta in self._metadata.items():
            created_at = meta.get("created_at", current_time)
            if current_time - created_at > max_age_seconds:
                expired_keys.append(key)

        expired_count = 0
        expired_size = 0
        for key in expired_keys:
            size = self._evict_entry(key)
            expired_count += 1
            expired_size += size

        if expired_count > 0:
            self._save_metadata()
            logger.info(
                f"Cleanup: removed {expired_count} entries ({expired_size / (1024 * 1024):.2f} MB)"
            )

        return {
            "removed_entries": expired_count,
            "removed_size_mb": expired_size / (1024 * 1024),
        }

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_rows = sum(m.get("rows", 0) for m in self._metadata.values())
        total_size_mb = self._get_total_size_mb()

        # Calculate age statistics
        current_time = time.time()
        ages = []
        for meta in self._metadata.values():
            created_at = meta.get("created_at", current_time)
            ages.append((current_time - created_at) / (24 * 60 * 60))  # days

        return {
            "entries": len(self._metadata),
            "total_rows": total_rows,
            "total_size_mb": round(total_size_mb, 2),
            "max_size_mb": self.max_size_mb,
            "max_entries": self.max_entries,
            "usage_pct": round((len(self._metadata) / self.max_entries) * 100, 1)
            if self.max_entries > 0
            else 0,
            "size_usage_pct": round((total_size_mb / self.max_size_mb) * 100, 1)
            if self.max_size_mb > 0
            else 0,
            "avg_age_days": round(sum(ages) / len(ages), 1) if ages else 0,
            "oldest_entry_days": round(max(ages), 1) if ages else 0,
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_cache: IndicatorCache | None = None


def get_cache() -> IndicatorCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = IndicatorCache()
    return _cache
