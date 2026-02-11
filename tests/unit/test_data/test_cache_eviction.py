"""Tests for cache eviction and cleanup logic."""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest

from src.data.cache.cache_eviction import (
    cleanup_expired,
    enforce_cache_limits,
    evict_entry,
    get_total_size_mb,
    _evict_to_entry_limit,
    _evict_to_size_limit,
)


def _make_cache_file(cache_dir: Path, key: str, size: int = 100) -> Path:
    """Create a fake cache file with given size."""
    path = cache_dir / f"{key}.parquet"
    path.write_bytes(b"x" * size)
    return path


# =========================================================================
# evict_entry
# =========================================================================


class TestEvictEntry:
    """Tests for evict_entry."""

    def test_evicts_existing_file(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {"k1": {"created_at": 0}}
        access_times: OrderedDict[str, float] = OrderedDict({"k1": 1.0})
        _make_cache_file(tmp_path, "k1", size=200)

        size = evict_entry("k1", metadata, access_times, tmp_path)
        assert size == 200
        assert "k1" not in metadata
        assert "k1" not in access_times
        assert not (tmp_path / "k1.parquet").exists()

    def test_evicts_missing_file(self, tmp_path: Path) -> None:
        """Evicts metadata even if file doesn't exist."""
        metadata: dict[str, Any] = {"k1": {"created_at": 0}}
        access_times: OrderedDict[str, float] = OrderedDict({"k1": 1.0})

        size = evict_entry("k1", metadata, access_times, tmp_path)
        assert size == 0
        assert "k1" not in metadata

    def test_evicts_missing_key(self, tmp_path: Path) -> None:
        """Handles key not in metadata gracefully."""
        metadata: dict[str, Any] = {}
        access_times: OrderedDict[str, float] = OrderedDict()

        size = evict_entry("nonexistent", metadata, access_times, tmp_path)
        assert size == 0


# =========================================================================
# cleanup_expired
# =========================================================================


class TestCleanupExpired:
    """Tests for cleanup_expired."""

    def test_removes_expired_entries(self, tmp_path: Path) -> None:
        old_time = time.time() - (10 * 24 * 60 * 60)  # 10 days ago
        metadata: dict[str, Any] = {
            "old": {"created_at": old_time},
            "new": {"created_at": time.time()},
        }
        access_times: OrderedDict[str, float] = OrderedDict(
            {"old": old_time, "new": time.time()}
        )
        _make_cache_file(tmp_path, "old")
        _make_cache_file(tmp_path, "new")

        cleaned = cleanup_expired(metadata, access_times, tmp_path, ttl_days=7)
        assert cleaned == 1
        assert "old" not in metadata
        assert "new" in metadata

    def test_ttl_zero_disables_cleanup(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {"k1": {"created_at": 0}}
        access_times: OrderedDict[str, float] = OrderedDict({"k1": 0.0})

        cleaned = cleanup_expired(metadata, access_times, tmp_path, ttl_days=0)
        assert cleaned == 0
        assert "k1" in metadata

    def test_no_expired_entries(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {"k1": {"created_at": time.time()}}
        access_times: OrderedDict[str, float] = OrderedDict({"k1": time.time()})

        cleaned = cleanup_expired(metadata, access_times, tmp_path, ttl_days=7)
        assert cleaned == 0


# =========================================================================
# _evict_to_entry_limit
# =========================================================================


class TestEvictToEntryLimit:
    """Tests for _evict_to_entry_limit."""

    def test_evicts_lru_when_over_limit(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {
            "k1": {"created_at": 0},
            "k2": {"created_at": 0},
            "k3": {"created_at": 0},
        }
        access_times: OrderedDict[str, float] = OrderedDict(
            {"k1": 1.0, "k2": 2.0, "k3": 3.0}
        )
        for key in metadata:
            _make_cache_file(tmp_path, key)

        _evict_to_entry_limit(metadata, access_times, tmp_path, max_entries=2)
        assert len(metadata) == 2
        assert "k1" not in metadata  # LRU evicted first

    def test_no_eviction_when_under_limit(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {"k1": {"created_at": 0}}
        access_times: OrderedDict[str, float] = OrderedDict({"k1": 1.0})

        _evict_to_entry_limit(metadata, access_times, tmp_path, max_entries=5)
        assert len(metadata) == 1

    def test_handles_empty_access_times(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {"k1": {"created_at": 0}, "k2": {"created_at": 0}}
        access_times: OrderedDict[str, float] = OrderedDict()

        _evict_to_entry_limit(metadata, access_times, tmp_path, max_entries=1)
        # Can't evict without access_times
        assert len(metadata) == 2


# =========================================================================
# _evict_to_size_limit
# =========================================================================


class TestEvictToSizeLimit:
    """Tests for _evict_to_size_limit."""

    def test_evicts_when_over_size_limit(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {
            "k1": {"created_at": 0},
            "k2": {"created_at": 0},
        }
        access_times: OrderedDict[str, float] = OrderedDict(
            {"k1": 1.0, "k2": 2.0}
        )
        # Create large files (each ~0.5MB)
        _make_cache_file(tmp_path, "k1", size=512 * 1024)
        _make_cache_file(tmp_path, "k2", size=512 * 1024)

        _evict_to_size_limit(metadata, access_times, tmp_path, max_size_mb=0.6)
        assert "k1" not in metadata  # LRU evicted

    def test_no_eviction_when_under_size(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {"k1": {"created_at": 0}}
        access_times: OrderedDict[str, float] = OrderedDict({"k1": 1.0})
        _make_cache_file(tmp_path, "k1", size=100)

        _evict_to_size_limit(metadata, access_times, tmp_path, max_size_mb=10.0)
        assert "k1" in metadata


# =========================================================================
# get_total_size_mb
# =========================================================================


class TestGetTotalSizeMb:
    """Tests for get_total_size_mb."""

    def test_calculates_total_size(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {"k1": {}, "k2": {}}
        _make_cache_file(tmp_path, "k1", size=1024)
        _make_cache_file(tmp_path, "k2", size=2048)

        size = get_total_size_mb(metadata, tmp_path)
        assert size == pytest.approx(3072 / (1024 * 1024), abs=0.001)

    def test_missing_files_counted_as_zero(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {"missing": {}}
        size = get_total_size_mb(metadata, tmp_path)
        assert size == 0.0

    def test_empty_metadata(self, tmp_path: Path) -> None:
        size = get_total_size_mb({}, tmp_path)
        assert size == 0.0


# =========================================================================
# enforce_cache_limits (integration)
# =========================================================================


class TestEnforceCacheLimits:
    """Integration tests for enforce_cache_limits."""

    def test_enforces_entry_and_size_limits(self, tmp_path: Path) -> None:
        metadata: dict[str, Any] = {
            "k1": {"created_at": time.time()},
            "k2": {"created_at": time.time()},
            "k3": {"created_at": time.time()},
        }
        access_times: OrderedDict[str, float] = OrderedDict(
            {"k1": 1.0, "k2": 2.0, "k3": 3.0}
        )
        for key in metadata:
            _make_cache_file(tmp_path, key, size=100)

        enforce_cache_limits(
            metadata, access_times, tmp_path,
            max_entries=2, max_size_mb=100.0, ttl_days=30,
        )
        assert len(metadata) <= 2

    def test_enforces_ttl(self, tmp_path: Path) -> None:
        old_time = time.time() - (10 * 24 * 60 * 60)
        metadata: dict[str, Any] = {
            "old": {"created_at": old_time},
            "new": {"created_at": time.time()},
        }
        access_times: OrderedDict[str, float] = OrderedDict(
            {"old": old_time, "new": time.time()}
        )
        _make_cache_file(tmp_path, "old")
        _make_cache_file(tmp_path, "new")

        enforce_cache_limits(
            metadata, access_times, tmp_path,
            max_entries=100, max_size_mb=100.0, ttl_days=7,
        )
        assert "old" not in metadata
        assert "new" in metadata
