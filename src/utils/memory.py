"""
Memory optimization utilities for backtesting.

Provides tools for:
- Memory profiling
- Efficient data type management
- Memory usage monitoring
"""

import sys
from typing import Any, cast

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_memory_usage_mb(obj: Any) -> float:
    """
    Estimate memory usage of an object in MB.

    Args:
        obj: Object to measure

    Returns:
        Memory usage in MB
    """
    if isinstance(obj, np.ndarray):
        return float(obj.nbytes / (1024 * 1024))
    elif isinstance(obj, pd.DataFrame):
        return float(obj.memory_usage(deep=True).sum() / (1024 * 1024))
    elif isinstance(obj, dict):
        total: float = float(sys.getsizeof(obj))
        for key, value in obj.items():
            total += float(sys.getsizeof(key)) + get_memory_usage_mb(value) * 1024 * 1024
        return total / (1024 * 1024)
    elif isinstance(obj, list):
        total_bytes: float = float(sys.getsizeof(obj))
        for item in obj:
            total_bytes += get_memory_usage_mb(item) * 1024 * 1024
        return total_bytes / (1024 * 1024)
    else:
        return float(sys.getsizeof(obj) / (1024 * 1024))


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.

    Args:
        df: DataFrame to optimize

    Returns:
        DataFrame with optimized dtypes
    """
    df = df.copy()
    start_memory: float = float(df.memory_usage(deep=True).sum() / 1024**2)

    for col in df.columns:
        col_type = df[col].dtype

        if not isinstance(col_type, type(object)):
            c_min: Any = df[col].min()
            c_max: Any = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif (
                str(col_type)[:5] == "float"
                and col in ["open", "high", "low", "close", "volume", "target", "sma", "sma_trend"]
                or c_min > np.finfo(np.float32).min
                and c_max < np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)

    end_memory: float = float(df.memory_usage(deep=True).sum() / 1024**2)
    reduction: float = ((start_memory - end_memory) / start_memory) * 100

    logger.debug(
        f"Memory optimization: {start_memory:.2f} MB -> {end_memory:.2f} MB "
        f"({reduction:.1f}% reduction)"
    )

    return df


def use_float32_for_arrays() -> bool:
    """
    Check if float32 should be used for arrays (memory optimization).

    Returns:
        True if float32 should be used
    """
    return True  # Always use float32 for memory efficiency


def get_float_dtype() -> type[np.floating]:
    """
    Get the float dtype to use for arrays.

    Returns:
        np.float32 or np.float64
    """
    return cast(type[np.floating], np.float32 if use_float32_for_arrays() else np.float64)


def log_memory_usage(label: str, *objects: Any) -> None:
    """
    Log memory usage of objects.

    Args:
        label: Label for logging
        *objects: Objects to measure
    """
    total_mb = sum(get_memory_usage_mb(obj) for obj in objects)
    logger.debug(f"Memory usage [{label}]: {total_mb:.2f} MB")


class MemoryProfiler:
    """Context manager for profiling memory usage."""

    def __init__(self, label: str = "Operation") -> None:
        """
        Initialize memory profiler.

        Args:
            label: Label for the profiled operation
        """
        self.label = label
        self.start_memory: float | None = None

    def __enter__(self) -> "MemoryProfiler":
        """Start profiling."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            self.start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            logger.warning("psutil not available, memory profiling disabled")
            self.start_memory = None
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End profiling and log results."""
        if self.start_memory is None:
            return

        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            delta = end_memory - self.start_memory

            logger.info(
                f"Memory [{self.label}]: {self.start_memory:.2f} MB -> "
                f"{end_memory:.2f} MB (Δ {delta:+.2f} MB)"
            )
        except ImportError:
            pass
