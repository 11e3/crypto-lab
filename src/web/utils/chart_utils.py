"""Chart utility functions.

Utilities for chart rendering optimization:
- Data downsampling for large datasets
- Color schemes
- Layout presets
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "downsample_timeseries",
    "CHART_HEIGHT_FULL",
    "CHART_HEIGHT_SECONDARY",
    "COLOR_PRIMARY",
    "COLOR_POSITIVE",
    "COLOR_NEGATIVE",
    "COLOR_NEUTRAL",
]

# Consistent chart height constants
CHART_HEIGHT_FULL = 450  # Main charts (equity curve)
CHART_HEIGHT_SECONDARY = 300  # Supporting charts (drawdown, yearly bar)

# Consistent color palette
COLOR_PRIMARY = "#1f77b4"  # Portfolio line, primary data
COLOR_POSITIVE = "#00C853"  # Positive returns, gains
COLOR_NEGATIVE = "#FF1744"  # Negative returns, losses, drawdown
COLOR_NEUTRAL = "#9E9E9E"  # Grid lines, baselines
COLOR_BENCHMARK = "#ff7f0e"  # Benchmark overlay


def downsample_timeseries(
    dates: pd.DatetimeIndex | np.ndarray,
    values: np.ndarray,
    max_points: int = 1000,
) -> tuple[pd.DatetimeIndex | np.ndarray, np.ndarray]:
    """Downsample time series data to improve chart rendering performance.

    Use simplified version of LTTB (Largest-Triangle-Three-Buckets) algorithm.
    Reduce large dataset to fixed number of points while preserving important features.

    Args:
        dates: Date/time array
        values: Value array
        max_points: Maximum number of points (default: 1000)

    Returns:
        (downsampled_dates, downsampled_values) tuple

    Examples:
        >>> dates = pd.date_range("2020-01-01", periods=10000, freq="1h")
        >>> values = np.random.randn(10000).cumsum()
        >>> ds_dates, ds_values = downsample_timeseries(dates, values, max_points=1000)
        >>> len(ds_dates)
        1000
    """
    n_points = len(values)

    # Return as-is if dataset is already small
    if n_points <= max_points:
        return dates, values

    # Uniform interval sampling
    # Always include start and end points
    indices = np.linspace(0, n_points - 1, max_points, dtype=int)

    # Remove duplicates (if any)
    indices = np.unique(indices)

    downsampled_dates = dates[indices] if isinstance(dates, np.ndarray) else dates[indices]
    downsampled_values = values[indices]

    return downsampled_dates, downsampled_values


def downsample_timeseries_lttb(
    dates: pd.DatetimeIndex | np.ndarray,
    values: np.ndarray,
    max_points: int = 1000,
) -> tuple[pd.DatetimeIndex | np.ndarray, np.ndarray]:
    """Advanced downsampling using LTTB algorithm.

    More sophisticated algorithm that selects visually important points.
    Advantageous for preserving sharp changes or peaks.

    Args:
        dates: Date/time array
        values: Value array
        max_points: Maximum number of points

    Returns:
        (downsampled_dates, downsampled_values) tuple
    """
    n_points = len(values)

    if n_points <= max_points:
        return dates, values

    # Calculate bucket size
    bucket_size = (n_points - 2) / (max_points - 2)

    # Store result indices
    sampled_indices: list[int] = [0]  # Always include first point

    # Select point with maximum triangle area from each bucket
    a_idx = 0
    for i in range(1, max_points - 1):
        # Current bucket range
        bucket_start = int((i - 1) * bucket_size) + 1
        bucket_end = int(i * bucket_size) + 1

        # Average point of next bucket
        next_bucket_start = int(i * bucket_size) + 1
        next_bucket_end = min(int((i + 1) * bucket_size) + 1, n_points)
        next_avg_x = (next_bucket_start + next_bucket_end) / 2
        next_avg_y = np.mean(values[next_bucket_start:next_bucket_end])

        # Find point with maximum triangle area in current bucket
        max_area = -1.0
        max_idx = bucket_start

        for idx in range(bucket_start, bucket_end):
            # Calculate triangle area
            area = abs(
                (a_idx - next_avg_x) * (values[idx] - values[a_idx])
                - (a_idx - idx) * (next_avg_y - values[a_idx])
            )

            if area > max_area:
                max_area = area
                max_idx = idx

        sampled_indices.append(max_idx)
        a_idx = max_idx

    # Always include last point
    sampled_indices.append(n_points - 1)

    # Sample by index
    sampled_indices_array = np.array(sampled_indices)
    downsampled_dates = (
        dates[sampled_indices_array]
        if isinstance(dates, np.ndarray)
        else dates[sampled_indices_array]
    )
    downsampled_values = values[sampled_indices_array]

    return downsampled_dates, downsampled_values
