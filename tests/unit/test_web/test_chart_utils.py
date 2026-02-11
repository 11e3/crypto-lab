"""Tests for chart utility functions: downsample_timeseries, downsample_timeseries_lttb."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.web.utils.chart_utils import (
    CHART_HEIGHT_FULL,
    CHART_HEIGHT_SECONDARY,
    COLOR_BENCHMARK,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_PRIMARY,
    downsample_timeseries,
    downsample_timeseries_lttb,
)

# =========================================================================
# Constants
# =========================================================================


class TestChartConstants:
    """Verify chart constants are defined correctly."""

    def test_heights_are_positive(self) -> None:
        assert CHART_HEIGHT_FULL > 0
        assert CHART_HEIGHT_SECONDARY > 0
        assert CHART_HEIGHT_FULL > CHART_HEIGHT_SECONDARY

    def test_colors_are_strings(self) -> None:
        for color in [
            COLOR_PRIMARY,
            COLOR_POSITIVE,
            COLOR_NEGATIVE,
            COLOR_NEUTRAL,
            COLOR_BENCHMARK,
        ]:
            assert isinstance(color, str)
            assert color.startswith("#")


# =========================================================================
# downsample_timeseries
# =========================================================================


class TestDownsampleTimeseries:
    """Tests for downsample_timeseries."""

    def test_no_downsampling_needed(self) -> None:
        dates = np.arange(100)
        values = np.random.randn(100)
        ds_dates, ds_values = downsample_timeseries(dates, values, max_points=200)
        assert len(ds_dates) == 100
        assert len(ds_values) == 100

    def test_exact_boundary(self) -> None:
        """When n_points == max_points, no downsampling."""
        dates = np.arange(500)
        values = np.random.randn(500)
        ds_dates, ds_values = downsample_timeseries(dates, values, max_points=500)
        assert len(ds_dates) == 500

    def test_downsampling_reduces_points(self) -> None:
        dates = np.arange(5000)
        values = np.random.randn(5000)
        ds_dates, ds_values = downsample_timeseries(dates, values, max_points=500)
        assert len(ds_dates) <= 500
        assert len(ds_values) <= 500

    def test_preserves_first_and_last(self) -> None:
        n = 10000
        dates = np.arange(n)
        values = np.arange(n, dtype=float)
        ds_dates, ds_values = downsample_timeseries(dates, values, max_points=100)
        assert ds_dates[0] == 0
        assert ds_dates[-1] == n - 1

    def test_works_with_datetime_index(self) -> None:
        dates = pd.date_range("2020-01-01", periods=2000, freq="h")
        values = np.random.randn(2000).cumsum()
        ds_dates, ds_values = downsample_timeseries(dates, values, max_points=500)
        assert len(ds_dates) <= 500

    def test_works_with_numpy_array(self) -> None:
        dates = np.array(pd.date_range("2020-01-01", periods=2000, freq="h"))
        values = np.random.randn(2000).cumsum()
        ds_dates, ds_values = downsample_timeseries(dates, values, max_points=500)
        assert len(ds_dates) <= 500


# =========================================================================
# downsample_timeseries_lttb
# =========================================================================


class TestDownsampleTimeseriesLttb:
    """Tests for LTTB-based downsampling."""

    def test_no_downsampling_needed(self) -> None:
        dates = np.arange(100)
        values = np.random.randn(100)
        ds_dates, ds_values = downsample_timeseries_lttb(dates, values, max_points=200)
        assert len(ds_dates) == 100

    def test_downsampling_to_target(self) -> None:
        n = 5000
        dates = np.arange(n)
        values = np.sin(np.linspace(0, 10 * np.pi, n))  # Sinusoidal
        ds_dates, ds_values = downsample_timeseries_lttb(dates, values, max_points=500)
        assert len(ds_dates) == 500

    def test_preserves_first_and_last(self) -> None:
        n = 3000
        dates = np.arange(n)
        values = np.arange(n, dtype=float)
        ds_dates, ds_values = downsample_timeseries_lttb(dates, values, max_points=100)
        assert ds_dates[0] == 0
        assert ds_dates[-1] == n - 1

    def test_lttb_preserves_peaks_better(self) -> None:
        """LTTB should capture sharp peaks better than uniform sampling."""
        n = 10000
        dates = np.arange(n)
        # Create data with a sharp spike
        values = np.zeros(n)
        values[5000] = 100.0  # Single spike

        _, lttb_values = downsample_timeseries_lttb(dates, values, max_points=500)
        _, uniform_values = downsample_timeseries(dates, values, max_points=500)

        # LTTB should capture the spike; uniform might miss it
        # At minimum, LTTB max should be >= uniform max
        assert max(lttb_values) >= max(uniform_values)

    def test_works_with_datetime_index(self) -> None:
        dates = pd.date_range("2020-01-01", periods=2000, freq="h")
        values = np.random.randn(2000).cumsum()
        ds_dates, ds_values = downsample_timeseries_lttb(dates, values, max_points=500)
        assert len(ds_dates) == 500
