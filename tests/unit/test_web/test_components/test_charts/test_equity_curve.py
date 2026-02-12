"""Tests for equity curve chart component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.web.components.charts.equity_curve import (
    _create_benchmark_trace,
    _create_portfolio_trace,
    _get_equity_layout,
    _prepare_data,
    render_equity_curve,
)

# =========================================================================
# _prepare_data
# =========================================================================


class TestPrepareData:
    """Tests for _prepare_data."""

    def test_no_downsampling_when_under_limit(self) -> None:
        """Should not downsample when data is under max_points."""
        dates = np.arange(100)
        equity = np.linspace(100, 200, 100)
        d, e, b = _prepare_data(dates, equity, None, max_points=200)
        assert len(d) == 100
        assert len(e) == 100
        assert b is None

    def test_downsampling_when_over_limit(self) -> None:
        """Should downsample when data exceeds max_points."""
        dates = np.array(pd.date_range("2020-01-01", periods=500))
        equity = np.linspace(100, 200, 500)
        d, e, b = _prepare_data(dates, equity, None, max_points=100)
        assert len(d) <= 100
        assert len(e) <= 100

    def test_benchmark_under_limit_preserved(self) -> None:
        """Benchmark should be preserved as-is when under max_points."""
        dates = np.array(pd.date_range("2020-01-01", periods=50))
        equity = np.linspace(100, 200, 50)
        benchmark = np.linspace(100, 150, 50)
        d, e, b = _prepare_data(dates, equity, benchmark, max_points=100)
        assert b is not None
        assert len(b) == 50

    def test_benchmark_downsampled_with_equity(self) -> None:
        """Benchmark should be downsampled correctly when over max_points."""
        dates = np.array(pd.date_range("2020-01-01", periods=500))
        equity = np.linspace(100, 200, 500)
        benchmark = np.linspace(100, 150, 500)
        d, e, b = _prepare_data(dates, equity, benchmark, max_points=100)
        assert b is not None
        assert len(d) == len(e) == len(b)

    def test_none_benchmark_stays_none(self) -> None:
        """None benchmark should remain None after downsampling."""
        dates = np.array(pd.date_range("2020-01-01", periods=500))
        equity = np.linspace(100, 200, 500)
        d, e, b = _prepare_data(dates, equity, None, max_points=100)
        assert b is None


# =========================================================================
# _create_portfolio_trace
# =========================================================================


class TestCreatePortfolioTrace:
    """Tests for _create_portfolio_trace."""

    def test_returns_scatter(self) -> None:
        dates = np.array(pd.date_range("2023-01-01", periods=5))
        equity = np.array([1.0, 1.1, 1.2, 1.15, 1.25])
        returns = (equity - 1) * 100
        trace = _create_portfolio_trace(dates, equity, returns)
        assert isinstance(trace, go.Scatter)
        assert trace.name == "Portfolio"

    def test_custom_data_has_returns(self) -> None:
        dates = np.array(pd.date_range("2023-01-01", periods=3))
        equity = np.array([1.0, 1.1, 1.2])
        returns = np.array([0.0, 10.0, 20.0])
        trace = _create_portfolio_trace(dates, equity, returns)
        np.testing.assert_array_equal(trace.customdata, returns)


# =========================================================================
# _create_benchmark_trace
# =========================================================================


class TestCreateBenchmarkTrace:
    """Tests for _create_benchmark_trace."""

    def test_none_benchmark_returns_none(self) -> None:
        dates = np.array(pd.date_range("2023-01-01", periods=5))
        assert _create_benchmark_trace(dates, None, "BTC") is None

    def test_mismatched_length_returns_none(self) -> None:
        dates = np.array(pd.date_range("2023-01-01", periods=5))
        benchmark = np.array([100.0, 110.0, 120.0])  # length 3 != 5
        assert _create_benchmark_trace(dates, benchmark, "BTC") is None

    def test_valid_benchmark_returns_scatter(self) -> None:
        dates = np.array(pd.date_range("2023-01-01", periods=5))
        benchmark = np.array([100.0, 110.0, 120.0, 115.0, 125.0])
        trace = _create_benchmark_trace(dates, benchmark, "BTC")
        assert isinstance(trace, go.Scatter)
        assert trace.name == "BTC"


# =========================================================================
# _get_equity_layout
# =========================================================================


class TestGetEquityLayout:
    """Tests for _get_equity_layout."""

    def test_returns_dict(self) -> None:
        layout = _get_equity_layout()
        assert isinstance(layout, dict)
        assert "xaxis" in layout
        assert "yaxis" in layout
        assert "height" in layout

    def test_log_scale_y(self) -> None:
        layout = _get_equity_layout()
        assert layout["yaxis"]["type"] == "log"


# =========================================================================
# render_equity_curve
# =========================================================================


class TestRenderEquityCurve:
    """Tests for render_equity_curve."""

    @patch("src.web.components.charts.equity_curve.st")
    def test_empty_dates(self, mock_st: MagicMock) -> None:
        render_equity_curve(np.array([]), np.array([100.0]))
        mock_st.warning.assert_called_once()

    @patch("src.web.components.charts.equity_curve.st")
    def test_empty_equity(self, mock_st: MagicMock) -> None:
        render_equity_curve(np.array([1, 2, 3]), np.array([]))
        mock_st.warning.assert_called_once()

    @patch("src.web.components.charts.equity_curve.st")
    def test_renders_chart(self, mock_st: MagicMock) -> None:
        dates = np.array(pd.date_range("2023-01-01", periods=10))
        equity = np.linspace(100, 120, 10)
        render_equity_curve(dates, equity)
        mock_st.plotly_chart.assert_called_once()

    @patch("src.web.components.charts.equity_curve.st")
    def test_with_benchmark(self, mock_st: MagicMock) -> None:
        dates = np.array(pd.date_range("2023-01-01", periods=10))
        equity = np.linspace(100, 120, 10)
        benchmark = np.linspace(100, 110, 10)
        render_equity_curve(dates, equity, benchmark=benchmark, benchmark_name="BTC")
        mock_st.plotly_chart.assert_called_once()

    @patch("src.web.components.charts.equity_curve.st")
    def test_zero_initial_equity(self, mock_st: MagicMock) -> None:
        """Zero initial equity should not cause division by zero."""
        dates = np.array(pd.date_range("2023-01-01", periods=5))
        equity = np.array([0.0, 10.0, 20.0, 15.0, 25.0])
        render_equity_curve(dates, equity)
        mock_st.plotly_chart.assert_called_once()
