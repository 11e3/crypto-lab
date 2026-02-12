"""Tests for underwater (drawdown) chart component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.web.components.charts.underwater import (
    _build_underwater_figure,
    calculate_drawdown,
    render_underwater_curve,
)

# =========================================================================
# calculate_drawdown (pure function)
# =========================================================================


class TestCalculateDrawdown:
    """Tests for calculate_drawdown."""

    def test_no_drawdown(self) -> None:
        """Monotonically increasing equity has zero drawdown."""
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        dd = calculate_drawdown(equity)
        np.testing.assert_array_almost_equal(dd, [0.0, 0.0, 0.0, 0.0])

    def test_single_drawdown(self) -> None:
        """Equity that drops after peak."""
        equity = np.array([100.0, 120.0, 100.0, 110.0])
        dd = calculate_drawdown(equity)
        # At index 0: 0%, index 1: 0% (new peak)
        # At index 2: (100 - 120) / 120 * 100 = -16.67%
        # At index 3: (110 - 120) / 120 * 100 = -8.33%
        assert dd[0] == pytest.approx(0.0)
        assert dd[1] == pytest.approx(0.0)
        assert dd[2] == pytest.approx(-16.6667, abs=0.01)
        assert dd[3] == pytest.approx(-8.3333, abs=0.01)

    def test_full_recovery(self) -> None:
        """Drawdown that fully recovers."""
        equity = np.array([100.0, 80.0, 100.0, 120.0])
        dd = calculate_drawdown(equity)
        assert dd[0] == pytest.approx(0.0)
        assert dd[1] == pytest.approx(-20.0)
        assert dd[2] == pytest.approx(0.0)
        assert dd[3] == pytest.approx(0.0)

    def test_consecutive_drops(self) -> None:
        """Monotonically decreasing equity."""
        equity = np.array([100.0, 90.0, 80.0, 70.0])
        dd = calculate_drawdown(equity)
        assert dd[0] == pytest.approx(0.0)
        assert dd[1] == pytest.approx(-10.0)
        assert dd[2] == pytest.approx(-20.0)
        assert dd[3] == pytest.approx(-30.0)

    def test_returns_array(self) -> None:
        """Return type is numpy array."""
        equity = np.array([100.0, 110.0])
        result = calculate_drawdown(equity)
        assert isinstance(result, np.ndarray)


# =========================================================================
# _build_underwater_figure
# =========================================================================


class TestBuildUnderwaterFigure:
    """Tests for _build_underwater_figure."""

    def test_returns_figure(self) -> None:
        """Should return a plotly Figure."""
        import plotly.graph_objects as go

        dates = np.array(pd.date_range("2023-01-01", periods=5))
        drawdown = np.array([0.0, -5.0, -10.0, -3.0, 0.0])
        fig = _build_underwater_figure(dates, drawdown)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self) -> None:
        """Figure should have drawdown fill and MDD marker."""
        dates = np.array(pd.date_range("2023-01-01", periods=5))
        drawdown = np.array([0.0, -5.0, -10.0, -3.0, 0.0])
        fig = _build_underwater_figure(dates, drawdown)
        assert len(fig.data) == 2

    def test_mdd_marker_at_min(self) -> None:
        """MDD marker should be at the minimum drawdown point."""
        dates = np.array(pd.date_range("2023-01-01", periods=5))
        drawdown = np.array([0.0, -5.0, -20.0, -3.0, 0.0])
        fig = _build_underwater_figure(dates, drawdown)
        # Second trace is MDD marker
        mdd_trace = fig.data[1]
        assert mdd_trace.y[0] == pytest.approx(-20.0)


# =========================================================================
# render_underwater_curve
# =========================================================================


class TestRenderUnderwaterCurve:
    """Tests for render_underwater_curve."""

    @patch("src.web.components.charts.underwater.st")
    def test_empty_dates(self, mock_st: MagicMock) -> None:
        """Should show warning for empty data."""
        render_underwater_curve(np.array([]), np.array([100.0]))
        mock_st.warning.assert_called_once()

    @patch("src.web.components.charts.underwater.st")
    def test_empty_equity(self, mock_st: MagicMock) -> None:
        """Should show warning for empty equity."""
        render_underwater_curve(np.array([1, 2, 3]), np.array([]))
        mock_st.warning.assert_called_once()

    @patch("src.web.components.charts.underwater.st")
    def test_renders_chart(self, mock_st: MagicMock) -> None:
        """Should call st.plotly_chart with valid data."""
        dates = np.array(pd.date_range("2023-01-01", periods=10))
        equity = np.array([100.0, 110.0, 105.0, 115.0, 108.0, 120.0, 118.0, 125.0, 130.0, 128.0])
        render_underwater_curve(dates, equity)
        mock_st.plotly_chart.assert_called_once()
