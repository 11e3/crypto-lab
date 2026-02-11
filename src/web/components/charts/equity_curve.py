"""Equity curve chart component.

Interactive equity curve chart based on Plotly.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.web.utils.chart_utils import (
    CHART_HEIGHT_FULL,
    COLOR_BENCHMARK,
    COLOR_PRIMARY,
    downsample_timeseries,
)

__all__ = ["render_equity_curve"]


def _prepare_data(
    dates: np.ndarray,
    equity: np.ndarray,
    benchmark: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Downsample dates, equity, and benchmark arrays if they exceed max_points.

    Returns:
        Tuple of (dates, equity, benchmark) after optional downsampling.
    """
    if len(dates) > max_points:
        downsampled_dates, downsampled_equity = downsample_timeseries(
            dates, equity, max_points=max_points
        )
        dates = np.asarray(downsampled_dates)
        equity = downsampled_equity
        if benchmark is not None:
            _, downsampled_benchmark = downsample_timeseries(
                dates, benchmark, max_points=max_points
            )
            benchmark = downsampled_benchmark
    return dates, equity, benchmark


def _create_portfolio_trace(
    dates: np.ndarray,
    normalized_equity: np.ndarray,
    returns_pct: np.ndarray,
) -> go.Scatter:
    """Create the portfolio equity curve trace."""
    return go.Scatter(
        x=dates,
        y=normalized_equity,
        mode="lines",
        name="Portfolio",
        line={"color": COLOR_PRIMARY, "width": 2},
        hovertemplate=(
            "<b>Date</b>: %{x|%Y-%m-%d}<br>"
            "<b>Value</b>: %{y:.2f}x<br>"
            "<b>Return</b>: %{customdata:.2f}%<extra></extra>"
        ),
        customdata=returns_pct,
    )


def _create_benchmark_trace(
    dates: np.ndarray,
    benchmark: np.ndarray | None,
    benchmark_name: str,
) -> go.Scatter | None:
    """Create the benchmark trace, or return None if benchmark is unavailable."""
    if benchmark is None or len(benchmark) != len(dates):
        return None
    normalized_benchmark = benchmark / benchmark[0]
    benchmark_returns = (normalized_benchmark - 1) * 100
    return go.Scatter(
        x=dates,
        y=normalized_benchmark,
        mode="lines",
        name=benchmark_name,
        line={"color": COLOR_BENCHMARK, "width": 1.5, "dash": "dash"},
        hovertemplate=(
            f"<b>{benchmark_name}</b><br>"
            "<b>Date</b>: %{x|%Y-%m-%d}<br>"
            "<b>Value</b>: %{y:.2f}x<br>"
            "<b>Return</b>: %{customdata:.2f}%<extra></extra>"
        ),
        customdata=benchmark_returns,
    )


def _get_equity_layout() -> dict:
    """Return the Plotly layout configuration for the equity curve chart."""
    return {
        "title": {
            "text": "\U0001f4c8 Portfolio Equity Curve",
            "font": {"size": 18},
        },
        "xaxis": {
            "title": "Date",
            "showgrid": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "rangeslider": {"visible": True},
            "rangeselector": {
                "buttons": [
                    {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ],
            },
        },
        "yaxis": {
            "title": "Normalized Value (1 = Start)",
            "type": "log",
            "showgrid": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "tickformat": ".2f",
        },
        "hovermode": "x unified",
        "template": "plotly_white",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        "height": CHART_HEIGHT_FULL,
        "margin": {"l": 60, "r": 20, "t": 80, "b": 60},
    }


def render_equity_curve(
    dates: np.ndarray,
    equity: np.ndarray,
    initial_capital: float = 1.0,
    benchmark: np.ndarray | None = None,
    benchmark_name: str = "Benchmark",
    max_points: int = 2000,
) -> None:
    """Render interactive equity curve.

    Automatically downsamples large datasets to improve rendering performance.

    Args:
        dates: Date array
        equity: Portfolio value array
        initial_capital: Initial capital
        benchmark: Benchmark value array (optional)
        benchmark_name: Benchmark name
        max_points: Maximum chart points (default: 2000, for performance optimization)
    """
    if len(dates) == 0 or len(equity) == 0:
        st.warning("\U0001f4ca No data to display.")
        return

    dates, equity, benchmark = _prepare_data(dates, equity, benchmark, max_points)

    # Normalize equity to start at 1
    initial_value = equity[0] if equity[0] != 0 else 1.0
    normalized_equity = equity / initial_value
    returns_pct = (normalized_equity - 1) * 100

    fig = go.Figure()
    fig.add_trace(_create_portfolio_trace(dates, normalized_equity, returns_pct))

    benchmark_trace = _create_benchmark_trace(dates, benchmark, benchmark_name)
    if benchmark_trace is not None:
        fig.add_trace(benchmark_trace)

    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="gray",
        annotation_text="Start (1.0)",
        annotation_position="bottom right",
    )

    fig.update_layout(**_get_equity_layout())
    st.plotly_chart(fig, use_container_width=True)
