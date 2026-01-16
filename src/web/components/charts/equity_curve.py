"""Equity curve chart component.

Interactive equity curve chart based on Plotly.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.web.utils.chart_utils import downsample_timeseries

__all__ = ["render_equity_curve"]


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
        st.warning("📊 No data to display.")
        return

    # Downsample data (improve performance for large datasets)
    if len(dates) > max_points:
        downsampled_dates, downsampled_equity = downsample_timeseries(
            dates, equity, max_points=max_points
        )
        dates = downsampled_dates  # type: ignore[assignment]
        equity = downsampled_equity
        if benchmark is not None:
            _, downsampled_benchmark = downsample_timeseries(
                dates, benchmark, max_points=max_points
            )
            benchmark = downsampled_benchmark

    # Convert to returns (%)
    returns_pct = (equity / initial_capital - 1) * 100

    fig = go.Figure()

    # Portfolio curve
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            mode="lines",
            name="Portfolio",
            line={"color": "#1f77b4", "width": 2},
            hovertemplate=(
                "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                "<b>Value</b>: %{y:,.0f} KRW<br>"
                "<b>Return</b>: %{customdata:.2f}%<extra></extra>"
            ),
            customdata=returns_pct,
        )
    )

    # Benchmark curve (if provided)
    if benchmark is not None and len(benchmark) == len(dates):
        benchmark_returns = (benchmark / benchmark[0] - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=benchmark,
                mode="lines",
                name=benchmark_name,
                line={"color": "#ff7f0e", "width": 1.5, "dash": "dash"},
                hovertemplate=(
                    f"<b>{benchmark_name}</b><br>"
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                    "<b>Value</b>: %{y:,.0f}<br>"
                    "<b>Return</b>: %{customdata:.2f}%<extra></extra>"
                ),
                customdata=benchmark_returns,
            )
        )

    # Initial capital baseline
    fig.add_hline(
        y=initial_capital,
        line_dash="dot",
        line_color="gray",
        annotation_text="Initial Capital",
        annotation_position="bottom right",
    )

    # Layout configuration
    fig.update_layout(
        title={
            "text": "📈 Portfolio Equity Curve",
            "font": {"size": 18},
        },
        xaxis={
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
                ]
            },
        },
        yaxis={
            "title": "Portfolio Value (KRW)",
            "showgrid": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "tickformat": ",",
        },
        hovermode="x unified",
        template="plotly_white",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 60, "r": 20, "t": 80, "b": 60},
    )

    st.plotly_chart(fig, use_container_width=True)
