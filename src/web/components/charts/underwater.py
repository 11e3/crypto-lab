"""Underwater (drawdown) chart component.

Underwater chart visualizing drawdowns.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.web.utils.chart_utils import CHART_HEIGHT_SECONDARY, COLOR_NEGATIVE

__all__ = ["render_underwater_curve", "calculate_drawdown"]


def calculate_drawdown(equity: np.ndarray) -> np.ndarray:
    """Calculate drawdown series from equity curve.

    Args:
        equity: Portfolio value array.

    Returns:
        Drawdown percentage array (negative values).
    """
    cummax = np.maximum.accumulate(equity)
    return (equity - cummax) / cummax * 100


def _build_underwater_figure(
    dates: np.ndarray,
    drawdown: np.ndarray,
) -> go.Figure:
    """Build Plotly figure for the underwater curve.

    Args:
        dates: Date array.
        drawdown: Drawdown percentage array.

    Returns:
        Configured Plotly Figure.
    """
    mdd_idx = int(np.argmin(drawdown))
    mdd_value = drawdown[mdd_idx]
    mdd_date = dates[mdd_idx]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(255, 23, 68, 0.2)",
            line={"color": COLOR_NEGATIVE, "width": 1},
            hovertemplate=(
                "<b>Date</b>: %{x|%Y-%m-%d}<br><b>Drawdown</b>: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[mdd_date],
            y=[mdd_value],
            mode="markers+text",
            name=f"MDD: {mdd_value:.2f}%",
            marker={"color": "darkred", "size": 12, "symbol": "triangle-down"},
            text=[f"MDD: {mdd_value:.2f}%"],
            textposition="bottom center",
            textfont={"color": "darkred", "size": 11},
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title={"text": "📉 Underwater Curve (Drawdown)", "font": {"size": 18}},
        xaxis={
            "title": "Date",
            "showgrid": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
        },
        yaxis={
            "title": "Drawdown (%)",
            "showgrid": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "ticksuffix": "%",
            "range": [min(drawdown) * 1.1, 5],
        },
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
        height=CHART_HEIGHT_SECONDARY,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
    )

    return fig


def render_underwater_curve(
    dates: np.ndarray,
    equity: np.ndarray,
) -> None:
    """Render underwater (drawdown) curve.

    Args:
        dates: Date array
        equity: Portfolio value array
    """
    if len(dates) == 0 or len(equity) == 0:
        st.warning("📊 No data to display.")
        return

    drawdown = calculate_drawdown(equity)
    fig = _build_underwater_figure(dates, drawdown)
    st.plotly_chart(fig, use_container_width=True)
