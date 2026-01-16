"""Underwater (drawdown) chart component.

Underwater chart visualizing drawdowns.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

__all__ = ["render_underwater_curve"]


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

    # Calculate drawdown
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax * 100  # Percent

    # Maximum drawdown location
    mdd_idx = np.argmin(drawdown)
    mdd_value = drawdown[mdd_idx]
    mdd_date = dates[mdd_idx]

    fig = go.Figure()

    # Drawdown area chart
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line={"color": "red", "width": 1},
            hovertemplate=(
                "<b>Date</b>: %{x|%Y-%m-%d}<br><b>Drawdown</b>: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    # Maximum drawdown point
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

    # Layout configuration
    fig.update_layout(
        title={
            "text": "📉 Underwater Curve (Drawdown)",
            "font": {"size": 18},
        },
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
            "range": [min(drawdown) * 1.1, 5],  # Slight margin
        },
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
    )

    st.plotly_chart(fig, use_container_width=True)
