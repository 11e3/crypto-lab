"""Yearly returns bar chart component.

Yearly returns bar chart.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.utils.chart_utils import CHART_HEIGHT_SECONDARY, COLOR_NEGATIVE, COLOR_POSITIVE

__all__ = ["render_yearly_bar_chart", "calculate_yearly_returns"]


def calculate_yearly_returns(
    dates: np.ndarray,
    equity: np.ndarray,
) -> pd.DataFrame:
    """Calculate yearly returns.

    Args:
        dates: Date array
        equity: Portfolio value array

    Returns:
        Yearly returns DataFrame (columns: year, return_pct)
    """
    if len(dates) == 0 or len(equity) == 0:
        return pd.DataFrame(columns=["year", "return_pct"])

    # Create DataFrame
    df = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity})
    df["year"] = df["date"].dt.year

    # Calculate first/last day values for each year
    yearly = (
        df.groupby("year")
        .agg(
            first_equity=("equity", "first"),
            last_equity=("equity", "last"),
        )
        .reset_index()
    )

    # Calculate yearly returns
    yearly["return_pct"] = (yearly["last_equity"] / yearly["first_equity"] - 1) * 100

    return yearly[["year", "return_pct"]]


def _build_yearly_bar_figure(yearly: pd.DataFrame) -> go.Figure:
    """Build Plotly bar chart for yearly returns.

    Args:
        yearly: DataFrame with year, return_pct columns.

    Returns:
        Configured Plotly Figure.
    """
    colors = [COLOR_POSITIVE if r >= 0 else COLOR_NEGATIVE for r in yearly["return_pct"]]
    avg_return = yearly["return_pct"].mean()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[str(y) for y in yearly["year"]],
            y=yearly["return_pct"],
            marker_color=colors,
            text=[f"{r:.1f}%" for r in yearly["return_pct"]],
            textposition="outside",
            textfont={"size": 12},
            hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
    fig.add_hline(
        y=avg_return,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Avg: {avg_return:.1f}%",
        annotation_position="right",
    )

    fig.update_layout(
        title={"text": "📊 Yearly Returns", "font": {"size": 18}},
        xaxis={"title": "Year", "tickmode": "linear"},
        yaxis={
            "title": "Return (%)",
            "ticksuffix": "%",
            "showgrid": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
        },
        template="plotly_white",
        showlegend=False,
        height=CHART_HEIGHT_SECONDARY,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
    )

    return fig


def _render_yearly_summary(yearly: pd.DataFrame) -> None:
    """Render summary statistics for yearly returns."""
    avg_return = yearly["return_pct"].mean()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Return", f"{avg_return:.1f}%")
    with col2:
        st.metric("Best Return", f"{yearly['return_pct'].max():.1f}%")
    with col3:
        st.metric("Worst Return", f"{yearly['return_pct'].min():.1f}%")
    with col4:
        positive_years = (yearly["return_pct"] > 0).sum()
        total_years = len(yearly)
        st.metric("Positive Years", f"{positive_years}/{total_years}")


def render_yearly_bar_chart(
    dates: np.ndarray,
    equity: np.ndarray,
) -> None:
    """Render yearly returns bar chart.

    Args:
        dates: Date array
        equity: Portfolio value array
    """
    if len(dates) == 0 or len(equity) == 0:
        st.warning("📊 No data to display.")
        return

    yearly = calculate_yearly_returns(dates, equity)
    if yearly.empty:
        st.warning("📊 No yearly data available.")
        return

    fig = _build_yearly_bar_figure(yearly)
    st.plotly_chart(fig, use_container_width=True)
    _render_yearly_summary(yearly)
