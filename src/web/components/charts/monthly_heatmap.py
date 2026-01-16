"""Monthly returns heatmap component.

Monthly returns heatmap chart.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

__all__ = ["render_monthly_heatmap", "calculate_monthly_returns"]


def calculate_monthly_returns(
    dates: np.ndarray,
    equity: np.ndarray,
) -> pd.DataFrame:
    """Calculate monthly returns.

    Args:
        dates: Date array
        equity: Portfolio value array

    Returns:
        Monthly returns DataFrame (columns: year, month, return_pct)
    """
    if len(dates) == 0 or len(equity) == 0:
        return pd.DataFrame(columns=["year", "month", "return_pct"])

    # Create DataFrame
    df = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Calculate first/last day values for each month
    monthly = (
        df.groupby(["year", "month"])
        .agg(
            first_equity=("equity", "first"),
            last_equity=("equity", "last"),
        )
        .reset_index()
    )

    # Calculate monthly returns
    monthly["return_pct"] = (monthly["last_equity"] / monthly["first_equity"] - 1) * 100

    return monthly[["year", "month", "return_pct"]]


def render_monthly_heatmap(
    dates: np.ndarray,
    equity: np.ndarray,
) -> None:
    """Render monthly returns heatmap.

    Args:
        dates: Date array
        equity: Portfolio value array
    """
    if len(dates) == 0 or len(equity) == 0:
        st.warning("📊 No data to display.")
        return

    # Calculate monthly returns
    monthly = calculate_monthly_returns(dates, equity)

    if monthly.empty:
        st.warning("📊 No monthly data available.")
        return

    # Create pivot table (rows: year, columns: month)
    pivot = monthly.pivot(index="year", columns="month", values="return_pct")

    # Fill empty months
    all_months = list(range(1, 13))
    for month in all_months:
        if month not in pivot.columns:
            pivot[month] = np.nan
    pivot = pivot[all_months]

    # Month names
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Heatmap data
    z_data = pivot.values
    years = pivot.index.tolist()

    # Annotation text (return values)
    annotations = []
    for i, year in enumerate(years):
        for j, _month in enumerate(all_months):
            value = z_data[i, j]
            if not np.isnan(value):
                annotations.append(
                    {
                        "x": month_names[j],
                        "y": str(year),
                        "text": f"{value:.1f}%",
                        "showarrow": False,
                        "font": {
                            "color": "white" if abs(value) > 5 else "black",
                            "size": 10,
                        },
                    }
                )

    # Heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=month_names,
            y=[str(y) for y in years],
            colorscale=[
                [0.0, "rgb(165, 0, 38)"],  # Dark red (large loss)
                [0.25, "rgb(215, 48, 39)"],  # Red
                [0.4, "rgb(244, 109, 67)"],  # Light red
                [0.5, "rgb(255, 255, 255)"],  # White (0%)
                [0.6, "rgb(166, 217, 106)"],  # Light green
                [0.75, "rgb(102, 189, 99)"],  # Green
                [1.0, "rgb(0, 104, 55)"],  # Dark green (large profit)
            ],
            zmid=0,
            colorbar={
                "title": "Return (%)",
                "ticksuffix": "%",
            },
            hovertemplate=("<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>"),
        )
    )

    # Add annotations
    fig.update_layout(annotations=annotations)

    # Layout
    fig.update_layout(
        title={
            "text": "📅 Monthly Returns Heatmap",
            "font": {"size": 18},
        },
        xaxis={
            "title": "Month",
            "side": "top",
        },
        yaxis={
            "title": "Year",
            "autorange": "reversed",  # Latest year on top
        },
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 80, "b": 40},
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display yearly totals (compounded returns, not summed)
    yearly_data = []
    for year, group in monthly.groupby("year"):
        compounded_return = (np.prod(1 + group["return_pct"] / 100) - 1) * 100
        yearly_data.append({"year": year, "return_pct": compounded_return})

    yearly_returns = pd.DataFrame(yearly_data).set_index("year")["return_pct"]
    if not yearly_returns.empty:
        cols = st.columns(len(yearly_returns))
        for i, (year, ret) in enumerate(yearly_returns.items()):
            with cols[i]:
                st.metric(
                    label=f"{year}",
                    value=f"{ret:.1f}%",
                    delta=None,
                )
