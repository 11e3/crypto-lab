"""Monthly returns heatmap component.

Monthly returns heatmap chart.
"""

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
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

    # Create DataFrame with date index
    df = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity})
    df = df.set_index("date").sort_index()

    # Resample to get end-of-month values (forward fill to handle missing dates)
    monthly_equity = df["equity"].resample("ME").last()

    # Calculate monthly returns
    monthly_returns = monthly_equity.pct_change() * 100

    # Create result DataFrame
    idx = pd.to_datetime(monthly_equity.index)
    result = pd.DataFrame(
        {"year": idx.year, "month": idx.month, "return_pct": monthly_returns.values}
    )

    # For the first month, calculate return from start to end of that month
    if len(result) > 0 and len(df) > 0:
        first_month_start_equity = df["equity"].iloc[0]
        first_month_end_equity = monthly_equity.iloc[0]
        first_month_start_date = df.index[0]
        first_month_end_date = monthly_equity.index[0]

        if first_month_start_date.month == first_month_end_date.month:
            # Both in same month - calculate from first data point to month end
            first_return = (first_month_end_equity / first_month_start_equity - 1) * 100
            result.loc[0, "return_pct"] = first_return

    # Drop rows with NaN returns (partial months, edge cases)
    result = result.dropna(subset=["return_pct"])

    return result


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
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    # Prepare data
    z_data = pivot.values
    years = [str(y) for y in pivot.index.tolist()]
    num_years = len(years)

    # Compute symmetric color range so 0% always maps to white
    non_nan_values = z_data[~np.isnan(z_data)]
    if len(non_nan_values) > 0:
        abs_max = max(abs(float(non_nan_values.min())), abs(float(non_nan_values.max())))
        abs_max = max(abs_max, 1.0)
    else:
        abs_max = 10.0

    # Cap extreme outliers for better color scale (95th percentile)
    if len(non_nan_values) > 4:
        p95 = float(np.percentile(np.abs(non_nan_values), 95))
        abs_max = min(abs_max, max(p95 * 1.2, 5.0))

    # Build annotation text matrix (show value or empty for NaN)
    annotation_text = []
    for i in range(num_years):
        row = []
        for j in range(12):
            val = z_data[i, j]
            row.append(f"{val:.1f}%" if not np.isnan(val) else "")
        annotation_text.append(row)

    # Replace NaN with 0 for plotly (ff.create_annotated_heatmap can't handle None)
    # Empty months show as white (0% = center of diverging scale) with no annotation
    z_display = np.nan_to_num(z_data, nan=0.0).tolist()

    # Color scale: red-white-green, normalized to [-abs_max, abs_max]
    colorscale = [
        [0.0, "rgb(165, 0, 38)"],
        [0.25, "rgb(215, 48, 39)"],
        [0.4, "rgb(244, 109, 67)"],
        [0.5, "rgb(255, 255, 255)"],
        [0.6, "rgb(166, 217, 106)"],
        [0.75, "rgb(102, 189, 99)"],
        [1.0, "rgb(0, 104, 55)"],
    ]

    # Use annotated heatmap — handles single-row properly
    fig = ff.create_annotated_heatmap(
        z=z_display,
        x=month_names,
        y=years,
        annotation_text=annotation_text,
        colorscale=colorscale,
        zmin=-abs_max,
        zmax=abs_max,
        showscale=True,
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
        xgap=3,
        ygap=3,
    )

    # Fix annotation font colors based on cell value
    color_threshold = abs_max * 0.35
    for ann in fig.layout.annotations:
        text = ann.text
        if text and text != "":
            try:
                val = float(text.replace("%", ""))
                ann.font.color = "white" if abs(val) > color_threshold else "black"
                ann.font.size = 12
            except ValueError:
                ann.font.color = "black"
                ann.font.size = 12

    # Fixed row height for consistent cell sizing
    row_height = 50
    chart_height = max(150, 80 + num_years * row_height)
    chart_height = min(chart_height, 800)

    fig.update_layout(
        title={"text": "📅 Monthly Returns Heatmap", "font": {"size": 16}},
        xaxis={"title": "", "side": "top"},
        yaxis={"title": "", "autorange": "reversed"},
        template="plotly_white",
        height=chart_height,
        margin={"l": 50, "r": 20, "t": 60, "b": 20},
    )

    # Add colorbar config to the heatmap trace
    fig.data[0].update(
        colorbar={"title": "Return (%)", "ticksuffix": "%"},
    )

    st.plotly_chart(fig, use_container_width=True)
