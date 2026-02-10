"""
HTML-specific monthly returns calculation for heatmap visualization.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.backtester.report_pkg.report_returns import build_equity_dataframe

__all__ = ["calculate_monthly_returns_for_html"]


def calculate_monthly_returns_for_html(
    equity_curve: np.ndarray,
    dates: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate monthly and yearly returns for heatmap in HTML format.

    Args:
        equity_curve: Array of equity values
        dates: Array of dates

    Returns:
        Dictionary with years, months, values, text, yearly_returns, yearly_labels
    """
    df = build_equity_dataframe(equity_curve, dates)

    # Resample to monthly
    monthly = df["equity"].resample("ME").last()
    monthly_returns = monthly.pct_change() * 100

    # Resample to yearly
    yearly = df["equity"].resample("YE").last()
    yearly_returns = yearly.pct_change() * 100

    # Create monthly pivot table
    monthly_index = pd.DatetimeIndex(monthly_returns.index)
    monthly_df = pd.DataFrame(
        {
            "year": monthly_index.year,
            "month": monthly_index.month,
            "return": monthly_returns.values,
        }
    )

    pivot = monthly_df.pivot(index="year", columns="month", values="return")
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
    pivot.columns = pd.Index(month_names[: len(pivot.columns)])

    # Prepare monthly data for Plotly
    years: list[str] = [str(y) for y in pivot.index.tolist()]
    months: list[str] = list(pivot.columns)
    values: list[list[float]] = pivot.values.tolist()
    text: list[list[str]] = [
        [f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in values
    ]

    # Prepare yearly data
    yearly_index = pd.DatetimeIndex(yearly_returns.index)
    yearly_data: list[float] = []
    yearly_labels: list[str] = []

    for year in pivot.index:
        year_mask = yearly_index.year == year
        year_return_series = yearly_returns[year_mask]

        if len(year_return_series) > 0 and not np.isnan(year_return_series.iloc[0]):
            yearly_data.append(float(year_return_series.iloc[0]))
            yearly_labels.append(f"{year}: {year_return_series.iloc[0]:.1f}%")
        else:
            monthly_index_check = pd.DatetimeIndex(monthly_returns.index)
            year_monthly = monthly_returns[monthly_index_check.year == year]
            if len(year_monthly) > 0:
                year_total = float((1 + year_monthly / 100).prod() - 1)
                yearly_data.append(year_total * 100)
                yearly_labels.append(f"{year}: {year_total * 100:.1f}%")
            else:
                yearly_data.append(float("nan"))
                yearly_labels.append("")

    return {
        "years": years,
        "months": months,
        "values": values,
        "text": text,
        "yearly_returns": yearly_data,
        "yearly_labels": yearly_labels,
    }
