"""
Monthly and yearly returns calculation utilities.
"""

import numpy as np
import pandas as pd

__all__ = ["calculate_monthly_returns", "calculate_yearly_returns", "build_equity_dataframe"]


def build_equity_dataframe(equity_curve: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    """Build a DatetimeIndex DataFrame from equity curve and dates arrays."""
    df = pd.DataFrame({"date": dates, "equity": equity_curve})
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def calculate_monthly_returns(
    equity_curve: np.ndarray,
    dates: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate monthly returns for heatmap.

    Args:
        equity_curve: Array of equity values
        dates: Array of dates

    Returns:
        Pivot table with monthly returns by year
    """
    df = build_equity_dataframe(equity_curve, dates)

    monthly = df["equity"].resample("ME").last()
    monthly_returns = monthly.pct_change() * 100

    dt_index = pd.DatetimeIndex(monthly_returns.index)
    monthly_df = pd.DataFrame(
        {
            "year": dt_index.year,
            "month": dt_index.month,
            "return": monthly_returns.values,
        }
    )

    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = [
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
    ][: len(pivot.columns)]

    return pivot


def calculate_yearly_returns(
    equity_curve: np.ndarray,
    dates: np.ndarray,
) -> pd.Series:
    """
    Calculate yearly returns.

    Args:
        equity_curve: Array of equity values
        dates: Array of dates

    Returns:
        Series with yearly returns indexed by year
    """
    df = build_equity_dataframe(equity_curve, dates)

    yearly = df["equity"].resample("YE").last()
    yearly_returns = yearly.pct_change() * 100

    if len(yearly) > 0:
        first_year_return = (yearly.iloc[0] / equity_curve[0] - 1) * 100
        yearly_returns.iloc[0] = first_year_return

    dt_index = pd.DatetimeIndex(yearly_returns.index)
    yearly_returns.index = pd.Index(dt_index.year)

    return yearly_returns
