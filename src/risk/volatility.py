"""Volatility calculation utilities for position sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["calculate_return_volatility"]


def calculate_return_volatility(
    historical_data: pd.DataFrame,
    lookback_period: int,
    price_column: str = "close",
) -> float | None:
    """
    Calculate return volatility from historical price data.

    Args:
        historical_data: DataFrame with price data
        lookback_period: Number of recent periods to use
        price_column: Column name for prices

    Returns:
        Volatility (std of returns) or None if data is insufficient,
        volatility is zero, or result is NaN.
    """
    if len(historical_data) < lookback_period:
        return None

    recent_data = historical_data.tail(lookback_period)
    returns = recent_data[price_column].pct_change().dropna()
    volatility = float(returns.std())

    if volatility <= 0 or np.isnan(volatility):
        return None

    return volatility
