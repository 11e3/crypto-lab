"""
Base indicator functions used by all indicator modules.

This module provides core calculation primitives to avoid circular imports.
Other indicator modules should import from here, not from indicators.py.
"""

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "atr",
    "noise_ratio",
]


def sma(
    series: pd.Series,
    period: int,
    *,
    exclude_current: bool = False,
) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Price series
        period: Lookback period
        exclude_current: If True, exclude current bar

    Returns:
        SMA series
    """
    if exclude_current:
        shifted = series.shift(1)
        return shifted.rolling(window=period, min_periods=period).mean()
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Price series
        period: Lookback period

    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period

    Returns:
        ATR series
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def noise_ratio(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Noise ratio: measures directional movement relative to range.

    Formula: abs(close - open) / (high - low)

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Noise ratio series
    """
    move = abs(close - open_)
    range_ = high - low
    # Avoid division by zero
    result = np.where(range_ > 0, move / range_, np.nan)
    return pd.Series(result, index=open_.index)
