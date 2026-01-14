"""
Technical indicators for trading strategies.

Provides vectorized indicator calculations using pandas/numpy.

기본 지표 (이 모듈):
    - sma, ema, atr, volatility_range, noise_ratio, noise_ratio_sma, target_price

VBO 지표 (indicators_vbo):
    - add_vbo_indicators, calculate_natr, calculate_volatility_regime
    - calculate_adaptive_noise, calculate_noise_ratio, calculate_adaptive_k_value
    - add_improved_indicators

모멘텀 지표 (indicators_momentum):
    - rsi, bollinger_bands, macd, stochastic
"""

import numpy as np
import pandas as pd

# Import base implementations
from src.utils.indicators_base import atr as _atr_base
from src.utils.indicators_base import ema as _ema_base
from src.utils.indicators_base import sma as _sma_base

# Re-export momentum indicators for backward compatibility
from src.utils.indicators_momentum import (
    bollinger_bands,
    macd,
    rsi,
    stochastic,
)

# Re-export VBO indicators for backward compatibility
from src.utils.indicators_vbo import (
    add_improved_indicators,
    add_vbo_indicators,
    calculate_adaptive_k_value,
    calculate_adaptive_noise,
    calculate_natr,
    calculate_noise_ratio,
    calculate_volatility_regime,
)

__all__ = [
    # 기본 지표
    "sma",
    "ema",
    "atr",
    "volatility_range",
    "noise_ratio",
    "noise_ratio_sma",
    "target_price",
    # 모멘텀 지표 (re-exported)
    "rsi",
    "bollinger_bands",
    "macd",
    "stochastic",
    # VBO 지표 (re-exported)
    "add_vbo_indicators",
    "calculate_natr",
    "calculate_volatility_regime",
    "calculate_adaptive_noise",
    "calculate_noise_ratio",
    "calculate_adaptive_k_value",
    "add_improved_indicators",
    # 편의 함수
    "calculate_atr",
    "calculate_noise",
    "calculate_sma",
]


def sma(series: pd.Series, period: int, exclude_current: bool = False) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Price series
        period: Lookback period
        exclude_current: If True, exclude current bar (matching legacy/bt.py behavior)

    Returns:
        SMA series
    """
    return _sma_base(series, period, exclude_current=exclude_current)


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Price series
        period: Lookback period

    Returns:
        EMA series
    """
    return _ema_base(series, period)


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
    return _atr_base(high, low, close, period)


def volatility_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Daily volatility range (high - low).

    Args:
        high: High prices
        low: Low prices

    Returns:
        Range series
    """
    return high - low


def noise_ratio(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Noise ratio: measures how much of the range is "noise" vs directional movement.

    Formula: Noise = 1 - abs(close - open) / (high - low)

    Higher values indicate more noise (less directional movement).

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Noise ratio series (0-1)
    """
    price_range = high - low
    body = (close - open_).abs()
    # Avoid division by zero
    result = np.where(price_range > 0, 1 - body / price_range, 0.0)
    return pd.Series(result, index=open_.index)


def noise_ratio_sma(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    exclude_current: bool = False,
) -> pd.Series:
    """
    Simple Moving Average of noise ratio (K value for VBO).

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
        exclude_current: If True, exclude current bar (matching legacy/bt.py behavior)

    Returns:
        SMA of noise ratio
    """
    nr = noise_ratio(open_, high, low, close)
    return pd.Series(sma(pd.Series(nr, index=open_.index), period, exclude_current=exclude_current))


def target_price(
    open_: pd.Series,
    prev_high: pd.Series,
    prev_low: pd.Series,
    k: pd.Series,
) -> pd.Series:
    """
    Calculate volatility breakout target price.

    Target = Open + (Prev_High - Prev_Low) * K

    Args:
        open_: Current open prices
        prev_high: Previous bar high prices
        prev_low: Previous bar low prices
        k: K value (noise ratio or constant)

    Returns:
        Target price series
    """
    prev_range = prev_high - prev_low
    return open_ + prev_range * k


# Convenience wrappers for ORB strategy
def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate ATR from DataFrame.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        window: ATR window

    Returns:
        ATR Series
    """
    return atr(df["high"], df["low"], df["close"], period=window)


def calculate_noise(df: pd.DataFrame) -> pd.Series:
    """
    Calculate noise ratio from DataFrame.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Noise ratio Series
    """
    return noise_ratio(df["open"], df["high"], df["low"], df["close"])


def calculate_sma(series: pd.Series | pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate SMA (Simple Moving Average).

    Args:
        series: Price series or DataFrame
        window: SMA window

    Returns:
        SMA Series
    """
    if isinstance(series, pd.DataFrame):
        series = series["close"]
    return sma(series, period=window, exclude_current=False)
