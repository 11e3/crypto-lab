"""
VBO Signal Generation Functions.

Contains vectorized signal generation logic extracted from VanillaVBO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.strategies.base import Condition


def build_entry_signal(
    df: pd.DataFrame,
    conditions: list["Condition"],
) -> pd.Series[bool]:
    """
    Build entry signal based on configured conditions.

    Conditions logic:
    - Breakout: high >= target (volatility breakout)
    - SMABreakout: target > SMA (short-term trend)
    - TrendCondition: target > sma_trend (long-term trend)
    - NoiseCondition: short_noise < long_noise (volatility stability)

    Args:
        df: DataFrame with OHLCV and indicators
        conditions: List of entry conditions to evaluate

    Returns:
        Boolean Series indicating entry signals
    """
    entry_signal = pd.Series(True, index=df.index)

    for condition in conditions:
        entry_signal = _apply_entry_condition(df, condition, entry_signal)

    return entry_signal


def _apply_entry_condition(
    df: pd.DataFrame,
    condition: "Condition",
    signal: pd.Series[bool],
) -> pd.Series[bool]:
    """Apply a single entry condition to the signal."""
    name = condition.name

    if name == "Breakout":
        return signal & (df["high"] >= df["target"])

    if name == "SMABreakout":
        return signal & (df["target"] > df["sma"])

    if name in ("TrendCondition", "TrendFilter"):
        return signal & (df["target"] > df["sma_trend"])

    if name in ("NoiseCondition", "NoiseFilter"):
        return signal & (df["short_noise"] < df["long_noise"])

    if name in ("NoiseThresholdCondition", "NoiseThresholdFilter"):
        max_noise = getattr(condition, "max_noise", 0.7)
        return signal & (df["short_noise"] <= max_noise)

    if name in ("VolatilityRangeCondition", "VolatilityFilter"):
        min_vol = getattr(condition, "min_volatility_pct", 0.005)
        max_vol = getattr(condition, "max_volatility_pct", 0.15)
        range_pct = df["prev_range"] / df["open"]
        return signal & (range_pct >= min_vol) & (range_pct <= max_vol)

    if name == "VolatilityThreshold":
        min_range_pct = getattr(condition, "min_range_pct", 0.01)
        range_pct = df["prev_range"] / df["open"]
        return signal & (range_pct >= min_range_pct)

    return signal


def build_exit_signal(
    df: pd.DataFrame,
    conditions: list["Condition"],
) -> pd.Series[bool]:
    """
    Build exit signal based on configured conditions.

    Exit logic:
    - PriceBelowSMA: close < SMA (trend reversal)

    Args:
        df: DataFrame with OHLCV and indicators
        conditions: List of exit conditions to evaluate

    Returns:
        Boolean Series indicating exit signals
    """
    exit_signal = pd.Series(False, index=df.index)

    for condition in conditions:
        if condition.name == "PriceBelowSMA":
            exit_signal = exit_signal | (df["close"] < df["sma"])

    return exit_signal


__all__ = [
    "build_entry_signal",
    "build_exit_signal",
]
