"""
Adaptive VBO indicators (Phase 2 improvements).

Contains volatility regime, adaptive noise, and adaptive K-value calculations.
"""

from __future__ import annotations

import pandas as pd

from src.utils.indicators_vbo import (
    _atr_local,
    calculate_natr,
)


def calculate_volatility_regime(
    high: pd.Series[float],
    low: pd.Series[float],
    close: pd.Series[float],
    period: int = 14,
    window: int = 100,
) -> pd.Series[int]:
    """
    Volatility regime classification.

    - 0 (Low): NATR < 33rd percentile
    - 1 (Medium): 33rd <= NATR < 67th percentile
    - 2 (High): NATR >= 67th percentile
    """
    natr = calculate_natr(high, low, close, period)

    actual_window = min(window, len(natr) // 4) if len(natr) > 0 else window
    p33 = natr.rolling(window=actual_window).quantile(0.33)
    p67 = natr.rolling(window=actual_window).quantile(0.67)

    regime: pd.Series[int] = pd.Series(0, index=natr.index)
    regime[(natr >= p33) & (natr < p67)] = 1  # Medium
    regime[natr >= p67] = 2  # High

    return regime


def calculate_adaptive_noise(
    high: pd.Series[float],
    low: pd.Series[float],
    close: pd.Series[float],
    short_period: int = 4,
    long_period: int = 8,
    atr_period: int = 14,
) -> tuple[pd.Series[float], pd.Series[float]]:
    """
    ATR-normalized adaptive noise calculation.

    Returns:
        (short_noise_adaptive, long_noise_adaptive)
    """
    atr_values = _atr_local(high, low, close, atr_period)

    short_noise_raw = (
        high.rolling(window=short_period).max() - low.rolling(window=short_period).min()
    )
    long_noise_raw = high.rolling(window=long_period).max() - low.rolling(window=long_period).min()

    short_noise_adaptive: pd.Series[float] = short_noise_raw / (atr_values + 1e-8)
    long_noise_adaptive: pd.Series[float] = long_noise_raw / (atr_values + 1e-8)

    return short_noise_adaptive, long_noise_adaptive


def calculate_noise_ratio(
    high: pd.Series[float],
    low: pd.Series[float],
    close: pd.Series[float],
    short_period: int = 4,
    long_period: int = 8,
    atr_period: int = 14,
) -> pd.Series[float]:
    """
    Noise ratio (short / long) calculation.

    ratio >= 0.5: exclude signal (low confidence)
    ratio < 0.5: high confidence
    """
    short_noise, long_noise = calculate_adaptive_noise(
        high, low, close, short_period, long_period, atr_period
    )
    result: pd.Series[float] = short_noise / (long_noise + 1e-8)
    return result


def calculate_adaptive_k_value(
    high: pd.Series[float],
    low: pd.Series[float],
    close: pd.Series[float],
    base_k: float = 0.5,
    atr_period: int = 14,
    window: int = 100,
) -> pd.Series[float]:
    """
    Volatility regime-based adaptive K value.

    - Low volatility: K * 0.8 (increase sensitivity)
    - Medium volatility: K * 1.0 (baseline)
    - High volatility: K * 1.3 (reduce false signals)
    """
    regime = calculate_volatility_regime(high, low, close, atr_period, window)

    k_values: pd.Series[float] = pd.Series(base_k, index=regime.index)
    k_values[regime == 0] = base_k * 0.8
    k_values[regime == 1] = base_k * 1.0
    k_values[regime == 2] = base_k * 1.3

    return k_values


__all__ = [
    "calculate_volatility_regime",
    "calculate_adaptive_noise",
    "calculate_noise_ratio",
    "calculate_adaptive_k_value",
]
