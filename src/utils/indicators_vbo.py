"""
VBO (Volatility Breakout) 전략용 지표.

노이즈 비율, 적응형 K값, 변동성 레짐 등 VBO 전략에 필요한 지표.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sma_local(
    series: pd.Series,
    period: int,
    *,
    exclude_current: bool = False,
) -> pd.Series:
    """Local SMA to avoid circular imports."""
    if exclude_current:
        shifted = series.shift(1)
        return shifted.rolling(window=period, min_periods=period).mean()
    return series.rolling(window=period, min_periods=period).mean()


def _noise_ratio_local(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Local noise ratio to avoid circular imports."""
    move = abs(close - open_)
    range_ = high - low
    return move / range_.replace(0, np.nan)


def _atr_local(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Local ATR to avoid circular imports."""
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def add_vbo_indicators(
    df: pd.DataFrame,
    sma_period: int = 4,
    trend_sma_period: int = 8,
    short_noise_period: int = 4,
    long_noise_period: int = 8,
    exclude_current: bool = False,
) -> pd.DataFrame:
    """
    VBO 전략에 필요한 지표를 추가.

    Args:
        df: OHLCV 데이터프레임
        sma_period: Exit SMA 기간
        trend_sma_period: Trend SMA 기간
        short_noise_period: 단기 노이즈 기간 (K 값용)
        long_noise_period: 장기 노이즈 기간
        exclude_current: 현재 바 제외 여부

    Returns:
        지표가 추가된 DataFrame
    """
    df = df.copy()

    # Noise ratios
    df["noise"] = _noise_ratio_local(df["open"], df["high"], df["low"], df["close"])
    df["short_noise"] = _sma_local(df["noise"], short_noise_period, exclude_current=exclude_current)
    df["long_noise"] = _sma_local(df["noise"], long_noise_period, exclude_current=exclude_current)

    # Moving averages
    df["sma"] = _sma_local(df["close"], sma_period, exclude_current=exclude_current)
    df["sma_trend"] = _sma_local(df["close"], trend_sma_period, exclude_current=exclude_current)

    # Previous day values
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_range"] = df["prev_high"] - df["prev_low"]

    # Target price (Volatility Breakout)
    df["target"] = df["open"] + df["prev_range"] * df["short_noise"]

    return df


def calculate_natr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Normalized Average True Range (NATR).

    NATR = (ATR / Close) * 100

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR 기간

    Returns:
        NATR series (%)
    """
    atr_values = _atr_local(high, low, close, period)
    return (atr_values / close) * 100


def calculate_volatility_regime(
    high: pd.Series[float],
    low: pd.Series[float],
    close: pd.Series[float],
    period: int = 14,
    window: int = 100,
) -> pd.Series[int]:
    """
    변동성 레짐 분류.

    - 0 (Low): NATR < 33rd percentile
    - 1 (Medium): 33rd <= NATR < 67th percentile
    - 2 (High): NATR >= 67th percentile

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR 기간
        window: 퍼센타일 계산 윈도우

    Returns:
        Volatility regime series (0/1/2)
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
    ATR로 정규화된 적응형 노이즈 계산.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        short_period: 단기 노이즈 윈도우
        long_period: 장기 노이즈 윈도우
        atr_period: ATR 기간

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


def add_improved_indicators(
    df: pd.DataFrame,
    short_period: int = 4,
    long_period: int = 8,
    atr_period: int = 14,
    base_k: float = 0.5,
) -> pd.DataFrame:
    """
    Phase 2 개선된 지표를 DataFrame에 추가.

    추가되는 컬럼:
    - atr, natr, volatility_regime
    - short_noise_adaptive, long_noise_adaptive, noise_ratio
    - k_value_adaptive

    Args:
        df: OHLC 데이터프레임
        short_period: 단기 기간
        long_period: 장기 기간
        atr_period: ATR 기간
        base_k: 기본 K 값

    Returns:
        지표가 추가된 DataFrame
    """
    result = df.copy()

    result["atr"] = _atr_local(df["high"], df["low"], df["close"], atr_period)
    result["natr"] = calculate_natr(df["high"], df["low"], df["close"], atr_period)
    result["volatility_regime"] = calculate_volatility_regime(
        df["high"], df["low"], df["close"], atr_period
    )

    short_noise, long_noise = calculate_adaptive_noise(
        df["high"], df["low"], df["close"], short_period, long_period, atr_period
    )
    result["short_noise_adaptive"] = short_noise
    result["long_noise_adaptive"] = long_noise
    result["noise_ratio"] = calculate_noise_ratio(
        df["high"], df["low"], df["close"], short_period, long_period, atr_period
    )

    result["k_value_adaptive"] = calculate_adaptive_k_value(
        df["high"], df["low"], df["close"], base_k, atr_period
    )

    return result


__all__ = [
    # Core VBO functions
    "add_vbo_indicators",
    "calculate_natr",
    "add_improved_indicators",
    # Local functions (for internal use)
    "_sma_local",
    "_noise_ratio_local",
    "_atr_local",
    "calculate_volatility_regime",
    "calculate_adaptive_noise",
    "calculate_noise_ratio",
    "calculate_adaptive_k_value",
]
