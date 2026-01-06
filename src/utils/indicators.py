"""
Technical indicators for trading strategies.

Provides vectorized indicator calculations using pandas/numpy.
"""

import numpy as np
import pandas as pd


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
    if exclude_current:
        # Exclude current bar: use past 'period' bars (matching legacy/bt.py)
        # rolling(window=period) includes current bar, so we shift by 1
        return series.rolling(window=period, min_periods=period).mean().shift(1)
    else:
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
    return np.where(price_range > 0, 1 - body / price_range, 0.0)


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
    return sma(pd.Series(nr, index=open_.index), period, exclude_current=exclude_current)


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


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        series: Price series
        period: Lookback period

    Returns:
        RSI series (0-100)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Args:
        series: Price series
        period: SMA period
        std_dev: Standard deviation multiplier

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.

    Args:
        series: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    return k, d


def add_vbo_indicators(
    df: pd.DataFrame,
    sma_period: int = 4,
    trend_sma_period: int = 8,
    short_noise_period: int = 4,
    long_noise_period: int = 8,
    exclude_current: bool = False,
) -> pd.DataFrame:
    """
    Add all indicators required for Volatility Breakout strategy.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        sma_period: Period for exit SMA
        trend_sma_period: Period for trend SMA
        short_noise_period: Period for short-term noise (K value)
        long_noise_period: Period for long-term noise baseline
        exclude_current: If True, exclude current bar from calculations (matching legacy/bt.py)

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # Noise ratios
    df["noise"] = noise_ratio(df["open"], df["high"], df["low"], df["close"])
    df["short_noise"] = sma(df["noise"], short_noise_period, exclude_current=exclude_current)
    df["long_noise"] = sma(df["noise"], long_noise_period, exclude_current=exclude_current)

    # Moving averages
    df["sma"] = sma(df["close"], sma_period, exclude_current=exclude_current)
    df["sma_trend"] = sma(df["close"], trend_sma_period, exclude_current=exclude_current)

    # Previous day values for target calculation
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_range"] = df["prev_high"] - df["prev_low"]

    # Target price (Volatility Breakout)
    df["target"] = df["open"] + df["prev_range"] * df["short_noise"]

    return df
