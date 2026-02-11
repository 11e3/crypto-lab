"""Binance API candle fetching utilities."""

import time
from datetime import datetime
from typing import Any

import pandas as pd

from src.config.constants import BINANCE_API_RATE_LIMIT_DELAY, BINANCE_MAX_CANDLES_PER_REQUEST
from src.data.exchange_mapping import BinanceInterval
from src.utils.logger import get_logger

__all__ = ["fetch_binance_candles", "fetch_all_binance_candles"]

logger = get_logger(__name__)


def _get_exchange() -> Any:
    """Get ccxt Binance exchange instance.

    Returns:
        ccxt.binance exchange instance
    """
    import ccxt

    return ccxt.binance({"enableRateLimit": True})


def _ohlcv_to_dataframe(data: list[list[float]]) -> pd.DataFrame:
    """Convert ccxt OHLCV data to pandas DataFrame.

    Args:
        data: List of [timestamp, open, high, low, close, volume] lists

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns
    """
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.set_index("timestamp")
    df.index.name = "datetime"
    return df


def fetch_binance_candles(
    symbol: str,
    interval: BinanceInterval,
    count: int = BINANCE_MAX_CANDLES_PER_REQUEST,
    since: int | None = None,
) -> pd.DataFrame | None:
    """Fetch candle data from Binance API with retry logic.

    Args:
        symbol: Binance symbol (e.g., 'BTCUSDT', 'BTC/USDT')
        interval: Candle interval (e.g., '1d', '4h', '1m')
        count: Number of candles to fetch (max 1000)
        since: Start timestamp in milliseconds (optional)

    Returns:
        DataFrame with OHLCV data or None if error
    """
    max_retries = 3
    retry_delay = 1.0
    exchange = _get_exchange()

    for attempt in range(max_retries):
        try:
            ohlcv: list[list[float]] = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=interval,
                since=since,
                limit=min(count, BINANCE_MAX_CANDLES_PER_REQUEST),
            )
            if not ohlcv:
                return None
            return _ohlcv_to_dataframe(ohlcv)
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2**attempt)
                logger.warning(
                    f"Error fetching {symbol} (attempt {attempt + 1}): {e}. "
                    f"Retrying in {sleep_time}s..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(f"Error fetching {symbol} after {max_retries} attempts: {e}")
                return None
    return None


def fetch_all_binance_candles(
    symbol: str,
    interval: BinanceInterval,
    since: datetime | None = None,
    max_candles: int = 50000,
) -> pd.DataFrame | None:
    """Fetch all candles with pagination support.

    Args:
        symbol: Binance symbol (e.g., 'BTCUSDT', 'BTC/USDT')
        interval: Candle interval
        since: Only fetch candles after this datetime
        max_candles: Maximum number of candles to fetch

    Returns:
        DataFrame with all fetched OHLCV data
    """
    all_data: list[pd.DataFrame] = []
    since_ms: int | None = None
    total_fetched = 0

    if since is not None:
        since_ms = int(since.timestamp() * 1000)

    while total_fetched < max_candles:
        df = fetch_binance_candles(symbol, interval, since=since_ms)

        if df is None or df.empty:
            break

        all_data.append(df)
        total_fetched += len(df)

        # Move since forward to last candle's timestamp + 1ms
        last_ts = df.index.max()
        since_ms = int(last_ts.timestamp() * 1000) + 1

        if len(df) < BINANCE_MAX_CANDLES_PER_REQUEST:
            break

        time.sleep(BINANCE_API_RATE_LIMIT_DELAY)
        logger.debug(f"Fetched {total_fetched} candles for {symbol}...")

    if not all_data:
        return None

    combined = pd.concat(all_data)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    return combined
