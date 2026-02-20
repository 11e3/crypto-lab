"""BTC reference data loader for volatility breakout strategies.

Loads KRW-BTC OHLCV data from local parquet files, used as a market filter
in VBO strategies (entry only when BTC trend is bullish).
"""

import logging
from pathlib import Path

import pandas as pd

from src.config.constants import UPBIT_DATA_DIR, parquet_filename

logger = logging.getLogger(__name__)

_BTC_TICKER = "KRW-BTC"


def _load_btc_data(data_dir: Path, interval: str) -> pd.DataFrame | None:
    """Load KRW-BTC OHLCV data from a parquet file.

    Searches data_dir first, then falls back to UPBIT_DATA_DIR.

    Args:
        data_dir: Primary directory to search for parquet files.
        interval: Candle interval (e.g. "day", "minute240").

    Returns:
        DataFrame with DatetimeIndex, or None if file is not found.
    """
    filename = parquet_filename(_BTC_TICKER, interval)

    candidates = [data_dir / filename, UPBIT_DATA_DIR / filename]
    for filepath in candidates:
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                df.index = pd.to_datetime(df.index)
                logger.debug(f"Loaded BTC data from {filepath} ({len(df)} rows)")
                return df
            except Exception as e:
                logger.warning(f"Failed to load BTC data from {filepath}: {e}")

    logger.debug(f"BTC data file not found for interval={interval!r}; BTC filter disabled")
    return None


__all__ = ["_load_btc_data"]
