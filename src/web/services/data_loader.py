"""Data loading service.

OHLCV data loading and caching service.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import BINANCE_DATA_DIR, RAW_DATA_DIR, parquet_filename
from src.data.collector_fetch import Interval
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "load_ticker_data",
    "get_data_files",
    "load_multiple_tickers_parallel",
    "get_data_date_range",
    "get_data_dir",
]


def get_data_dir(exchange: str = "upbit") -> Path:
    """Get the data directory for an exchange.

    Args:
        exchange: Exchange name ("upbit" or "binance")

    Returns:
        Path to the data directory
    """
    if exchange == "binance":
        return BINANCE_DATA_DIR
    return RAW_DATA_DIR


@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_ticker_data(
    ticker: str,
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
    exchange: str = "upbit",
) -> pd.DataFrame | None:
    """Load OHLCV data (1 hour cache).

    Args:
        ticker: Ticker symbol (e.g., KRW-BTC, BTCUSDT)
        interval: Candle interval
        start_date: Start date (optional)
        end_date: End date (optional)
        exchange: Exchange name ("upbit" or "binance")

    Returns:
        OHLCV DataFrame or None (on failure)
    """
    try:
        data_dir = get_data_dir(exchange)
        file_path = data_dir / parquet_filename(ticker, interval)

        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            return None

        # Load data
        df = pd.read_parquet(file_path)

        # Date filtering
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        logger.info(f"Loaded {ticker} {interval}: {len(df)} rows ({df.index[0]} ~ {df.index[-1]})")

        return df

    except Exception as e:
        logger.exception(f"Failed to load data for {ticker}: {e}")
        return None


def get_data_files(
    tickers: list[str],
    interval: Interval,
    exchange: str = "upbit",
) -> dict[str, Path]:
    """Create data file path dictionary for ticker list.

    Args:
        tickers: Ticker list
        interval: Candle interval
        exchange: Exchange name ("upbit" or "binance")

    Returns:
        {ticker: file_path} dictionary
    """
    data_files: dict[str, Path] = {}
    data_dir = get_data_dir(exchange)

    for ticker in tickers:
        file_path = data_dir / parquet_filename(ticker, interval)
        if file_path.exists():
            data_files[ticker] = file_path
        else:
            logger.warning(f"Data file not found for {ticker}: {file_path}")

    return data_files


def validate_data_availability(
    tickers: list[str],
    interval: Interval,
    exchange: str = "upbit",
) -> tuple[list[str], list[str]]:
    """Validate data availability.

    Args:
        tickers: Ticker list
        interval: Candle interval
        exchange: Exchange name ("upbit" or "binance")

    Returns:
        (available_tickers, missing_tickers) tuple
    """
    available: list[str] = []
    missing: list[str] = []
    data_dir = get_data_dir(exchange)

    for ticker in tickers:
        file_path = data_dir / parquet_filename(ticker, interval)
        if file_path.exists():
            available.append(ticker)
        else:
            missing.append(ticker)

    return available, missing


def load_multiple_tickers_parallel(
    tickers: list[str],
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
    max_workers: int = 4,
    exchange: str = "upbit",
) -> dict[str, pd.DataFrame]:
    """Load multiple ticker data in parallel.

    Use ThreadPoolExecutor to parallelize I/O bound operations.

    Args:
        tickers: Ticker list
        interval: Candle interval
        start_date: Start date (optional)
        end_date: End date (optional)
        max_workers: Maximum number of workers (default: 4)
        exchange: Exchange name ("upbit" or "binance")

    Returns:
        {ticker: DataFrame} dictionary
    """
    ticker_data: dict[str, pd.DataFrame] = {}

    def load_single_ticker(ticker: str) -> tuple[str, pd.DataFrame | None]:
        """Load single ticker."""
        try:
            df = load_ticker_data(ticker, interval, start_date, end_date, exchange)
            return ticker, df
        except Exception as e:
            logger.warning(f"Failed to load {ticker}: {e}")
            return ticker, None

    # Parallel loading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_ticker, ticker): ticker for ticker in tickers}

        for future in as_completed(futures):
            ticker, df = future.result()
            if df is not None:
                ticker_data[ticker] = df

    logger.info(f"Loaded {len(ticker_data)}/{len(tickers)} tickers in parallel")
    return ticker_data


@st.cache_data(ttl=3600)
def get_data_date_range(
    interval: Interval = "day",
    exchange: str = "upbit",
) -> tuple[date | None, date | None]:
    """Get the date range of available data.

    Scans all parquet files for the given interval and returns
    the earliest start date and latest end date.

    Args:
        interval: Candle interval
        exchange: Exchange name ("upbit" or "binance")

    Returns:
        (start_date, end_date) tuple, or (None, None) if no data
    """
    min_date: date | None = None
    max_date: date | None = None

    data_dir = get_data_dir(exchange)

    # Find all parquet files for this interval
    pattern = f"*_{interval}.parquet"
    files = list(data_dir.glob(pattern))

    if not files:
        logger.warning(f"No data files found for interval: {interval}")
        return None, None

    for file_path in files:
        try:
            df = pd.read_parquet(file_path)
            if len(df) > 0:
                # Get date from index
                if isinstance(df.index, pd.DatetimeIndex):
                    file_min = df.index.min().date()
                    file_max = df.index.max().date()
                elif "datetime" in df.columns:
                    file_min = pd.to_datetime(df["datetime"]).min().date()
                    file_max = pd.to_datetime(df["datetime"]).max().date()
                else:
                    continue

                if min_date is None or file_min < min_date:
                    min_date = file_min
                if max_date is None or file_max > max_date:
                    max_date = file_max
        except Exception as e:
            logger.debug(f"Could not read date range from {file_path}: {e}")
            continue

    logger.info(f"Data date range for {interval}: {min_date} ~ {max_date}")
    return min_date, max_date
