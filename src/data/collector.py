"""
Upbit candle data collector with incremental update support.

Fetches OHLCV data from Upbit API and stores in parquet format.
Supports incremental updates by fetching only new data since last update.
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import UPBIT_API_RATE_LIMIT_DELAY, UPBIT_DATA_DIR, parquet_filename
from src.data.collector_fetch import Interval, fetch_all_candles
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Re-export Interval for backward compatibility
__all__ = ["UpbitDataCollector", "Interval"]


class UpbitDataCollector:
    """Collects and manages Upbit OHLCV candle data."""

    def __init__(self, data_dir: Path | None = None) -> None:
        """
        Initialize the data collector.

        Args:
            data_dir: Directory to store parquet files. Defaults to data/upbit/
        """
        self.data_dir = data_dir or UPBIT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_parquet_path(self, ticker: str, interval: Interval) -> Path:
        """
        Generate parquet file path for given ticker and interval.

        Args:
            ticker: Upbit ticker (e.g., 'KRW-BTC')
            interval: Candle interval (e.g., 'day', 'minute240')

        Returns:
            Path to parquet file
        """
        return self.data_dir / parquet_filename(ticker, interval)

    def _load_existing_data(self, filepath: Path) -> pd.DataFrame | None:
        """
        Load existing parquet data if available.

        Args:
            filepath: Path to parquet file

        Returns:
            DataFrame with existing data or None if file doesn't exist
        """
        if filepath.exists():
            df = pd.read_parquet(filepath)
            df.index = pd.to_datetime(df.index)
            return df
        return None

    def collect(
        self,
        ticker: str,
        interval: Interval,
        full_refresh: bool = False,
    ) -> int:
        """
        Collect candle data for a ticker and interval.

        Performs incremental update if data already exists, unless full_refresh is True.

        Args:
            ticker: Upbit ticker (e.g., 'KRW-BTC')
            interval: Candle interval (e.g., 'day', 'minute240')
            full_refresh: If True, fetches all data ignoring existing

        Returns:
            Number of new candles added
        """
        filepath = self._get_parquet_path(ticker, interval)

        # Load existing data
        existing_df = None if full_refresh else self._load_existing_data(filepath)

        # Determine since datetime for incremental update
        since: datetime | None = None
        if existing_df is not None and not existing_df.empty:
            since = existing_df.index.max()
            logger.info(f"Incremental update for {ticker} ({interval}) since {since}")
        else:
            logger.info(f"Full collection for {ticker} ({interval})")

        # Fetch new candles
        new_df = fetch_all_candles(ticker, interval, since=since)

        if new_df is None or new_df.empty:
            logger.info(f"No new data for {ticker} ({interval})")
            return 0

        # Merge with existing data
        if existing_df is not None and not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
            combined_df = combined_df.sort_index()
        else:
            combined_df = new_df

        # Ensure index name
        combined_df.index.name = "datetime"

        # Save to parquet
        combined_df.to_parquet(filepath, engine="pyarrow")

        new_count = len(new_df)
        total_count = len(combined_df)
        logger.info(f"Saved {ticker} ({interval}): +{new_count} new, {total_count} total")

        return new_count

    def collect_multiple(
        self,
        tickers: list[str],
        intervals: list[Interval],
        full_refresh: bool = False,
    ) -> dict[str, int]:
        """
        Collect data for multiple tickers and intervals.

        Args:
            tickers: List of Upbit tickers
            intervals: List of intervals
            full_refresh: If True, fetches all data ignoring existing

        Returns:
            Dictionary with counts of new candles per ticker-interval
        """
        results: dict[str, int] = {}

        for ticker in tickers:
            for interval in intervals:
                key = f"{ticker}_{interval}"
                try:
                    count = self.collect(ticker, interval, full_refresh)
                    results[key] = count
                    time.sleep(UPBIT_API_RATE_LIMIT_DELAY)  # Rate limiting between pairs
                except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                    logger.error(f"Error collecting {key}: {e}", exc_info=True)
                    results[key] = -1

        return results
