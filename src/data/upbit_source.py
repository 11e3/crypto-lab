"""
Upbit data source implementation.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyupbit

from src.config.constants import RAW_DATA_DIR, parquet_filename
from src.data.base import DataSource
from src.data.upbit_source_utils import calculate_update_count, merge_ohlcv_data
from src.exceptions.data import (
    DataSourceConnectionError,
    DataSourceError,
    DataSourceNotFoundError,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class UpbitDataSource(DataSource):
    """
    Upbit data source implementation.

    Fetches data from Upbit API and manages local storage.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """
        Initialize Upbit data source.

        Args:
            data_dir: Directory for storing data files (defaults to RAW_DATA_DIR)
        """
        self.data_dir = data_dir or RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, symbol: str, interval: str) -> Path:
        """
        Get filepath for storing OHLCV data.

        Args:
            symbol: Trading pair symbol
            interval: Data interval

        Returns:
            Path to data file
        """
        return self.data_dir / parquet_filename(symbol, interval)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "day",
        count: int = 200,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame | None:
        """
        Get OHLCV data from Upbit API.

        Args:
            symbol: Trading pair symbol (e.g., 'KRW-BTC')
            interval: Data interval (e.g., 'day', 'minute240', 'week')
            count: Number of candles to fetch (ignored if start_date/end_date provided)
            start_date: Start date for data range (optional)
            end_date: End date for data range (optional)

        Returns:
            DataFrame with OHLCV data or None on error
        """
        try:
            if start_date and end_date:
                # Fetch data for date range
                df = pyupbit.get_ohlcv(
                    symbol,
                    interval=interval,
                    to=end_date.strftime("%Y%m%d"),
                    count=count,
                )
                if df is not None and len(df) > 0:
                    # Filter to date range
                    df = df[df.index >= start_date]
                    df = df[df.index <= end_date]
            else:
                # Fetch recent data
                df = pyupbit.get_ohlcv(symbol, interval=interval, count=count)

            if df is None or len(df) == 0:
                logger.warning(f"No OHLCV data for {symbol} {interval}")
                return None

            # Explicit conversion from Any to DataFrame
            return pd.DataFrame(df)
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}", exc_info=True)
            raise DataSourceConnectionError(f"Failed to fetch data: {e}") from e

    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current market price

        Raises:
            DataSourceError: If price fetch fails
        """
        try:
            price = pyupbit.get_current_price(symbol)
            if price is None:
                raise DataSourceNotFoundError(f"No price data for {symbol}")
            return float(price)
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}", exc_info=True)
            raise DataSourceError(f"Failed to get price: {e}") from e

    def save_ohlcv(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
        filepath: Path | str | None = None,
    ) -> bool:
        """
        Save OHLCV data to parquet file.

        Args:
            symbol: Trading pair symbol
            interval: Data interval
            df: DataFrame with OHLCV data
            filepath: Optional custom filepath (uses default if None)

        Returns:
            True if save successful
        """
        try:
            if filepath is None:
                file_path = self._get_filepath(symbol, interval)
            else:
                file_path = Path(filepath) if isinstance(filepath, str) else filepath

            file_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(file_path, index=True)
            logger.info(f"Saved OHLCV data to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving OHLCV data: {e}", exc_info=True)
            return False

    def load_ohlcv(
        self,
        symbol: str,
        interval: str,
        filepath: Path | str | None = None,
    ) -> pd.DataFrame | None:
        """
        Load OHLCV data from parquet file.

        Args:
            symbol: Trading pair symbol
            interval: Data interval
            filepath: Optional custom filepath (uses default if None)

        Returns:
            DataFrame with OHLCV data or None if file not found
        """
        try:
            if filepath is None:
                file_path = self._get_filepath(symbol, interval)
            else:
                file_path = Path(filepath) if isinstance(filepath, str) else filepath

            if not file_path.exists():
                logger.debug(f"Data file not found: {file_path}")
                return None

            df = pd.read_parquet(file_path)
            logger.debug(f"Loaded OHLCV data from {file_path}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading OHLCV data: {e}", exc_info=True)
            return None

    def update_ohlcv(
        self,
        symbol: str,
        interval: str,
        filepath: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Incrementally update OHLCV data.

        Fetches new data since last update and merges with existing data.

        Args:
            symbol: Trading pair symbol
            interval: Data interval
            filepath: Optional custom filepath (uses default if None)

        Returns:
            Updated DataFrame with OHLCV data or None on error
        """
        try:
            # Load existing data
            existing_df = self.load_ohlcv(symbol, interval, filepath)

            if existing_df is None or len(existing_df) == 0:
                # No existing data, fetch full dataset
                logger.info(f"No existing data for {symbol} {interval}, fetching full dataset")
                df = self.get_ohlcv(symbol, interval, count=200)
                if df is not None:
                    self.save_ohlcv(symbol, interval, df, filepath)
                return df

            # Get latest timestamp and calculate fetch count
            latest_timestamp = existing_df.index.max()
            count = calculate_update_count(latest_timestamp, interval)

            # Fetch new data
            logger.info(f"Fetching new data for {symbol} {interval} (since {latest_timestamp})")
            new_df = self.get_ohlcv(symbol, interval, count=count)

            if new_df is None or len(new_df) == 0:
                logger.warning(f"No new data for {symbol} {interval}")
                return existing_df

            # Merge data
            updated_df, new_count = merge_ohlcv_data(existing_df, new_df, latest_timestamp)

            if new_count == 0:
                logger.info(f"No new data to add for {symbol} {interval}")
                return existing_df

            # Save updated data
            self.save_ohlcv(symbol, interval, updated_df, filepath)
            logger.info(f"Updated {symbol} {interval}: +{new_count} new, {len(updated_df)} total")

            return updated_df
        except Exception as e:
            logger.error(f"Error updating OHLCV data for {symbol}: {e}", exc_info=True)
            return None
