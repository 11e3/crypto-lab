"""Binance data source implementation."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.constants import BINANCE_DATA_DIR, BINANCE_MAX_CANDLES_PER_REQUEST, parquet_filename
from src.data.base import DataSource
from src.data.binance_source_utils import calculate_binance_update_count, merge_ohlcv_data
from src.exceptions.data import (
    DataSourceConnectionError,
    DataSourceError,
    DataSourceNotFoundError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BinanceDataSource(DataSource):
    """Binance data source implementation.

    Fetches data from Binance API and manages local storage.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize Binance data source.

        Args:
            data_dir: Directory for storing data files (defaults to BINANCE_DATA_DIR)
        """
        self.data_dir = data_dir or BINANCE_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_exchange(self) -> Any:
        """Get ccxt Binance exchange instance."""
        import ccxt

        return ccxt.binance({"enableRateLimit": True})

    def _get_filepath(self, symbol: str, interval: str) -> Path:
        """Get filepath for storing OHLCV data.

        Args:
            symbol: Trading pair symbol
            interval: Data interval

        Returns:
            Path to data file
        """
        return self.data_dir / parquet_filename(symbol, interval)

    def _parse_candles_to_dataframe(
        self,
        ohlcv: list[list[float]],
        symbol: str,
        interval: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pd.DataFrame | None:
        """Convert raw OHLCV candles to a filtered, datetime-indexed DataFrame."""
        if not ohlcv:
            logger.warning(f"No OHLCV data for {symbol} {interval}")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df = df.set_index("timestamp")
        df.index.name = "datetime"

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df if len(df) > 0 else None

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        count: int = 200,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame | None:
        """Get OHLCV data from Binance API.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'BTC/USDT')
            interval: Data interval (e.g., '1d', '4h', '1m')
            count: Number of candles to fetch
            start_date: Start date for data range (optional)
            end_date: End date for data range (optional)

        Returns:
            DataFrame with OHLCV data or None on error
        """
        try:
            exchange = self._get_exchange()
            since_ms: int | None = int(start_date.timestamp() * 1000) if start_date else None
            ohlcv: list[list[float]] = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=interval,
                since=since_ms,
                limit=min(count, BINANCE_MAX_CANDLES_PER_REQUEST),
            )
            return self._parse_candles_to_dataframe(ohlcv, symbol, interval, start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}", exc_info=True)
            raise DataSourceConnectionError(f"Failed to fetch data: {e}") from e

    def get_current_price(self, symbol: str) -> float:
        """Get current market price.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current market price

        Raises:
            DataSourceError: If price fetch fails
        """
        try:
            exchange = self._get_exchange()
            ticker: dict[str, float] = exchange.fetch_ticker(symbol)
            price = ticker.get("last")
            if price is None:
                raise DataSourceNotFoundError(f"No price data for {symbol}")
            return float(price)
        except DataSourceNotFoundError:
            raise
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
        """Save OHLCV data to parquet file.

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
        """Load OHLCV data from parquet file.

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

    def _fetch_incremental_candles(
        self,
        symbol: str,
        interval: str,
        existing_df: pd.DataFrame,
        filepath: str | None,
    ) -> pd.DataFrame:
        """Fetch candles since last update, merge with existing data, and save."""
        latest_timestamp = existing_df.index.max()
        count = calculate_binance_update_count(latest_timestamp, interval)

        logger.info(f"Fetching new data for {symbol} {interval} (since {latest_timestamp})")
        new_df = self.get_ohlcv(symbol, interval, count=count)

        if new_df is None or len(new_df) == 0:
            logger.warning(f"No new data for {symbol} {interval}")
            return existing_df

        updated_df, new_count = merge_ohlcv_data(existing_df, new_df, latest_timestamp)

        if new_count == 0:
            logger.info(f"No new data to add for {symbol} {interval}")
            return existing_df

        self.save_ohlcv(symbol, interval, updated_df, filepath)
        logger.info(f"Updated {symbol} {interval}: +{new_count} new, {len(updated_df)} total")
        return updated_df

    def update_ohlcv(
        self,
        symbol: str,
        interval: str,
        filepath: str | None = None,
    ) -> pd.DataFrame | None:
        """Incrementally update OHLCV data.

        Args:
            symbol: Trading pair symbol
            interval: Data interval
            filepath: Optional custom filepath (uses default if None)

        Returns:
            Updated DataFrame with OHLCV data or None on error
        """
        try:
            existing_df = self.load_ohlcv(symbol, interval, filepath)

            if existing_df is None or len(existing_df) == 0:
                logger.info(f"No existing data for {symbol} {interval}, fetching full dataset")
                df = self.get_ohlcv(symbol, interval, count=BINANCE_MAX_CANDLES_PER_REQUEST)
                if df is not None:
                    self.save_ohlcv(symbol, interval, df, filepath)
                return df

            return self._fetch_incremental_candles(symbol, interval, existing_df, filepath)
        except Exception as e:
            logger.error(f"Error updating OHLCV data for {symbol}: {e}", exc_info=True)
            return None
