"""
Abstract base class for data sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataSource(ABC):
    """
    Abstract interface for data sources.

    Provides a unified interface for fetching historical and real-time market data.
    """

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "day",
        count: int = 200,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame | None:
        """
        Get OHLCV (candlestick) data.

        Args:
            symbol: Trading pair symbol (e.g., 'KRW-BTC')
            interval: Data interval (e.g., 'day', 'minute240', 'week')
            count: Number of candles to fetch (ignored if start_date/end_date provided)
            start_date: Start date for data range (optional)
            end_date: End date for data range (optional)

        Returns:
            DataFrame with OHLCV data (columns: open, high, low, close, volume)
            Returns None on error

        Raises:
            DataSourceError: If data fetch fails
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def save_ohlcv(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
        filepath: str | None = None,
    ) -> bool:
        """
        Save OHLCV data to storage.

        Args:
            symbol: Trading pair symbol
            interval: Data interval
            df: DataFrame with OHLCV data
            filepath: Optional custom filepath (uses default if None)

        Returns:
            True if save successful

        Raises:
            DataSourceError: If save fails
        """
        pass

    @abstractmethod
    def load_ohlcv(
        self,
        symbol: str,
        interval: str,
        filepath: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Load OHLCV data from storage.

        Args:
            symbol: Trading pair symbol
            interval: Data interval
            filepath: Optional custom filepath (uses default if None)

        Returns:
            DataFrame with OHLCV data or None if file not found

        Raises:
            DataSourceError: If load fails
        """
        pass

    @abstractmethod
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

        Raises:
            DataSourceError: If update fails
        """
        pass


# Exceptions are now defined in src.exceptions.data

__all__ = ["DataSource"]
