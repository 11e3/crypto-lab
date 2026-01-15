"""
OHLCV data loading for signal generation.

Separates data fetching responsibility from signal detection (SRP).
"""

import pandas as pd

from src.exchange import MarketDataService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalDataLoader:
    """
    Loads OHLCV data for signal generation.

    Handles data fetching and validation, separate from signal logic.
    """

    def __init__(
        self,
        exchange: MarketDataService,
        min_data_points: int = 10,
    ) -> None:
        """
        Initialize data loader.

        Args:
            exchange: Service implementing MarketDataService protocol
            min_data_points: Minimum data points required for signal generation
        """
        self.exchange = exchange
        self.min_data_points = min_data_points

    def get_ohlcv(
        self,
        ticker: str,
        interval: str = "day",
        count: int | None = None,
    ) -> pd.DataFrame | None:
        """
        Get OHLCV data for signal generation.

        Args:
            ticker: Trading pair symbol
            interval: Data interval (default: "day")
            count: Number of candles to fetch (uses min_data_points if None)

        Returns:
            DataFrame with OHLCV data or None on error
        """
        if count is None:
            count = max(self.min_data_points * 2, 20)

        try:
            df = self.exchange.get_ohlcv(ticker, interval=interval, count=count)
            if df is None or len(df) < self.min_data_points:
                logger.warning(
                    f"Insufficient data for {ticker}: "
                    f"got {len(df) if df is not None else 0} rows, need {self.min_data_points}"
                )
                return None
            return df
        except Exception as e:
            logger.error(f"Error getting OHLCV for {ticker}: {e}", exc_info=True)
            return None
