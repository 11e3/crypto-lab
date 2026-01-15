"""
Base classes for data sources.

Provides abstract DataSource class and common data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import pandas as pd


class AssetClass(str, Enum):
    """Asset class classification."""

    CRYPTO = "crypto"
    EQUITY = "equity"
    ETF = "etf"
    FOREX = "forex"
    COMMODITY = "commodity"
    FIXED_INCOME = "fixed_income"
    INDEX = "index"


class Exchange(str, Enum):
    """Supported exchanges."""

    UPBIT = "upbit"
    BINANCE = "binance"
    YAHOO = "yahoo"  # Pseudo-exchange for Yahoo Finance
    KIS = "kis"  # Korea Investment & Securities
    POLYGON = "polygon"


@dataclass
class AssetMetadata:
    """Metadata for an asset."""

    # Identification
    symbol: str  # Ticker symbol
    name: str  # Full name
    exchange: Exchange

    # Classification
    asset_class: AssetClass
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    currency: str = "USD"

    # Market data
    market_cap: float | None = None
    avg_daily_volume: float | None = None

    # Trading info
    min_trade_size: float = 0.0
    tick_size: float = 0.01
    trading_hours: str | None = None

    # Timestamps
    listing_date: datetime | None = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class OHLCVData:
    """OHLCV data container."""

    symbol: str
    data: pd.DataFrame  # Columns: open, high, low, close, volume
    interval: str  # e.g., "1d", "1h", "15m"
    exchange: Exchange
    currency: str = "USD"
    start_date: datetime | None = None
    end_date: datetime | None = None

    def __post_init__(self):
        # Ensure required columns
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(self.data.columns)):
            missing = required - set(self.data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Set date range
        if len(self.data) > 0:
            self.start_date = self.data.index.min()
            self.end_date = self.data.index.max()

    @property
    def returns(self) -> pd.Series:
        """Calculate simple returns."""
        return self.data["close"].pct_change()

    @property
    def log_returns(self) -> pd.Series:
        """Calculate log returns."""
        import numpy as np
        return np.log(self.data["close"] / self.data["close"].shift(1))


class DataSource(ABC):
    """
    Abstract base class for data sources.

    Provides unified interface for fetching market data.

    Example:
        >>> source = UpbitDataSource()
        >>> data = source.fetch_ohlcv("BTC-USDT", "1d", start, end)
        >>> metadata = source.get_asset_metadata("BTC-USDT")
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        rate_limit_delay: float = 0.1,
    ) -> None:
        """
        Initialize data source.

        Args:
            cache_dir: Directory for caching data
            rate_limit_delay: Delay between API calls
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rate_limit_delay = rate_limit_delay

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def exchange(self) -> Exchange:
        """Return exchange identifier."""
        pass

    @property
    @abstractmethod
    def supported_intervals(self) -> list[str]:
        """Return list of supported time intervals."""
        pass

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVData:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Asset symbol
            interval: Time interval (e.g., "1d", "1h")
            start: Start date
            end: End date

        Returns:
            OHLCVData container
        """
        pass

    @abstractmethod
    def get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """
        Get metadata for an asset.

        Args:
            symbol: Asset symbol

        Returns:
            AssetMetadata object
        """
        pass

    @abstractmethod
    def list_symbols(
        self,
        asset_class: AssetClass | None = None,
    ) -> list[str]:
        """
        List available symbols.

        Args:
            asset_class: Filter by asset class

        Returns:
            List of symbol strings
        """
        pass

    def fetch_multiple(
        self,
        symbols: list[str],
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, OHLCVData]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of symbols
            interval: Time interval
            start: Start date
            end: End date

        Returns:
            Dict of symbol -> OHLCVData
        """
        import time

        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_ohlcv(symbol, interval, start, end)
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                from src.utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Failed to fetch {symbol}: {e}")

        return results

    def to_price_dataframe(
        self,
        data: dict[str, OHLCVData],
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Convert multiple OHLCVData to single price DataFrame.

        Args:
            data: Dict of symbol -> OHLCVData
            price_col: Which price column to use

        Returns:
            DataFrame with symbols as columns
        """
        prices = {}
        for symbol, ohlcv in data.items():
            prices[symbol] = ohlcv.data[price_col]

        return pd.DataFrame(prices)

    def _cache_path(self, symbol: str, interval: str) -> Path | None:
        """Get cache file path."""
        if not self.cache_dir:
            return None
        safe_symbol = symbol.replace("/", "-").replace(":", "-")
        return self.cache_dir / f"{self.exchange.value}_{safe_symbol}_{interval}.parquet"

    def _load_from_cache(
        self,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame | None:
        """Load data from cache if available."""
        cache_path = self._cache_path(symbol, interval)
        if cache_path and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass
        return None

    def _save_to_cache(
        self,
        symbol: str,
        interval: str,
        data: pd.DataFrame,
    ) -> None:
        """Save data to cache."""
        cache_path = self._cache_path(symbol, interval)
        if cache_path:
            try:
                data.to_parquet(cache_path)
            except Exception:
                pass


__all__ = [
    "DataSource",
    "AssetMetadata",
    "AssetClass",
    "Exchange",
    "OHLCVData",
]
