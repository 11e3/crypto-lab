"""
Yahoo Finance data source.

Provides access to global stocks, ETFs, indices, and other assets.
Uses yfinance library for data fetching.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.data.sources.base import (
    AssetClass,
    AssetMetadata,
    DataSource,
    Exchange,
    OHLCVData,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YahooDataSource(DataSource):
    """
    Yahoo Finance data source.

    Fetches data for stocks, ETFs, indices from Yahoo Finance.
    Requires yfinance package: pip install yfinance

    Example:
        >>> source = YahooDataSource()
        >>> data = source.fetch_ohlcv("AAPL", "1d")
        >>> print(data.data.tail())
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        rate_limit_delay: float = 0.5,
    ) -> None:
        super().__init__(cache_dir, rate_limit_delay)

        # Interval mapping
        self._interval_map = {
            "1m": "1m",
            "2m": "2m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "1d": "1d",
            "5d": "5d",
            "1w": "1wk",
            "1M": "1mo",
            "3M": "3mo",
        }

    @property
    def exchange(self) -> Exchange:
        return Exchange.YAHOO

    @property
    def supported_intervals(self) -> list[str]:
        return list(self._interval_map.keys())

    def _get_yfinance(self):
        """Lazy import yfinance."""
        try:
            import yfinance as yf
            return yf
        except ImportError:
            raise ImportError(
                "yfinance is required for Yahoo data source. "
                "Install with: pip install yfinance"
            )

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVData:
        """Fetch OHLCV data from Yahoo Finance."""
        yf = self._get_yfinance()

        # Map interval
        yf_interval = self._interval_map.get(interval, interval)

        # Default date range
        if start is None:
            start = datetime.now() - timedelta(days=365 * 5)  # 5 years
        if end is None:
            end = datetime.now()

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=yf_interval,
            )
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return OHLCVData(
                symbol=symbol,
                data=pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
                interval=interval,
                exchange=self.exchange,
            )

        if df.empty:
            return OHLCVData(
                symbol=symbol,
                data=pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
                interval=interval,
                exchange=self.exchange,
            )

        # Normalize column names
        df.columns = df.columns.str.lower()

        # Select and rename columns
        df = df[["open", "high", "low", "close", "volume"]].copy()

        return OHLCVData(
            symbol=symbol,
            data=df,
            interval=interval,
            exchange=self.exchange,
            currency="USD",  # Usually USD, could be inferred
        )

    def get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """Get asset metadata from Yahoo Finance."""
        yf = self._get_yfinance()

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
            info = {}

        # Determine asset class
        quote_type = info.get("quoteType", "").upper()
        if quote_type == "ETF":
            asset_class = AssetClass.ETF
        elif quote_type == "INDEX":
            asset_class = AssetClass.INDEX
        elif quote_type == "CRYPTOCURRENCY":
            asset_class = AssetClass.CRYPTO
        elif quote_type == "CURRENCY":
            asset_class = AssetClass.FOREX
        else:
            asset_class = AssetClass.EQUITY

        return AssetMetadata(
            symbol=symbol,
            name=info.get("longName", info.get("shortName", symbol)),
            exchange=self.exchange,
            asset_class=asset_class,
            sector=info.get("sector"),
            industry=info.get("industry"),
            country=info.get("country"),
            currency=info.get("currency", "USD"),
            market_cap=info.get("marketCap"),
            avg_daily_volume=info.get("averageVolume"),
        )

    def list_symbols(
        self,
        asset_class: AssetClass | None = None,
    ) -> list[str]:
        """
        List common symbols.

        Yahoo Finance doesn't have a listing API, so we return
        common symbols by asset class.
        """
        common_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "DIS",
        ]

        common_etfs = [
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO",
            "BND", "TLT", "GLD", "SLV", "USO", "XLF", "XLK", "XLE",
        ]

        common_indices = [
            "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX",
            "^FTSE", "^N225", "^HSI",
        ]

        common_crypto = [
            "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
            "ADA-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
        ]

        if asset_class == AssetClass.EQUITY:
            return common_stocks
        elif asset_class == AssetClass.ETF:
            return common_etfs
        elif asset_class == AssetClass.INDEX:
            return common_indices
        elif asset_class == AssetClass.CRYPTO:
            return common_crypto
        else:
            return common_stocks + common_etfs + common_indices + common_crypto


__all__ = ["YahooDataSource"]
