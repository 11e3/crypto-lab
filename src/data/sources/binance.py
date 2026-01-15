"""
Binance data source.

Provides access to global cryptocurrency data from Binance.
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


class BinanceDataSource(DataSource):
    """
    Binance exchange data source.

    Fetches cryptocurrency data from Binance.
    Uses public API (no authentication required for market data).

    Example:
        >>> source = BinanceDataSource()
        >>> data = source.fetch_ohlcv("BTCUSDT", "1d")
        >>> print(data.data.tail())
    """

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        rate_limit_delay: float = 0.1,
    ) -> None:
        super().__init__(cache_dir, rate_limit_delay)

        # Interval mapping
        self._interval_map = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1M": "1M",
        }

    @property
    def exchange(self) -> Exchange:
        return Exchange.BINANCE

    @property
    def supported_intervals(self) -> list[str]:
        return list(self._interval_map.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVData:
        """Fetch OHLCV data from Binance."""
        import time
        import requests

        # Map interval
        binance_interval = self._interval_map.get(interval, interval)

        # Default date range
        if start is None:
            start = datetime.now() - timedelta(days=365)
        if end is None:
            end = datetime.now()

        # Binance uses milliseconds
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_data = []
        limit = 1000  # Max per request

        current_start = start_ms

        while current_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": binance_interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": limit,
            }

            try:
                response = requests.get(
                    f"{self.BASE_URL}/klines",
                    params=params,
                )
                response.raise_for_status()
                klines = response.json()
            except Exception as e:
                logger.error(f"Binance API error: {e}")
                break

            if not klines:
                break

            all_data.extend(klines)

            # Pagination - next batch starts after last candle
            current_start = klines[-1][0] + 1
            time.sleep(self.rate_limit_delay)

            # Limit iterations
            if len(all_data) >= 100000:
                break

        if not all_data:
            return OHLCVData(
                symbol=symbol,
                data=pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
                interval=interval,
                exchange=self.exchange,
            )

        # Convert to DataFrame
        # Kline format: [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

        # Convert to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["open", "high", "low", "close", "volume"]]

        return OHLCVData(
            symbol=symbol,
            data=df,
            interval=interval,
            exchange=self.exchange,
            currency="USDT" if "USDT" in symbol else "USD",
        )

    def get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """Get asset metadata from Binance."""
        import requests

        try:
            response = requests.get(
                f"{self.BASE_URL}/exchangeInfo",
                params={"symbol": symbol},
            )
            response.raise_for_status()
            info = response.json()
        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
            info = {}

        symbols = info.get("symbols", [])
        symbol_info = symbols[0] if symbols else {}

        # Get 24h ticker for volume
        try:
            ticker = requests.get(
                f"{self.BASE_URL}/ticker/24hr",
                params={"symbol": symbol},
            ).json()
        except Exception:
            ticker = {}

        return AssetMetadata(
            symbol=symbol,
            name=symbol_info.get("baseAsset", symbol),
            exchange=self.exchange,
            asset_class=AssetClass.CRYPTO,
            currency=symbol_info.get("quoteAsset", "USDT"),
            avg_daily_volume=float(ticker.get("volume", 0)) if ticker else None,
            min_trade_size=float(symbol_info.get("filters", [{}])[0].get("minQty", 0)),
        )

    def list_symbols(
        self,
        asset_class: AssetClass | None = None,
    ) -> list[str]:
        """List available symbols on Binance."""
        import requests

        try:
            response = requests.get(f"{self.BASE_URL}/exchangeInfo")
            response.raise_for_status()
            info = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch Binance symbols: {e}")
            return []

        symbols = []
        for s in info.get("symbols", []):
            # Filter for USDT pairs (most liquid)
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING":
                symbols.append(s["symbol"])

        return symbols


__all__ = ["BinanceDataSource"]
