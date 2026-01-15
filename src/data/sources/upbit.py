"""
Upbit data source.

Wraps existing UpbitDataCollector with unified DataSource interface.
"""

from datetime import datetime
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


class UpbitDataSource(DataSource):
    """
    Upbit exchange data source.

    Fetches cryptocurrency data from Upbit Korean exchange.

    Example:
        >>> source = UpbitDataSource()
        >>> data = source.fetch_ohlcv("KRW-BTC", "day")
        >>> print(data.data.tail())
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        rate_limit_delay: float = 0.1,
    ) -> None:
        super().__init__(cache_dir, rate_limit_delay)

        # Interval mapping
        self._interval_map = {
            "1m": "minute1",
            "3m": "minute3",
            "5m": "minute5",
            "10m": "minute10",
            "15m": "minute15",
            "30m": "minute30",
            "1h": "minute60",
            "4h": "minute240",
            "1d": "day",
            "1w": "week",
            "1M": "month",
            # Also support original names
            "minute1": "minute1",
            "minute60": "minute60",
            "minute240": "minute240",
            "day": "day",
            "week": "week",
        }

    @property
    def exchange(self) -> Exchange:
        return Exchange.UPBIT

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
        """Fetch OHLCV data from Upbit."""
        import time
        import requests

        # Map interval
        upbit_interval = self._interval_map.get(interval, interval)

        # Build URL based on interval
        if upbit_interval.startswith("minute"):
            minutes = upbit_interval.replace("minute", "")
            url = f"https://api.upbit.com/v1/candles/minutes/{minutes}"
        elif upbit_interval == "day":
            url = "https://api.upbit.com/v1/candles/days"
        elif upbit_interval == "week":
            url = "https://api.upbit.com/v1/candles/weeks"
        elif upbit_interval == "month":
            url = "https://api.upbit.com/v1/candles/months"
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Fetch data
        all_data = []
        to_date = end or datetime.now()
        count = 200  # Max per request

        while True:
            params = {
                "market": symbol,
                "count": count,
                "to": to_date.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Upbit API error: {e}")
                break

            if not data:
                break

            all_data.extend(data)

            # Check if we've gone past start date
            oldest = datetime.fromisoformat(data[-1]["candle_date_time_utc"])
            if start and oldest <= start:
                break

            # Pagination
            to_date = oldest
            time.sleep(self.rate_limit_delay)

            # Limit iterations
            if len(all_data) >= 10000:
                break

        if not all_data:
            return OHLCVData(
                symbol=symbol,
                data=pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
                interval=interval,
                exchange=self.exchange,
                currency="KRW",
            )

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["candle_date_time_utc"])
        df = df.set_index("timestamp").sort_index()

        df = df.rename(columns={
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
        })

        df = df[["open", "high", "low", "close", "volume"]]

        # Filter by date range
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        return OHLCVData(
            symbol=symbol,
            data=df,
            interval=interval,
            exchange=self.exchange,
            currency="KRW",
        )

    def get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """Get asset metadata from Upbit."""
        import requests

        try:
            response = requests.get(
                "https://api.upbit.com/v1/market/all",
                params={"isDetails": "true"},
            )
            response.raise_for_status()
            markets = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch Upbit markets: {e}")
            markets = []

        for market in markets:
            if market["market"] == symbol:
                return AssetMetadata(
                    symbol=symbol,
                    name=market.get("korean_name", symbol),
                    exchange=self.exchange,
                    asset_class=AssetClass.CRYPTO,
                    currency="KRW",
                )

        # Default metadata
        return AssetMetadata(
            symbol=symbol,
            name=symbol,
            exchange=self.exchange,
            asset_class=AssetClass.CRYPTO,
            currency="KRW",
        )

    def list_symbols(
        self,
        asset_class: AssetClass | None = None,
    ) -> list[str]:
        """List available symbols on Upbit."""
        import requests

        try:
            response = requests.get("https://api.upbit.com/v1/market/all")
            response.raise_for_status()
            markets = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch Upbit markets: {e}")
            return []

        # Filter for KRW markets (main trading pairs)
        symbols = [m["market"] for m in markets if m["market"].startswith("KRW-")]

        return symbols


__all__ = ["UpbitDataSource"]
