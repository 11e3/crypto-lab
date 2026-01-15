"""
Universe management for multi-asset portfolios.

Provides:
- Universe definition and filtering
- Cross-asset data aggregation
- Universe membership tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

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


@dataclass
class UniverseFilter:
    """Filter criteria for universe membership."""

    # Asset class filter
    asset_classes: list[AssetClass] | None = None

    # Exchange filter
    exchanges: list[Exchange] | None = None

    # Liquidity filter
    min_avg_volume: float | None = None
    min_market_cap: float | None = None

    # Price filter
    min_price: float | None = None
    max_price: float | None = None

    # Data quality
    min_history_days: int = 60

    # Custom filter function
    custom_filter: Callable[[AssetMetadata], bool] | None = None


@dataclass
class Universe:
    """Investment universe definition."""

    name: str
    symbols: list[str] = field(default_factory=list)
    sources: dict[str, DataSource] = field(default_factory=dict)  # symbol -> source
    metadata: dict[str, AssetMetadata] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.symbols

    def add_symbol(
        self,
        symbol: str,
        source: DataSource,
        metadata: AssetMetadata | None = None,
    ) -> None:
        """Add symbol to universe."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        self.sources[symbol] = source
        if metadata:
            self.metadata[symbol] = metadata

    def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from universe."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
        self.sources.pop(symbol, None)
        self.metadata.pop(symbol, None)

    def get_source(self, symbol: str) -> DataSource | None:
        """Get data source for symbol."""
        return self.sources.get(symbol)


class UniverseManager:
    """
    Manages investment universes across multiple data sources.

    Provides unified access to multi-asset data.

    Example:
        >>> from src.data.sources import UpbitDataSource, YahooDataSource
        >>>
        >>> manager = UniverseManager()
        >>> manager.add_source("upbit", UpbitDataSource())
        >>> manager.add_source("yahoo", YahooDataSource())
        >>>
        >>> universe = manager.create_universe(
        ...     name="crypto_global",
        ...     sources=["upbit", "binance"],
        ...     filter_criteria=UniverseFilter(min_avg_volume=1000000),
        ... )
        >>> prices = manager.fetch_prices(universe, "1d")
    """

    def __init__(self) -> None:
        """Initialize universe manager."""
        self.sources: dict[str, DataSource] = {}
        self.universes: dict[str, Universe] = {}

    def add_source(self, name: str, source: DataSource) -> None:
        """Add a data source."""
        self.sources[name] = source
        logger.info(f"Added data source: {name} ({source.exchange.value})")

    def remove_source(self, name: str) -> None:
        """Remove a data source."""
        self.sources.pop(name, None)

    def create_universe(
        self,
        name: str,
        sources: list[str] | None = None,
        symbols: list[str] | None = None,
        filter_criteria: UniverseFilter | None = None,
    ) -> Universe:
        """
        Create an investment universe.

        Args:
            name: Universe name
            sources: List of source names to include
            symbols: Explicit list of symbols (optional)
            filter_criteria: Filter criteria for automatic selection

        Returns:
            Universe object
        """
        universe = Universe(name=name)

        if symbols:
            # Use explicit symbols
            for symbol in symbols:
                # Find source for symbol
                for source_name, source in self.sources.items():
                    if sources and source_name not in sources:
                        continue
                    try:
                        metadata = source.get_asset_metadata(symbol)
                        universe.add_symbol(symbol, source, metadata)
                        break
                    except Exception:
                        continue

        else:
            # Auto-discover from sources
            source_list = sources or list(self.sources.keys())

            for source_name in source_list:
                if source_name not in self.sources:
                    continue

                source = self.sources[source_name]

                try:
                    symbols = source.list_symbols()
                except Exception as e:
                    logger.warning(f"Failed to list symbols from {source_name}: {e}")
                    continue

                for symbol in symbols:
                    try:
                        metadata = source.get_asset_metadata(symbol)

                        # Apply filters
                        if filter_criteria and not self._passes_filter(
                            metadata, filter_criteria
                        ):
                            continue

                        universe.add_symbol(symbol, source, metadata)

                    except Exception as e:
                        logger.debug(f"Failed to get metadata for {symbol}: {e}")
                        continue

        self.universes[name] = universe
        logger.info(f"Created universe '{name}' with {len(universe)} symbols")

        return universe

    def fetch_prices(
        self,
        universe: Universe | str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Fetch prices for entire universe.

        Args:
            universe: Universe object or name
            interval: Time interval
            start: Start date
            end: End date
            price_col: Which price column to use

        Returns:
            DataFrame with symbols as columns
        """
        if isinstance(universe, str):
            universe = self.universes[universe]

        prices = {}

        for symbol in universe.symbols:
            source = universe.get_source(symbol)
            if not source:
                continue

            try:
                ohlcv = source.fetch_ohlcv(symbol, interval, start, end)
                if not ohlcv.data.empty:
                    prices[symbol] = ohlcv.data[price_col]
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue

        if not prices:
            return pd.DataFrame()

        return pd.DataFrame(prices)

    def fetch_ohlcv(
        self,
        universe: Universe | str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, OHLCVData]:
        """
        Fetch full OHLCV data for universe.

        Args:
            universe: Universe object or name
            interval: Time interval
            start: Start date
            end: End date

        Returns:
            Dict of symbol -> OHLCVData
        """
        if isinstance(universe, str):
            universe = self.universes[universe]

        data = {}

        for symbol in universe.symbols:
            source = universe.get_source(symbol)
            if not source:
                continue

            try:
                ohlcv = source.fetch_ohlcv(symbol, interval, start, end)
                if not ohlcv.data.empty:
                    data[symbol] = ohlcv
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue

        return data

    def get_metadata(
        self,
        universe: Universe | str,
    ) -> pd.DataFrame:
        """
        Get metadata for all symbols in universe.

        Args:
            universe: Universe object or name

        Returns:
            DataFrame with metadata
        """
        if isinstance(universe, str):
            universe = self.universes[universe]

        records = []
        for symbol in universe.symbols:
            meta = universe.metadata.get(symbol)
            if meta:
                records.append({
                    "symbol": meta.symbol,
                    "name": meta.name,
                    "asset_class": meta.asset_class.value,
                    "exchange": meta.exchange.value,
                    "sector": meta.sector,
                    "currency": meta.currency,
                    "market_cap": meta.market_cap,
                    "avg_volume": meta.avg_daily_volume,
                })

        return pd.DataFrame(records)

    def filter_universe(
        self,
        universe: Universe | str,
        filter_criteria: UniverseFilter,
    ) -> Universe:
        """
        Apply filter to existing universe.

        Args:
            universe: Universe to filter
            filter_criteria: Filter criteria

        Returns:
            New filtered universe
        """
        if isinstance(universe, str):
            universe = self.universes[universe]

        filtered = Universe(name=f"{universe.name}_filtered")

        for symbol in universe.symbols:
            metadata = universe.metadata.get(symbol)
            if metadata and self._passes_filter(metadata, filter_criteria):
                filtered.add_symbol(
                    symbol,
                    universe.sources[symbol],
                    metadata,
                )

        return filtered

    def _passes_filter(
        self,
        metadata: AssetMetadata,
        criteria: UniverseFilter,
    ) -> bool:
        """Check if asset passes filter criteria."""
        # Asset class filter
        if criteria.asset_classes:
            if metadata.asset_class not in criteria.asset_classes:
                return False

        # Exchange filter
        if criteria.exchanges:
            if metadata.exchange not in criteria.exchanges:
                return False

        # Volume filter
        if criteria.min_avg_volume:
            if metadata.avg_daily_volume is None:
                return False
            if metadata.avg_daily_volume < criteria.min_avg_volume:
                return False

        # Market cap filter
        if criteria.min_market_cap:
            if metadata.market_cap is None:
                return False
            if metadata.market_cap < criteria.min_market_cap:
                return False

        # Custom filter
        if criteria.custom_filter:
            if not criteria.custom_filter(metadata):
                return False

        return True


def create_crypto_universe() -> Universe:
    """Create default crypto universe with Upbit and Binance."""
    from src.data.sources.upbit import UpbitDataSource
    from src.data.sources.binance import BinanceDataSource

    manager = UniverseManager()
    manager.add_source("upbit", UpbitDataSource())
    manager.add_source("binance", BinanceDataSource())

    return manager.create_universe(
        name="crypto_all",
        sources=["upbit", "binance"],
    )


def create_global_universe() -> Universe:
    """Create global multi-asset universe."""
    from src.data.sources.yahoo import YahooDataSource

    manager = UniverseManager()
    manager.add_source("yahoo", YahooDataSource())

    return manager.create_universe(
        name="global_mixed",
        sources=["yahoo"],
    )


__all__ = [
    "Universe",
    "UniverseFilter",
    "UniverseManager",
    "create_crypto_universe",
    "create_global_universe",
]
