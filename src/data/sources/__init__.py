"""
Multi-asset data sources.

Provides unified interface for fetching market data from:
- Upbit (Korean crypto exchange)
- Yahoo Finance (Global stocks, ETFs, indices)
- Binance (Global crypto)
- Custom data sources
"""

from src.data.sources.base import (
    DataSource,
    AssetMetadata,
    AssetClass,
    OHLCVData,
)
from src.data.sources.upbit import UpbitDataSource
from src.data.sources.yahoo import YahooDataSource
from src.data.sources.binance import BinanceDataSource
from src.data.sources.universe import UniverseManager, Universe

__all__ = [
    # Base
    "DataSource",
    "AssetMetadata",
    "AssetClass",
    "OHLCVData",
    # Sources
    "UpbitDataSource",
    "YahooDataSource",
    "BinanceDataSource",
    # Universe
    "UniverseManager",
    "Universe",
]
