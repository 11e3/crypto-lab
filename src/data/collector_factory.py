"""
Data collector factory for creating exchange-specific data collectors.

Supports multiple exchanges (Upbit, Binance, etc.) with factory pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.config.loader import get_config
from src.data.collector import UpbitDataCollector
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.data.binance_collector import BinanceDataCollector

logger = get_logger(__name__)

ExchangeName = Literal["upbit", "binance"]


class DataCollectorFactory:
    """Factory for creating data collector instances."""

    @staticmethod
    def create(
        exchange_name: ExchangeName | None = None, data_dir: Path | None = None
    ) -> UpbitDataCollector | BinanceDataCollector:
        """
        Create a data collector instance.

        Args:
            exchange_name: Name of exchange to create collector for (e.g., "upbit", "binance")
                          If None, uses configured default from settings
            data_dir: Directory for storing data files (optional)

        Returns:
            Data collector instance

        Raises:
            ValueError: If exchange_name is not supported
        """
        # Get exchange name from config if not provided
        exchange_name_resolved: str
        if exchange_name is None:
            config = get_config()
            config_name: str | None = config.get("exchange.name", "upbit")
            exchange_name_resolved = config_name if config_name else "upbit"
        else:
            exchange_name_resolved = exchange_name

        # Normalize exchange name
        exchange_name_lower: str = exchange_name_resolved.lower()

        # Create collector instance
        if exchange_name_lower == "upbit":
            return UpbitDataCollector(data_dir=data_dir)
        elif exchange_name_lower == "binance":
            from src.data.binance_collector import BinanceDataCollector

            return BinanceDataCollector(data_dir=data_dir)
        else:
            raise ValueError(
                f"Unsupported exchange: {exchange_name_lower}. Supported exchanges: upbit, binance"
            )
