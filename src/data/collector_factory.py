"""
Data collector factory for creating exchange-specific data collectors.

Supports multiple exchanges (Upbit, Binance, etc.) with factory pattern.
"""

from pathlib import Path
from typing import Literal

from src.config.loader import get_config
from src.data.collector import Interval, UpbitDataCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)

ExchangeName = Literal["upbit"]  # Can be extended: "binance", "coinbase", etc.


class DataCollectorFactory:
    """Factory for creating data collector instances."""

    @staticmethod
    def create(
        exchange_name: ExchangeName | None = None, data_dir: Path | None = None
    ) -> UpbitDataCollector:  # Return type will be Union when more exchanges are added
        """
        Create a data collector instance.

        Args:
            exchange_name: Name of exchange to create collector for (e.g., "upbit")
                          If None, uses configured default from settings
            data_dir: Directory for storing data files (optional)

        Returns:
            Data collector instance

        Raises:
            ValueError: If exchange_name is not supported
        """
        # Get exchange name from config if not provided
        if exchange_name is None:
            config = get_config()
            exchange_name = config.get("exchange.name", "upbit") or "upbit"

        # Normalize exchange name
        exchange_name = exchange_name.lower()

        # Create collector instance
        if exchange_name == "upbit":
            return UpbitDataCollector(data_dir=data_dir)
        else:
            raise ValueError(
                f"Unsupported exchange: {exchange_name}. "
                f"Supported exchanges: upbit (more exchanges coming soon)"
            )
