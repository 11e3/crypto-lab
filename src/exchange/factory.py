"""
Exchange factory for creating exchange instances.

Supports multiple exchanges (Upbit, Binance, etc.) with factory pattern.
"""

import contextlib
from typing import Literal

from src.config.loader import get_config
from src.exchange.base import Exchange
from src.exchange.upbit import UpbitExchange
from src.utils.logger import get_logger

logger = get_logger(__name__)

ExchangeName = Literal["upbit"]  # Can be extended: "binance", "coinbase", etc.


class ExchangeFactory:
    """Factory for creating exchange instances."""

    @staticmethod
    def create(exchange_name: str | ExchangeName | None = None, **kwargs: str) -> Exchange:
        """
        Create an exchange instance.

        Args:
            exchange_name: Name of exchange to create (e.g., "upbit")
                          If None, uses configured default from settings
            **kwargs: Additional arguments for exchange initialization

        Returns:
            Exchange instance

        Raises:
            ValueError: If exchange_name is not supported or API keys are missing
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

        # Create exchange instance
        if exchange_name_lower == "upbit":
            return ExchangeFactory._create_upbit(**kwargs)
        else:
            raise ValueError(
                f"Unsupported exchange: {exchange_name}. "
                f"Supported exchanges: upbit (more exchanges coming soon)"
            )

    @staticmethod
    def _create_upbit(**kwargs: str) -> UpbitExchange:
        """Create Upbit exchange instance."""
        config = get_config()

        # Get API keys from kwargs or config
        access_key = kwargs.get("access_key") or config.get("upbit.access_key") or ""
        secret_key = kwargs.get("secret_key") or config.get("upbit.secret_key") or ""

        # Try Pydantic Settings
        if not access_key or not secret_key:
            with contextlib.suppress(ValueError):
                access_key, secret_key = config.get_upbit_keys()

        if not access_key or not secret_key:
            raise ValueError(
                "Upbit API keys not configured. "
                "Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY environment variables, "
                "or configure in settings.yaml"
            )

        return UpbitExchange(access_key, secret_key)
