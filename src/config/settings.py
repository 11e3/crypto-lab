"""
Type-safe configuration settings using Pydantic Settings.

This module provides type-safe environment variable management following
Modern Python Development Standards.
"""

import threading
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.constants import PROJECT_ROOT


class Settings(BaseSettings):
    """
    Type-safe configuration settings.

    Environment variables are automatically loaded with the following priority:
    1. Environment variables (highest priority)
    2. .env file (via pydantic-settings)
    3. Default values (fallback)

    All settings are validated and type-checked at initialization.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )

    # Exchange Configuration
    exchange_name: str = Field(
        default="upbit", description="Exchange name (e.g., 'upbit', 'binance')"
    )

    # Upbit API Configuration
    upbit_access_key: str = Field(default="", description="Upbit API access key")
    upbit_secret_key: str = Field(default="", description="Upbit API secret key")

    # Telegram Configuration
    telegram_token: str = Field(default="", description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID")
    telegram_enabled: bool = Field(default=True, description="Enable Telegram notifications")

    # Trading Configuration
    trading_tickers: str = Field(
        default="KRW-BTC,KRW-ETH,KRW-XRP,KRW-TRX",
        description="Comma-separated list of trading tickers",
    )
    trading_fee_rate: float = Field(default=0.0005, description="Trading fee rate (0.05%)")
    trading_max_slots: int = Field(default=4, description="Maximum number of trading slots")
    trading_min_order_amount: float = Field(
        default=5000.0, description="Minimum order amount in KRW"
    )

    # Strategy Configuration
    strategy_name: str = Field(default="VanillaVBO", description="Strategy name")
    strategy_sma_period: int = Field(default=5, description="SMA period for strategy")
    strategy_trend_sma_period: int = Field(default=10, description="Trend SMA period")
    strategy_short_noise_period: int = Field(default=5, description="Short noise period")
    strategy_long_noise_period: int = Field(default=10, description="Long noise period")
    strategy_exclude_current: bool = Field(
        default=True, description="Exclude current bar from calculations (matches legacy/bt.py)"
    )

    # Bot Configuration
    bot_daily_reset_hour: int = Field(default=9, description="Daily reset hour (0-23)")
    bot_daily_reset_minute: int = Field(default=0, description="Daily reset minute (0-59)")
    bot_websocket_reconnect_delay: float = Field(
        default=3.0, description="WebSocket reconnect delay in seconds"
    )
    bot_api_retry_delay: float = Field(default=0.5, description="API retry delay in seconds")

    @field_validator("trading_fee_rate", "bot_websocket_reconnect_delay", "bot_api_retry_delay")
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """Validate that float values are positive."""
        if v < 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator("trading_max_slots", "strategy_sma_period", "strategy_trend_sma_period")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that int values are positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator("bot_daily_reset_hour")
    @classmethod
    def validate_hour(cls, v: int) -> int:
        """Validate hour is in valid range."""
        if not 0 <= v <= 23:
            raise ValueError(f"Hour must be between 0 and 23, got {v}")
        return v

    @field_validator("bot_daily_reset_minute")
    @classmethod
    def validate_minute(cls, v: int) -> int:
        """Validate minute is in valid range."""
        if not 0 <= v <= 59:
            raise ValueError(f"Minute must be between 0 and 59, got {v}")
        return v

    def get_trading_tickers_list(self) -> list[str]:
        """
        Get trading tickers as a list.

        Returns:
            List of ticker symbols
        """
        return [ticker.strip() for ticker in self.trading_tickers.split(",") if ticker.strip()]

    def get_upbit_keys(self) -> tuple[str, str]:
        """
        Get Upbit API keys.

        Returns:
            Tuple of (access_key, secret_key)

        Raises:
            ValueError: If keys are not configured
        """
        if not self.upbit_access_key or not self.upbit_secret_key:
            raise ValueError(
                "Upbit API keys not configured. "
                "Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY environment variables, "
                "or create a .env file with these values."
            )
        return self.upbit_access_key, self.upbit_secret_key

    def get_telegram_config(self) -> dict[str, Any]:
        """
        Get Telegram configuration as a dictionary.

        Returns:
            Dictionary with telegram config
        """
        return {
            "token": self.telegram_token,
            "chat_id": self.telegram_chat_id,
            "enabled": self.telegram_enabled,
        }

    def get_trading_config(self) -> dict[str, Any]:
        """
        Get trading configuration as a dictionary.

        Returns:
            Dictionary with trading config
        """
        return {
            "tickers": self.get_trading_tickers_list(),
            "fee_rate": self.trading_fee_rate,
            "max_slots": self.trading_max_slots,
            "min_order_amount": self.trading_min_order_amount,
        }

    def get_strategy_config(self) -> dict[str, Any]:
        """
        Get strategy configuration as a dictionary.

        Returns:
            Dictionary with strategy config
        """
        return {
            "name": self.strategy_name,
            "sma_period": self.strategy_sma_period,
            "trend_sma_period": self.strategy_trend_sma_period,
            "short_noise_period": self.strategy_short_noise_period,
            "long_noise_period": self.strategy_long_noise_period,
            "exclude_current": self.strategy_exclude_current,
        }

    def get_bot_config(self) -> dict[str, Any]:
        """
        Get bot configuration as a dictionary.

        Returns:
            Dictionary with bot config
        """
        return {
            "daily_reset_hour": self.bot_daily_reset_hour,
            "daily_reset_minute": self.bot_daily_reset_minute,
            "websocket_reconnect_delay": self.bot_websocket_reconnect_delay,
            "api_retry_delay": self.bot_api_retry_delay,
        }


# Global settings instance
_settings: Settings | None = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """
    Get or create global settings instance.

    Returns:
        Settings instance with validated configuration
    """
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings()
    return _settings
