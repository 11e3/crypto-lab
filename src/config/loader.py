"""
Configuration loader following 12-Factor App principles.

Priority order:
1. Environment variables (highest priority) - via Pydantic Settings
2. .env file (via pydantic-settings)
3. YAML file (defaults only, optional)
4. Hardcoded defaults (fallback)

This module maintains backward compatibility while using Pydantic Settings
for type-safe environment variable management.
"""

import os
from pathlib import Path
from typing import Any

import yaml

from src.config.constants import PROJECT_ROOT
from src.config.settings import get_settings
from src.utils.logger import get_logger

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv

    # Load .env file from project root
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger = get_logger(__name__)
        logger.debug(f"Loaded .env file from {env_path}")
    else:
        # Try to load from current directory as fallback
        load_dotenv()  # pragma: no cover (module-level code, difficult to test - .env file usually exists in project root)
except ImportError:  # pragma: no cover (module-level code, difficult to test - python-dotenv is always installed in dev)
    # python-dotenv is optional, but recommended
    logger = get_logger(__name__)  # pragma: no cover (module-level code, difficult to test)
    logger.warning(  # pragma: no cover (module-level code, difficult to test)
        "python-dotenv not installed. Install it for .env file support: pip install python-dotenv"
    )

logger = get_logger(__name__)


class ConfigLoader:
    """
    Configuration loader following 12-Factor App principles.

    Environment variables take precedence over file-based configuration.
    Uses Pydantic Settings for type-safe environment variable management.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML config file (optional, defaults only)
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "config" / "settings.yaml"

        self.config_path = config_path
        self._config: dict[str, Any] = {}
        # Use Pydantic Settings for type-safe environment variable management
        self._settings = get_settings()

        # Load YAML only if file exists (optional, for defaults)
        if config_path.exists():
            self.load()

    def load(self) -> None:
        """Load default configuration from YAML file (optional)."""
        if not self.config_path.exists():
            logger.debug(
                f"Config file not found: {self.config_path} (using environment variables only)"
            )
            self._config = {}
            return

        try:
            with open(self.config_path, encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded default config from {self.config_path}")
        except Exception as e:
            logger.warning(
                f"Error loading config file: {e}. Using environment variables only.", exc_info=True
            )
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with environment variable priority.

        Priority order:
        1. Environment variable (e.g., UPBIT_ACCESS_KEY)
        2. YAML file value
        3. Default value

        Args:
            key: Configuration key (e.g., "upbit.access_key")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        # First, try environment variable (highest priority)
        env_key = key.upper().replace(".", "_")
        env_value = os.getenv(env_key)

        if env_value is not None:
            # Parse environment variable based on expected type
            yaml_value = self._get_yaml_value(key)
            return self._parse_env_value(
                env_value, yaml_value if yaml_value is not None else default
            )

        # Second, try YAML file
        yaml_value = self._get_yaml_value(key)
        if yaml_value is not None:
            return yaml_value

        # Finally, return default
        return default

    def _get_yaml_value(self, key: str) -> Any:
        """Get value from YAML config using dot notation."""
        keys = key.split(".")
        value: Any = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return None
            else:
                return None

        return value

    def _parse_env_value(self, env_value: str, type_hint: Any) -> Any:
        """
        Parse environment variable value to appropriate type.

        Args:
            env_value: Environment variable value (string)
            type_hint: Expected type (from YAML or default)

        Returns:
            Parsed value with appropriate type
        """
        # Boolean parsing
        if isinstance(type_hint, bool) or (
            isinstance(type_hint, str) and type_hint.lower() in ("true", "false")
        ):
            return env_value.lower() in ("true", "1", "yes", "on")

        # Integer parsing
        if isinstance(type_hint, int):
            try:
                return int(env_value)
            except ValueError:
                logger.warning(f"Could not parse {env_value} as int, using as string")
                return env_value

        # Float parsing
        if isinstance(type_hint, float):
            try:
                return float(env_value)
            except ValueError:
                logger.warning(f"Could not parse {env_value} as float, using as string")
                return env_value

        # List parsing (comma-separated)
        if isinstance(type_hint, list):
            return [item.strip() for item in env_value.split(",") if item.strip()]

        # Default: return as string
        return env_value

    def get_exchange_name(self) -> str:
        """
        Get configured exchange name.

        Returns:
            Exchange name (e.g., 'upbit')
        """
        return self.get("exchange.name", self._settings.exchange_name) or "upbit"

    def get_upbit_keys(self) -> tuple[str, str]:
        """
        Get Upbit API keys from Pydantic Settings or config.

        Returns:
            Tuple of (access_key, secret_key)

        Raises:
            ValueError: If keys are not configured
        """
        # Try Pydantic Settings first (type-safe, validated)
        try:
            return self._settings.get_upbit_keys()
        except ValueError as e:
            # Fallback to YAML if Pydantic Settings doesn't have keys
            access_key = self._get_yaml_value("upbit.access_key") or ""
            secret_key = self._get_yaml_value("upbit.secret_key") or ""

            if not access_key or not secret_key:
                raise ValueError(
                    "Upbit API keys not configured. "
                    "Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY environment variables, "
                    "or create a .env file with these values."
                ) from e

            return access_key, secret_key

    def get_exchange_keys(self, exchange_name: str | None = None) -> tuple[str, str]:
        """
        Get API keys for the specified exchange.

        Args:
            exchange_name: Exchange name (defaults to configured exchange)

        Returns:
            Tuple of (access_key, secret_key)

        Raises:
            ValueError: If exchange is not supported or keys are not configured
        """
        if exchange_name is None:
            exchange_name = self.get_exchange_name()

        exchange_name = exchange_name.lower()

        if exchange_name == "upbit":
            return self.get_upbit_keys()
        else:
            raise ValueError(
                f"Exchange '{exchange_name}' not supported yet. "
                f"Supported exchanges: upbit"
            )

    def get_telegram_config(self) -> dict[str, Any]:
        """
        Get Telegram configuration.

        Returns:
            Dictionary with telegram config
        """
        # Use Pydantic Settings (type-safe, validated)
        telegram_config = self._settings.get_telegram_config()
        # Override with YAML if needed (YAML takes precedence for enabled flag)
        yaml_enabled = self.get("telegram.enabled")
        if yaml_enabled is not None:
            telegram_config["enabled"] = yaml_enabled
        return telegram_config

    def get_trading_config(self) -> dict[str, Any]:
        """
        Get trading configuration.

        Returns:
            Dictionary with trading config
        """
        # Use Pydantic Settings (type-safe, validated)
        trading_config = self._settings.get_trading_config()
        # Override with YAML if needed (YAML takes precedence)
        yaml_tickers = self.get("trading.tickers")
        if yaml_tickers:
            trading_config["tickers"] = yaml_tickers
        yaml_fee_rate = self.get("trading.fee_rate")
        if yaml_fee_rate is not None:
            trading_config["fee_rate"] = yaml_fee_rate
        yaml_max_slots = self.get("trading.max_slots")
        if yaml_max_slots is not None:
            trading_config["max_slots"] = yaml_max_slots
        yaml_min_order_amount = self.get("trading.min_order_amount")
        if yaml_min_order_amount is not None:
            trading_config["min_order_amount"] = yaml_min_order_amount
        yaml_stop_loss_pct = self.get("trading.stop_loss_pct")
        if yaml_stop_loss_pct is not None:
            trading_config["stop_loss_pct"] = yaml_stop_loss_pct
        yaml_take_profit_pct = self.get("trading.take_profit_pct")
        if yaml_take_profit_pct is not None:
            trading_config["take_profit_pct"] = yaml_take_profit_pct
        yaml_trailing_stop_pct = self.get("trading.trailing_stop_pct")
        if yaml_trailing_stop_pct is not None:
            trading_config["trailing_stop_pct"] = yaml_trailing_stop_pct
        return trading_config

    def get_strategy_config(self) -> dict[str, Any]:
        """
        Get strategy configuration.

        Returns:
            Dictionary with strategy config
        """
        # Use Pydantic Settings (type-safe, validated)
        strategy_config = self._settings.get_strategy_config()
        # Override with YAML if needed (YAML takes precedence)
        for key in [
            "name",
            "sma_period",
            "trend_sma_period",
            "short_noise_period",
            "long_noise_period",
        ]:
            yaml_value = self.get(f"strategy.{key}")
            if yaml_value is not None:
                strategy_config[key] = yaml_value
        return strategy_config

    def get_bot_config(self) -> dict[str, Any]:
        """
        Get bot configuration.

        Returns:
            Dictionary with bot config
        """
        # Use Pydantic Settings (type-safe, validated)
        bot_config = self._settings.get_bot_config()
        # Override with YAML if needed (YAML takes precedence)
        for key in [
            "daily_reset_hour",
            "daily_reset_minute",
            "websocket_reconnect_delay",
            "api_retry_delay",
        ]:
            yaml_value = self.get(f"bot.{key}")
            if yaml_value is not None:
                bot_config[key] = yaml_value
        return bot_config


# Global config instance
_config: ConfigLoader | None = None


def get_config(config_path: Path | None = None) -> ConfigLoader:
    """
    Get or create global config loader instance.

    Args:
        config_path: Path to config file (optional)

    Returns:
        ConfigLoader instance
    """
    global _config
    if _config is None:
        _config = ConfigLoader(config_path)
    return _config
