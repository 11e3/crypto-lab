"""Configuration loader with environment variable and YAML support."""

import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml

_loader_logger = logging.getLogger(__name__)

from src.config.loader_getters import (  # noqa: E402
    get_bot_config as _get_bot_config,
)
from src.config.loader_getters import (  # noqa: E402
    get_strategy_config as _get_strategy_config,
)
from src.config.loader_getters import (  # noqa: E402
    get_telegram_config as _get_telegram_config,
)
from src.config.loader_getters import (  # noqa: E402
    get_trading_config as _get_trading_config,
)
from src.config.loader_parsers import get_yaml_value, parse_env_value  # noqa: E402
from src.config.settings import Settings  # noqa: E402


class ConfigLoader:
    """
    Configuration loader with support for Pydantic Settings and YAML.

    Priority: Environment Variables > YAML config > Default values
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML config file (optional)
        """
        self._config_path = config_path
        self._yaml_config: dict[str, Any] = {}
        self._settings = Settings()
        self.load()

    def load(self) -> None:
        """
        Load configuration from YAML file.

        Searches for config in these locations (in order):
        1. Explicit path provided to constructor
        2. config/settings.yaml (relative to project root)
        3. settings.yaml (in current directory)
        """
        search_paths: list[Path] = []

        if self._config_path:
            search_paths.append(self._config_path)
        else:
            project_root = Path(__file__).parent.parent.parent
            search_paths.extend(
                [
                    project_root / "config" / "settings.yaml",
                    Path("settings.yaml"),
                    Path("config/settings.yaml"),
                ]
            )

        for path in search_paths:
            _loader_logger.debug(f"Searching for config at: {path}")
            if path.exists():
                _loader_logger.info(f"Loaded config from: {path}")
                with path.open(encoding="utf-8") as f:
                    self._yaml_config = yaml.safe_load(f) or {}
                return

        _loader_logger.debug("No config file found in search paths")
        self._yaml_config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Priority: Environment Variables > YAML > Default

        Args:
            key: Dot-notation key (e.g., "trading.fee_rate")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        env_key = key.upper().replace(".", "_")
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return parse_env_value(env_value)

        yaml_value = self._get_yaml_value(key)
        if yaml_value is not None:
            return yaml_value

        return default

    def _get_yaml_value(self, key: str) -> Any:
        """Get value from YAML config using dot notation."""
        return get_yaml_value(self._yaml_config, key)

    def get_exchange_name(self) -> str:
        """
        Get configured exchange name.

        Returns:
            Exchange name (default: "upbit")
        """
        exchange: str = self._settings.exchange_name
        yaml_exchange = self.get("exchange")
        if yaml_exchange:
            exchange = str(yaml_exchange)
        return exchange.lower()

    def get_upbit_keys(self) -> tuple[str, str]:
        """
        Get Upbit API keys.

        Returns:
            Tuple of (access_key, secret_key)

        Raises:
            ValueError: If keys are not configured
        """
        try:
            keys: tuple[str, str] = self._settings.get_upbit_keys()
            return keys
        except ValueError as e:
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
                f"Exchange '{exchange_name}' not supported yet. Supported exchanges: upbit"
            )

    def get_telegram_config(self) -> dict[str, Any]:
        """Get Telegram configuration."""
        return _get_telegram_config(self)

    def get_trading_config(self) -> dict[str, Any]:
        """Get trading configuration."""
        return _get_trading_config(self)

    def get_strategy_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return _get_strategy_config(self)

    def get_bot_config(self) -> dict[str, Any]:
        """Get bot configuration."""
        return _get_bot_config(self)


# Global config instance
_config: ConfigLoader | None = None
_config_lock = threading.Lock()


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
        with _config_lock:
            if _config is None:
                _config = ConfigLoader(config_path)
    return _config
