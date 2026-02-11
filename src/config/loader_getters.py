"""Configuration getter functions for specific config sections."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader


def get_telegram_config(loader: "ConfigLoader") -> dict[str, Any]:
    """
    Get Telegram configuration.

    Args:
        loader: ConfigLoader instance

    Returns:
        Dictionary with telegram config
    """
    telegram_config: dict[str, Any] = loader.settings.get_telegram_config()
    yaml_enabled = loader.get("telegram.enabled")
    if yaml_enabled is not None:
        telegram_config["enabled"] = yaml_enabled
    return telegram_config


def get_trading_config(loader: "ConfigLoader") -> dict[str, Any]:
    """
    Get trading configuration.

    Args:
        loader: ConfigLoader instance

    Returns:
        Dictionary with trading config
    """
    trading_config: dict[str, Any] = loader.settings.get_trading_config()

    yaml_overrides: list[tuple[str, str]] = [
        ("tickers", "trading.tickers"),
        ("fee_rate", "trading.fee_rate"),
        ("max_slots", "trading.max_slots"),
        ("min_order_amount", "trading.min_order_amount"),
        ("stop_loss_pct", "trading.stop_loss_pct"),
        ("take_profit_pct", "trading.take_profit_pct"),
        ("trailing_stop_pct", "trading.trailing_stop_pct"),
    ]

    for config_key, yaml_key in yaml_overrides:
        yaml_value = loader.get(yaml_key)
        if yaml_value is not None:
            trading_config[config_key] = yaml_value

    return trading_config


def get_strategy_config(loader: "ConfigLoader") -> dict[str, Any]:
    """
    Get strategy configuration.

    Args:
        loader: ConfigLoader instance

    Returns:
        Dictionary with strategy config
    """
    strategy_config: dict[str, Any] = loader.settings.get_strategy_config()

    strategy_keys = [
        "name",
        "sma_period",
        "trend_sma_period",
        "short_noise_period",
        "long_noise_period",
    ]

    for key in strategy_keys:
        yaml_value = loader.get(f"strategy.{key}")
        if yaml_value is not None:
            strategy_config[key] = yaml_value

    return strategy_config


def get_bot_config(loader: "ConfigLoader") -> dict[str, Any]:
    """
    Get bot configuration.

    Args:
        loader: ConfigLoader instance

    Returns:
        Dictionary with bot config
    """
    bot_config: dict[str, Any] = loader.settings.get_bot_config()

    bot_keys = [
        "daily_reset_hour",
        "daily_reset_minute",
        "websocket_reconnect_delay",
        "api_retry_delay",
    ]

    for key in bot_keys:
        yaml_value = loader.get(f"bot.{key}")
        if yaml_value is not None:
            bot_config[key] = yaml_value

    return bot_config
