"""
Bot initialization module.

Re-exports from focused modules for backward compatibility.
Component creation delegated to BotComponentFactory (SRP).
Recovery/initialization delegated to bot_recovery module (SRP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.exchange import Exchange
from src.execution.bot.bot_factory import BotComponentFactory, BotComponents
from src.execution.bot.bot_recovery import check_existing_holdings, initialize_targets
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.signal_handler import SignalHandler
from src.strategies.volatility_breakout import VanillaVBO

__all__ = [
    "BotComponents",
    "BotComponentFactory",
    "create_bot_components",
    "initialize_targets",
    "check_existing_holdings",
]


def create_bot_components(
    config_path: Path | None = None,
    exchange: Exchange | None = None,
    position_manager: PositionManager | None = None,
    order_manager: OrderManager | None = None,
    signal_handler: SignalHandler | None = None,
    strategy: VanillaVBO | None = None,
) -> tuple[BotComponents, dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    """
    Create and initialize bot components.

    Backward compatibility wrapper for BotComponentFactory.

    Args:
        config_path: Path to configuration file
        exchange: Optional exchange instance for dependency injection
        position_manager: Optional position manager for DI
        order_manager: Optional order manager for DI
        signal_handler: Optional signal handler for DI
        strategy: Optional strategy for DI

    Returns:
        Tuple of (components, trading_config, strategy_config, bot_config, tickers)
    """
    factory = BotComponentFactory(config_path)

    components = factory.create(
        exchange=exchange,
        position_manager=position_manager,
        order_manager=order_manager,
        signal_handler=signal_handler,
        strategy=strategy,
    )

    trading_config, strategy_config, bot_config = factory.get_configs()
    tickers = factory.get_tickers()

    return components, trading_config, strategy_config, bot_config, tickers
