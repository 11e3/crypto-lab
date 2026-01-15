"""
Execution package for live trading.

Contains trading bot, order management, and position tracking.
"""

# Lazy imports to avoid cascade failures from optional dependencies (pyupbit)

__all__ = [
    "TradingBotFacade",
    "create_bot",
    "OrderManager",
    "PositionManager",
    "SignalHandler",
]


def __getattr__(name):
    """Lazy import to avoid loading pyupbit unless needed."""
    if name == "TradingBotFacade":
        from src.execution.bot.bot_facade import TradingBotFacade

        return TradingBotFacade
    elif name == "create_bot":
        from src.execution.bot.bot_facade import create_bot

        return create_bot
    elif name == "OrderManager":
        from src.execution.order_manager import OrderManager

        return OrderManager
    elif name == "PositionManager":
        from src.execution.position_manager import PositionManager

        return PositionManager
    elif name == "SignalHandler":
        from src.execution.signal_handler import SignalHandler

        return SignalHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
