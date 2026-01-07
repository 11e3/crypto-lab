"""
Execution package for live trading.

Contains trading bot, order management, and position tracking.
"""

import warnings

from src.execution.bot_facade import TradingBotFacade, create_bot
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.signal_handler import SignalHandler


# Deprecated: Import with warning
def __getattr__(name: str) -> type:
    """Lazy import for deprecated TradingBot with warning."""
    if name == "TradingBot":
        warnings.warn(
            "TradingBot is deprecated. Use TradingBotFacade instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.execution.bot import TradingBot

        return TradingBot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TradingBot",  # Deprecated, use TradingBotFacade
    "TradingBotFacade",
    "create_bot",
    "OrderManager",
    "PositionManager",
    "SignalHandler",
]
