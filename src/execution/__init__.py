"""
Execution package for live trading.

Contains trading bot, order management, and position tracking.
"""

# Lazy imports to avoid cascade failures from optional dependencies (pyupbit)

__all__ = [
    "TradingBotFacade",
    "create_bot",
    "OrderManager",
    "OrderTracker",
    "PositionManager",
    "PnLCalculator",
    "SignalHandler",
    "SignalDataLoader",
    "SignalMetricsCalculator",
]


from typing import Any


def __getattr__(name: str) -> Any:
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
    elif name == "OrderTracker":
        from src.execution.order_tracker import OrderTracker

        return OrderTracker
    elif name == "PositionManager":
        from src.execution.position_manager import PositionManager

        return PositionManager
    elif name == "PnLCalculator":
        from src.execution.pnl_calculator import PnLCalculator

        return PnLCalculator
    elif name == "SignalHandler":
        from src.execution.signal_handler import SignalHandler

        return SignalHandler
    elif name == "SignalDataLoader":
        from src.execution.signal_data import SignalDataLoader

        return SignalDataLoader
    elif name == "SignalMetricsCalculator":
        from src.execution.signal_metrics import SignalMetricsCalculator

        return SignalMetricsCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
