"""
Advanced orders package.

Contains modules for advanced order types (stop loss, take profit, trailing stop).
Used by backtester for order simulation.
"""

from src.orders.advanced_orders import (
    AdvancedOrderManager,
    create_stop_loss_order,
    create_take_profit_order,
    create_trailing_stop_order,
)
from src.orders.advanced_orders_models import (
    AdvancedOrder,
    OrderType,
)

__all__ = [
    "AdvancedOrderManager",
    "create_stop_loss_order",
    "create_take_profit_order",
    "create_trailing_stop_order",
    "AdvancedOrder",
    "OrderType",
]
