"""
Advanced orders subpackage for execution.

Contains modules for advanced order types (trailing stop, etc).
"""

from src.execution.orders.advanced_orders import (
    AdvancedOrderManager,
    create_stop_loss_order,
    create_take_profit_order,
    create_trailing_stop_order,
)
from src.execution.orders.advanced_orders_models import (
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
