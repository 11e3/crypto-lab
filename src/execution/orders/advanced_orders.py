"""
Advanced order types for risk management.

Supports:
- Stop Loss: Automatically sell when price drops below threshold
- Take Profit: Automatically sell when price reaches target
- Trailing Stop: Automatically adjust stop loss as price moves favorably

Uses specialized handlers for each order type (SRP).
"""

from datetime import date

from src.execution.orders.advanced_orders_models import AdvancedOrder, OrderType
from src.execution.orders.order_handlers import (
    StopLossHandler,
    TakeProfitHandler,
    TrailingStopHandler,
)
from src.utils.logger import get_logger

__all__ = [
    "OrderType",
    "AdvancedOrder",
    "AdvancedOrderManager",
    "StopLossHandler",
    "TakeProfitHandler",
    "TrailingStopHandler",
    # Backward compatibility
    "create_stop_loss_order",
    "create_take_profit_order",
    "create_trailing_stop_order",
    "check_stop_loss",
    "check_take_profit",
    "update_trailing_stop",
]

logger = get_logger(__name__)

# Backward compatibility aliases
create_stop_loss_order = StopLossHandler.create
create_take_profit_order = TakeProfitHandler.create
create_trailing_stop_order = TrailingStopHandler.create
check_stop_loss = StopLossHandler.check
check_take_profit = TakeProfitHandler.check
update_trailing_stop = TrailingStopHandler.update


class AdvancedOrderManager:
    """
    Manages advanced orders (stop loss, take profit, trailing stop).

    Uses specialized handlers for each order type.
    """

    def __init__(
        self,
        stop_loss_handler: StopLossHandler | None = None,
        take_profit_handler: TakeProfitHandler | None = None,
        trailing_stop_handler: TrailingStopHandler | None = None,
    ) -> None:
        """Initialize advanced order manager with handlers."""
        self.orders: dict[str, AdvancedOrder] = {}
        self.stop_loss = stop_loss_handler or StopLossHandler()
        self.take_profit = take_profit_handler or TakeProfitHandler()
        self.trailing_stop = trailing_stop_handler or TrailingStopHandler()

    def create_stop_loss(
        self,
        ticker: str,
        entry_price: float,
        entry_date: date,
        amount: float,
        stop_loss_price: float | None = None,
        stop_loss_pct: float | None = None,
    ) -> AdvancedOrder:
        """Create and register a stop loss order."""
        order = self.stop_loss.create(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date,
            amount=amount,
            order_count=len(self.orders),
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
        )
        self.orders[order.order_id] = order
        return order

    def create_take_profit(
        self,
        ticker: str,
        entry_price: float,
        entry_date: date,
        amount: float,
        take_profit_price: float | None = None,
        take_profit_pct: float | None = None,
    ) -> AdvancedOrder:
        """Create and register a take profit order."""
        order = self.take_profit.create(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date,
            amount=amount,
            order_count=len(self.orders),
            take_profit_price=take_profit_price,
            take_profit_pct=take_profit_pct,
        )
        self.orders[order.order_id] = order
        return order

    def create_trailing_stop(
        self,
        ticker: str,
        entry_price: float,
        entry_date: date,
        amount: float,
        trailing_stop_pct: float,
        initial_stop_loss_pct: float | None = None,
    ) -> AdvancedOrder:
        """Create and register a trailing stop order."""
        order = self.trailing_stop.create(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date,
            amount=amount,
            order_count=len(self.orders),
            trailing_stop_pct=trailing_stop_pct,
            initial_stop_loss_pct=initial_stop_loss_pct,
        )
        self.orders[order.order_id] = order
        return order

    def check_orders(
        self,
        ticker: str,
        current_price: float,
        current_date: date,
        low_price: float | None = None,
        high_price: float | None = None,
    ) -> list[AdvancedOrder]:
        """
        Check if any orders should be triggered.

        Args:
            ticker: Trading pair symbol
            current_price: Current market price
            current_date: Current date
            low_price: Low price of the period (for stop loss checking)
            high_price: High price of the period (for take profit checking)

        Returns:
            List of triggered orders
        """
        triggered_orders: list[AdvancedOrder] = []

        check_low = low_price if low_price is not None else current_price
        check_high = high_price if high_price is not None else current_price

        for order in self.orders.values():
            if not order.is_active or order.is_triggered or order.ticker != ticker:
                continue

            self.trailing_stop.update(order, check_high)

            if self.stop_loss.check(order, check_low, current_date):
                triggered_orders.append(order)
                continue

            if self.take_profit.check(order, check_high, current_date):
                triggered_orders.append(order)

        return triggered_orders

    def get_active_orders(self, ticker: str | None = None) -> list[AdvancedOrder]:
        """Get active orders, optionally filtered by ticker."""
        orders = [o for o in self.orders.values() if o.is_active and not o.is_triggered]
        if ticker:
            orders = [o for o in orders if o.ticker == ticker]
        return orders

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an advanced order by ID."""
        if order_id in self.orders:
            self.orders[order_id].is_active = False
            logger.info(f"Cancelled advanced order: {order_id}")
            return True
        return False

    def cancel_all_orders(self, ticker: str | None = None) -> int:
        """Cancel all orders, optionally filtered by ticker."""
        count = 0
        for order in self.orders.values():
            if ticker and order.ticker != ticker:
                continue
            if order.is_active:
                order.is_active = False
                count += 1

        if count > 0:
            logger.info(
                f"Cancelled {count} advanced order(s)" + (f" for {ticker}" if ticker else "")
            )
        return count
