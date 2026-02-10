"""
Order tracking and status management.

Separates order tracking responsibility from order execution (SRP).
"""

from src.exchange import ExchangeError, OrderExecutionService
from src.exchange.types import Order
from src.execution.event_bus import EventBus
from src.execution.events import EventType, OrderEvent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderTracker:
    """
    Tracks order status and manages active orders cache.

    Focuses on order lifecycle tracking, separate from order execution.
    """

    def __init__(
        self,
        exchange: OrderExecutionService,
        event_bus: EventBus | None = None,
    ) -> None:
        """
        Initialize order tracker.

        Args:
            exchange: Service implementing OrderExecutionService protocol
            event_bus: Optional EventBus for status change events
        """
        self.exchange = exchange
        self.event_bus = event_bus
        self.active_orders: dict[str, Order] = {}

    def add_order(self, order: Order) -> None:
        """Add an order to tracking."""
        self.active_orders[order.order_id] = order

    def remove_order(self, order_id: str) -> Order | None:
        """Remove an order from tracking."""
        return self.active_orders.pop(order_id, None)

    def get_order(self, order_id: str) -> Order | None:
        """Get tracked order by ID."""
        return self.active_orders.get(order_id)

    def get_status(self, order_id: str) -> Order | None:
        """
        Get order status from exchange and update cache.

        Args:
            order_id: Order identifier

        Returns:
            Order object with current status, None on error
        """
        try:
            order = self.exchange.get_order_status(order_id)
            old_order = self.active_orders.get(order_id)

            if order_id in self.active_orders:
                self.active_orders[order_id] = order

            self._publish_status_change(old_order, order)
            return order
        except (ExchangeError, ConnectionError, OSError) as e:
            logger.error(f"Error getting order status for {order_id}: {e}", exc_info=True)
            return None

    def _publish_status_change(self, old_order: Order | None, new_order: Order) -> None:
        """Publish event if order status changed."""
        if not self.event_bus or not old_order:
            return

        if old_order.status == new_order.status:
            return

        if new_order.is_filled:
            event_type = EventType.ORDER_FILLED
        elif new_order.status.value == "cancelled":
            event_type = EventType.ORDER_CANCELLED
        elif new_order.status.value == "failed":
            event_type = EventType.ORDER_FAILED
        else:
            event_type = EventType.ORDER_PLACED

        event = OrderEvent(
            event_type=event_type,
            source="OrderTracker",
            order_id=new_order.order_id,
            ticker=new_order.symbol,
            side=new_order.side.value,
            amount=new_order.amount,
            price=new_order.filled_price or new_order.price or 0.0,
            status=new_order.status.value,
        )
        self.event_bus.publish(event)

    def get_all_active(self) -> dict[str, Order]:
        """Get all tracked orders."""
        return self.active_orders.copy()

    def clear_filled(self) -> int:
        """
        Remove filled orders from tracking.

        Returns:
            Number of orders cleared
        """
        filled_ids = [order_id for order_id, order in self.active_orders.items() if order.is_filled]
        for order_id in filled_ids:
            self.active_orders.pop(order_id, None)

        if filled_ids:
            logger.debug(f"Cleared {len(filled_ids)} filled orders")

        return len(filled_ids)

    def clear_all(self) -> None:
        """Clear all tracked orders."""
        self.active_orders.clear()
