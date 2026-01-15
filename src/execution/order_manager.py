"""
Order manager for handling order execution and tracking.
"""

from typing import TYPE_CHECKING

from src.exchange import ExchangeOrderError, InsufficientBalanceError
from src.exchange.types import Order
from src.execution.event_bus import EventBus, get_event_bus
from src.execution.events import EventType, OrderEvent
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.exchange import Exchange

logger = get_logger(__name__)


class OrderManager:
    """
    Manages order execution and tracking.

    Handles order placement, status checking, and error handling.
    """

    def __init__(
        self,
        exchange: "Exchange",
        publish_events: bool = True,
        event_bus: EventBus | None = None,
    ) -> None:
        """
        Initialize order manager.

        Args:
            exchange: Exchange instance for order execution
            publish_events: Whether to publish events (default: True)
            event_bus: Optional EventBus instance (uses global if not provided)
        """
        self.exchange = exchange
        self.active_orders: dict[str, Order] = {}  # order_id -> Order
        self.publish_events = publish_events
        self.event_bus = event_bus if event_bus else (get_event_bus() if publish_events else None)

    def place_buy_order(
        self,
        ticker: str,
        amount: float,
        min_order_amount: float = 0.0,
    ) -> Order | None:
        """
        Place a market buy order.

        Args:
            ticker: Trading pair symbol
            amount: Amount to buy (in quote currency, e.g., KRW)
            min_order_amount: Minimum order amount (order skipped if below)

        Returns:
            Order object if successful, None otherwise
        """
        if amount < min_order_amount:
            logger.warning(
                f"Buy order amount {amount:.0f} below minimum {min_order_amount:.0f} for {ticker}"
            )
            return None

        try:
            order = self.exchange.buy_market_order(ticker, amount)
            self.active_orders[order.order_id] = order
            logger.info(f"Placed buy order: {order.order_id} for {ticker} @ {amount:.0f}")

            # Publish event
            if self.event_bus:
                event = OrderEvent(
                    event_type=EventType.ORDER_PLACED,
                    source="OrderManager",
                    order_id=order.order_id,
                    ticker=ticker,
                    side="buy",
                    amount=amount,
                    price=0.0,  # Market order, price unknown at placement
                    status="pending",
                )
                self.event_bus.publish(event)

            return order
        except InsufficientBalanceError as e:
            logger.error(f"Insufficient balance for buy order {ticker}: {e}")
            return None
        except ExchangeOrderError as e:
            logger.error(f"Failed to place buy order {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error placing buy order {ticker}: {e}", exc_info=True)
            return None

    def place_sell_order(
        self,
        ticker: str,
        amount: float,
        min_order_amount: float = 0.0,
    ) -> Order | None:
        """
        Place a market sell order.

        Args:
            ticker: Trading pair symbol
            amount: Amount to sell (in base currency, e.g., BTC)
            min_order_amount: Minimum order amount in quote currency

        Returns:
            Order object if successful, None otherwise
        """
        try:
            # Get current price to check minimum order value
            current_price = self.exchange.get_current_price(ticker)
            order_value = amount * current_price

            if order_value < min_order_amount:
                logger.warning(
                    f"Sell order value {order_value:.0f} below minimum {min_order_amount:.0f} for {ticker}"
                )
                return None

            order = self.exchange.sell_market_order(ticker, amount)
            self.active_orders[order.order_id] = order
            logger.info(f"Placed sell order: {order.order_id} for {ticker} @ {amount:.6f}")

            # Publish event
            if self.event_bus:
                event = OrderEvent(
                    event_type=EventType.ORDER_PLACED,
                    source="OrderManager",
                    order_id=order.order_id,
                    ticker=ticker,
                    side="sell",
                    amount=amount,
                    price=0.0,  # Market order, price unknown at placement
                    status="pending",
                )
                self.event_bus.publish(event)

            return order
        except InsufficientBalanceError as e:
            logger.error(f"Insufficient balance for sell order {ticker}: {e}")
            return None
        except ExchangeOrderError as e:
            logger.error(f"Failed to place sell order {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error placing sell order {ticker}: {e}", exc_info=True)
            return None

    def sell_all(self, ticker: str, min_order_amount: float = 0.0) -> Order | None:
        """
        Sell all holdings for a ticker.

        Args:
            ticker: Trading pair symbol
            min_order_amount: Minimum order amount in quote currency

        Returns:
            Order object if successful, None otherwise
        """
        try:
            currency = ticker.split("-")[1]
            balance = self.exchange.get_balance(currency)

            if balance.available <= 0:
                logger.debug(f"No balance to sell for {ticker}")
                return None

            return self.place_sell_order(ticker, balance.available, min_order_amount)
        except Exception as e:
            logger.error(f"Error selling all for {ticker}: {e}", exc_info=True)
            return None

    def get_order_status(self, order_id: str) -> Order | None:
        """
        Get status of an order.

        Args:
            order_id: Order identifier

        Returns:
            Order object with current status, None on error
        """
        try:
            order = self.exchange.get_order_status(order_id)
            # Update cached order
            old_order = self.active_orders.get(order_id)
            if order_id in self.active_orders:
                self.active_orders[order_id] = order

            # Publish event if status changed
            if self.event_bus and old_order and old_order.status != order.status:
                if order.is_filled:
                    event_type = EventType.ORDER_FILLED
                elif order.status.value == "cancelled":
                    event_type = EventType.ORDER_CANCELLED
                elif order.status.value == "failed":
                    event_type = EventType.ORDER_FAILED
                else:
                    event_type = EventType.ORDER_PLACED

                event = OrderEvent(
                    event_type=event_type,
                    source="OrderManager",
                    order_id=order.order_id,
                    ticker=order.symbol,
                    side=order.side.value,
                    amount=order.amount,
                    price=order.filled_price or order.price or 0.0,
                    status=order.status.value,
                )
                self.event_bus.publish(event)

            return order
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}", exc_info=True)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Order identifier

        Returns:
            True if cancellation successful
        """
        try:
            success = self.exchange.cancel_order(order_id)
            if success:
                self.active_orders.pop(order_id, None)
                logger.info(f"Cancelled order: {order_id}")
            return success
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}", exc_info=True)
            return False

    def get_active_orders(self) -> dict[str, Order]:
        """
        Get all active orders.

        Returns:
            Dictionary of order_id -> Order
        """
        return self.active_orders.copy()

    def clear_filled_orders(self) -> None:
        """Remove filled orders from active orders cache."""
        filled_order_ids = [
            order_id for order_id, order in self.active_orders.items() if order.is_filled
        ]
        for order_id in filled_order_ids:
            self.active_orders.pop(order_id, None)
        if filled_order_ids:
            logger.debug(f"Cleared {len(filled_order_ids)} filled orders")
