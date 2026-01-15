"""
Order manager for handling order execution.

Focuses on order placement (SRP). Order tracking is delegated to OrderTracker.
"""

from src.exchange import ExchangeOrderError, InsufficientBalanceError, OrderExecutionService
from src.exchange.types import Order
from src.execution.event_bus import EventBus, get_event_bus
from src.execution.events import EventType, OrderEvent
from src.execution.order_tracker import OrderTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderManager:
    """
    Manages order execution.

    Focuses on placing and cancelling orders.
    Delegates order tracking to OrderTracker.
    """

    def __init__(
        self,
        exchange: OrderExecutionService,
        publish_events: bool = True,
        event_bus: EventBus | None = None,
        order_tracker: OrderTracker | None = None,
    ) -> None:
        """
        Initialize order manager.

        Args:
            exchange: Service implementing OrderExecutionService protocol
            publish_events: Whether to publish events (default: True)
            event_bus: Optional EventBus instance (uses global if not provided)
            order_tracker: Optional OrderTracker (creates default if not provided)
        """
        self.exchange = exchange
        self.publish_events = publish_events
        self.event_bus = event_bus if event_bus else (get_event_bus() if publish_events else None)
        self.order_tracker = order_tracker or OrderTracker(exchange, self.event_bus)

    @property
    def active_orders(self) -> dict[str, Order]:
        """Backward compatibility: access active orders from tracker."""
        return self.order_tracker.active_orders

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
            amount: Amount to buy (in quote currency)
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
            self.order_tracker.add_order(order)
            logger.info(f"Placed buy order: {order.order_id} for {ticker} @ {amount:.0f}")

            self._publish_order_placed(order, ticker, "buy", amount)
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
            amount: Amount to sell (in base currency)
            min_order_amount: Minimum order amount in quote currency

        Returns:
            Order object if successful, None otherwise
        """
        try:
            current_price = self.exchange.get_current_price(ticker)
            order_value = amount * current_price

            if order_value < min_order_amount:
                logger.warning(
                    f"Sell order value {order_value:.0f} below minimum {min_order_amount:.0f} for {ticker}"
                )
                return None

            order = self.exchange.sell_market_order(ticker, amount)
            self.order_tracker.add_order(order)
            logger.info(f"Placed sell order: {order.order_id} for {ticker} @ {amount:.6f}")

            self._publish_order_placed(order, ticker, "sell", amount)
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

    def _publish_order_placed(
        self, order: Order, ticker: str, side: str, amount: float
    ) -> None:
        """Publish order placed event."""
        if not self.event_bus:
            return

        event = OrderEvent(
            event_type=EventType.ORDER_PLACED,
            source="OrderManager",
            order_id=order.order_id,
            ticker=ticker,
            side=side,
            amount=amount,
            price=0.0,
            status="pending",
        )
        self.event_bus.publish(event)

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
                self.order_tracker.remove_order(order_id)
                logger.info(f"Cancelled order: {order_id}")
            return success
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}", exc_info=True)
            return False

    def get_order_status(self, order_id: str) -> Order | None:
        """Get order status. Delegates to OrderTracker."""
        return self.order_tracker.get_status(order_id)

    def get_active_orders(self) -> dict[str, Order]:
        """Get all active orders. Delegates to OrderTracker."""
        return self.order_tracker.get_all_active()

    def clear_filled_orders(self) -> None:
        """Remove filled orders. Delegates to OrderTracker."""
        self.order_tracker.clear_filled()
