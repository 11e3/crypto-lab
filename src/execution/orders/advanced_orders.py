"""
Advanced order types for risk management.

Supports:
- Stop Loss: Automatically sell when price drops below threshold
- Take Profit: Automatically sell when price reaches target
- Trailing Stop: Automatically adjust stop loss as price moves favorably
"""

from datetime import date

from src.execution.orders.advanced_orders_models import AdvancedOrder, OrderType
from src.utils.logger import get_logger

__all__ = [
    "OrderType",
    "AdvancedOrder",
    "AdvancedOrderManager",
    "create_stop_loss_order",
    "create_take_profit_order",
    "create_trailing_stop_order",
    "check_stop_loss",
    "check_take_profit",
    "update_trailing_stop",
]

logger = get_logger(__name__)


# =============================================================================
# Factory Functions
# =============================================================================


def create_stop_loss_order(
    ticker: str,
    entry_price: float,
    entry_date: date,
    amount: float,
    order_count: int,
    stop_loss_price: float | None = None,
    stop_loss_pct: float | None = None,
) -> AdvancedOrder:
    """
    Create a stop loss order.

    Args:
        ticker: Trading pair symbol
        entry_price: Entry price of the position
        entry_date: Entry date
        amount: Position amount
        order_count: Current order count for ID generation
        stop_loss_price: Absolute stop loss price
        stop_loss_pct: Stop loss as percentage below entry (e.g., 0.05 = 5%)

    Returns:
        AdvancedOrder instance

    Raises:
        ValueError: If neither stop_loss_price nor stop_loss_pct is provided
    """
    if stop_loss_price is None and stop_loss_pct is None:
        raise ValueError("Either stop_loss_price or stop_loss_pct must be provided")

    if stop_loss_price is None and stop_loss_pct is not None:
        stop_loss_price = entry_price * (1 - stop_loss_pct)

    order_id = f"stop_loss_{ticker}_{entry_date}_{order_count}"
    order = AdvancedOrder(
        order_id=order_id,
        ticker=ticker,
        order_type=OrderType.STOP_LOSS,
        entry_price=entry_price,
        entry_date=entry_date,
        amount=amount,
        stop_loss_price=stop_loss_price,
        stop_loss_pct=stop_loss_pct,
    )

    logger.info(
        f"Created stop loss order: {ticker} @ {entry_price:.0f}, stop loss @ {stop_loss_price:.0f}"
    )
    return order


def create_take_profit_order(
    ticker: str,
    entry_price: float,
    entry_date: date,
    amount: float,
    order_count: int,
    take_profit_price: float | None = None,
    take_profit_pct: float | None = None,
) -> AdvancedOrder:
    """
    Create a take profit order.

    Args:
        ticker: Trading pair symbol
        entry_price: Entry price of the position
        entry_date: Entry date
        amount: Position amount
        order_count: Current order count for ID generation
        take_profit_price: Absolute take profit price
        take_profit_pct: Take profit as percentage above entry (e.g., 0.10 = 10%)

    Returns:
        AdvancedOrder instance

    Raises:
        ValueError: If neither take_profit_price nor take_profit_pct is provided
    """
    if take_profit_price is None and take_profit_pct is None:
        raise ValueError("Either take_profit_price or take_profit_pct must be provided")

    if take_profit_price is None and take_profit_pct is not None:
        take_profit_price = entry_price * (1 + take_profit_pct)

    order_id = f"take_profit_{ticker}_{entry_date}_{order_count}"
    order = AdvancedOrder(
        order_id=order_id,
        ticker=ticker,
        order_type=OrderType.TAKE_PROFIT,
        entry_price=entry_price,
        entry_date=entry_date,
        amount=amount,
        take_profit_price=take_profit_price,
        take_profit_pct=take_profit_pct,
    )

    logger.info(
        f"Created take profit order: {ticker} @ {entry_price:.0f}, "
        f"take profit @ {take_profit_price:.0f}"
    )
    return order


def create_trailing_stop_order(
    ticker: str,
    entry_price: float,
    entry_date: date,
    amount: float,
    order_count: int,
    trailing_stop_pct: float,
    initial_stop_loss_pct: float | None = None,
) -> AdvancedOrder:
    """
    Create a trailing stop order.

    Args:
        ticker: Trading pair symbol
        entry_price: Entry price of the position
        entry_date: Entry date
        amount: Position amount
        order_count: Current order count for ID generation
        trailing_stop_pct: Percentage to trail from peak (e.g., 0.05 = 5%)
        initial_stop_loss_pct: Initial stop loss percentage (defaults to trailing_stop_pct)

    Returns:
        AdvancedOrder instance
    """
    if initial_stop_loss_pct is None:
        initial_stop_loss_pct = trailing_stop_pct

    initial_stop_loss_price = entry_price * (1 - initial_stop_loss_pct)

    order_id = f"trailing_stop_{ticker}_{entry_date}_{order_count}"
    order = AdvancedOrder(
        order_id=order_id,
        ticker=ticker,
        order_type=OrderType.TRAILING_STOP,
        entry_price=entry_price,
        entry_date=entry_date,
        amount=amount,
        stop_loss_price=initial_stop_loss_price,
        stop_loss_pct=initial_stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
        highest_price=entry_price,
    )

    logger.info(
        f"Created trailing stop order: {ticker} @ {entry_price:.0f}, "
        f"trailing {trailing_stop_pct * 100:.1f}% from peak"
    )
    return order


# =============================================================================
# Check Functions
# =============================================================================


def check_stop_loss(
    order: AdvancedOrder,
    check_low: float,
    current_date: date,
) -> bool:
    """
    Check if stop loss is triggered.

    Args:
        order: Order to check
        check_low: Low price to check against
        current_date: Current date

    Returns:
        True if triggered
    """
    if order.stop_loss_price is None:
        return False

    if check_low <= order.stop_loss_price:
        order.is_triggered = True
        order.is_active = False
        order.triggered_price = order.stop_loss_price
        order.triggered_date = current_date
        logger.info(
            f"Stop loss triggered: {order.ticker} @ {order.stop_loss_price:.0f} "
            f"(entry: {order.entry_price:.0f})"
        )
        return True

    return False


def check_take_profit(
    order: AdvancedOrder,
    check_high: float,
    current_date: date,
) -> bool:
    """
    Check if take profit is triggered.

    Args:
        order: Order to check
        check_high: High price to check against
        current_date: Current date

    Returns:
        True if triggered
    """
    if order.take_profit_price is None:
        return False

    if check_high >= order.take_profit_price:
        order.is_triggered = True
        order.is_active = False
        order.triggered_price = order.take_profit_price
        order.triggered_date = current_date
        logger.info(
            f"Take profit triggered: {order.ticker} @ {order.take_profit_price:.0f} "
            f"(entry: {order.entry_price:.0f})"
        )
        return True

    return False


def update_trailing_stop(
    order: AdvancedOrder,
    check_high: float,
) -> None:
    """
    Update trailing stop highest price and stop loss.

    Args:
        order: Order to update
        check_high: Current high price
    """
    if order.order_type != OrderType.TRAILING_STOP:
        return

    if order.highest_price is None or check_high > order.highest_price:
        order.highest_price = check_high
        if order.trailing_stop_pct is not None:
            order.stop_loss_price = order.highest_price * (1 - order.trailing_stop_pct)
            logger.debug(
                f"Updated trailing stop for {order.ticker}: "
                f"high={order.highest_price:.0f}, "
                f"stop={order.stop_loss_price:.0f}"
            )


# =============================================================================
# Manager Class
# =============================================================================


class AdvancedOrderManager:
    """
    Manages advanced orders (stop loss, take profit, trailing stop).

    Tracks conditional orders and checks if they should be triggered.
    """

    def __init__(self) -> None:
        """Initialize advanced order manager."""
        self.orders: dict[str, AdvancedOrder] = {}

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
        order = create_stop_loss_order(
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
        order = create_take_profit_order(
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
        order = create_trailing_stop_order(
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

            update_trailing_stop(order, check_high)

            if check_stop_loss(order, check_low, current_date):
                triggered_orders.append(order)
                continue

            if check_take_profit(order, check_high, current_date):
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
