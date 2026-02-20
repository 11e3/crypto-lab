"""
Advanced order handlers.

Each handler is responsible for a single order type (SRP).
"""

from datetime import date

from src.orders.advanced_orders_models import AdvancedOrder, OrderType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StopLossHandler:
    """Handles stop loss order creation and checking."""

    @staticmethod
    def create(
        ticker: str,
        entry_price: float,
        entry_date: date,
        amount: float,
        order_count: int,
        stop_loss_price: float | None = None,
        stop_loss_pct: float | None = None,
    ) -> AdvancedOrder:
        """Create a stop loss order."""
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
            f"Created stop loss order: {ticker} @ {entry_price:.0f}, "
            f"stop loss @ {stop_loss_price:.0f}"
        )
        return order

    @staticmethod
    def check(order: AdvancedOrder, check_low: float, current_date: date) -> bool:
        """Check if stop loss is triggered."""
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


class TakeProfitHandler:
    """Handles take profit order creation and checking."""

    @staticmethod
    def create(
        ticker: str,
        entry_price: float,
        entry_date: date,
        amount: float,
        order_count: int,
        take_profit_price: float | None = None,
        take_profit_pct: float | None = None,
    ) -> AdvancedOrder:
        """Create a take profit order."""
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

    @staticmethod
    def check(order: AdvancedOrder, check_high: float, current_date: date) -> bool:
        """Check if take profit is triggered."""
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


class TrailingStopHandler:
    """Handles trailing stop order creation, updating, and checking."""

    @staticmethod
    def create(
        ticker: str,
        entry_price: float,
        entry_date: date,
        amount: float,
        order_count: int,
        trailing_stop_pct: float,
        initial_stop_loss_pct: float | None = None,
    ) -> AdvancedOrder:
        """Create a trailing stop order."""
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

    @staticmethod
    def update(order: AdvancedOrder, check_high: float) -> None:
        """Update trailing stop highest price and stop loss."""
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


__all__ = ["StopLossHandler", "TakeProfitHandler", "TrailingStopHandler"]
