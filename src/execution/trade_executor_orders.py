"""Advanced order creation and checking functions."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from src.exchange import ExchangeError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.execution.order_manager import OrderManager
    from src.execution.orders.advanced_orders import AdvancedOrderManager
    from src.execution.position_manager import PositionManager
    from src.execution.signal_handler import SignalHandler

logger = get_logger(__name__)


def create_advanced_orders(
    ticker: str,
    current_price: float,
    estimated_amount: float,
    advanced_order_manager: AdvancedOrderManager,
    trading_config: dict[str, float | None],
) -> None:
    """
    Create advanced orders (stop loss, take profit, trailing stop).

    Args:
        ticker: Trading pair ticker
        current_price: Entry price
        estimated_amount: Position amount
        advanced_order_manager: Advanced order manager instance
        trading_config: Trading configuration with stop loss/take profit settings
    """
    stop_loss_pct = trading_config.get("stop_loss_pct")
    take_profit_pct = trading_config.get("take_profit_pct")
    trailing_stop_pct = trading_config.get("trailing_stop_pct")

    if stop_loss_pct is not None:
        advanced_order_manager.create_stop_loss(
            ticker=ticker,
            entry_price=current_price,
            entry_date=date.today(),
            amount=estimated_amount,
            stop_loss_pct=stop_loss_pct,
        )

    if take_profit_pct is not None:
        advanced_order_manager.create_take_profit(
            ticker=ticker,
            entry_price=current_price,
            entry_date=date.today(),
            amount=estimated_amount,
            take_profit_pct=take_profit_pct,
        )

    if trailing_stop_pct is not None:
        advanced_order_manager.create_trailing_stop(
            ticker=ticker,
            entry_price=current_price,
            entry_date=date.today(),
            amount=estimated_amount,
            trailing_stop_pct=trailing_stop_pct,
        )


def check_advanced_orders(
    ticker: str,
    current_price: float,
    position_manager: PositionManager,
    order_manager: OrderManager,
    advanced_order_manager: AdvancedOrderManager,
    signal_handler: SignalHandler,
    trading_config: dict[str, Any],
) -> bool:
    """
    Check and execute advanced orders.

    Args:
        ticker: Trading pair ticker
        current_price: Current market price
        position_manager: Position manager instance
        order_manager: Order manager instance
        advanced_order_manager: Advanced order manager instance
        signal_handler: Signal handler instance
        trading_config: Trading configuration

    Returns:
        True if order triggered
    """
    try:
        ohlcv_data = signal_handler.get_ohlcv_data(ticker, count=1)
        if ohlcv_data is not None and len(ohlcv_data) > 0:
            latest = ohlcv_data.iloc[-1]
            low_price = latest.get("low", current_price)
            high_price = latest.get("high", current_price)

            triggered = advanced_order_manager.check_orders(
                ticker=ticker,
                current_price=current_price,
                current_date=date.today(),
                low_price=low_price,
                high_price=high_price,
            )

            if triggered:
                logger.info(
                    f"Advanced order triggered for {ticker}: "
                    f"{triggered[0].order_type.value} @ {triggered[0].triggered_price:.0f}"
                )
                return _execute_triggered_order(
                    ticker,
                    position_manager,
                    order_manager,
                    advanced_order_manager,
                    trading_config,
                )
    except (ExchangeError, ConnectionError, OSError, KeyError, IndexError, ValueError, TypeError) as e:
        logger.warning(f"Error checking advanced orders for {ticker}: {e}")
    return False


def _execute_triggered_order(
    ticker: str,
    position_manager: PositionManager,
    order_manager: OrderManager,
    advanced_order_manager: AdvancedOrderManager,
    trading_config: dict[str, Any],
) -> bool:
    """Execute sell order when advanced order is triggered."""
    position = position_manager.get_position(ticker)
    if position:
        min_order_amount = trading_config.get("min_order_amount", 5000.0)
        sell_order = order_manager.place_sell_order(
            ticker=ticker,
            amount=position.amount,
            min_order_amount=min_order_amount,
        )
        if sell_order:
            position_manager.remove_position(ticker)
            advanced_order_manager.cancel_all_orders(ticker=ticker)
            logger.info(f"Sold {ticker} due to advanced order trigger")
    return True


__all__ = [
    "create_advanced_orders",
    "check_advanced_orders",
]
