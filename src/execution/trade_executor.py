"""
Trade execution coordination.

Coordinates buy/sell executors and handles ticker updates (SRP).
Uses BuyExecutor and SellExecutor for actual execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.execution.trade_executor_orders import check_advanced_orders
from src.execution.trade_executors import BuyExecutor, SellExecutor
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.exchange import Exchange
    from src.execution.order_manager import OrderManager
    from src.execution.orders.advanced_orders import AdvancedOrderManager
    from src.execution.position_manager import PositionManager
    from src.execution.signal_handler import SignalHandler
    from src.utils.telegram import TelegramNotifier

logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = [
    "BuyExecutor",
    "SellExecutor",
    "sell_all",
    "execute_buy_order",
    "process_ticker_update",
]


def sell_all(
    ticker: str,
    order_manager: OrderManager,
    position_manager: PositionManager,
    exchange: Exchange,
    telegram: TelegramNotifier,
    min_amount: float,
) -> bool:
    """Sell all holdings. Backward compatibility wrapper for SellExecutor."""
    executor = SellExecutor(order_manager, position_manager, exchange, telegram)
    return executor.execute(ticker, min_amount)


def execute_buy_order(
    ticker: str,
    current_price: float,
    buy_amount: float,
    order_manager: OrderManager,
    position_manager: PositionManager,
    advanced_order_manager: AdvancedOrderManager,
    telegram: TelegramNotifier,
    trading_config: dict[str, float | None],
    target_info: dict[str, dict[str, float]],
    min_amount: float,
) -> bool:
    """Execute buy order. Backward compatibility wrapper for BuyExecutor."""
    executor = BuyExecutor(order_manager, position_manager, advanced_order_manager, telegram)
    return executor.execute(
        ticker, current_price, buy_amount, trading_config, target_info, min_amount
    )


def process_ticker_update(
    ticker: str,
    current_price: float,
    position_manager: PositionManager,
    order_manager: OrderManager,
    advanced_order_manager: AdvancedOrderManager,
    signal_handler: SignalHandler,
    trading_config: dict[str, Any],
    target_info: dict[str, dict[str, float]],
    telegram: TelegramNotifier,
    calculate_buy_amount_fn: Callable[[], float],
    execute_buy_fn: Callable[[str, float, float], bool],
) -> None:
    """
    Process real-time ticker update and check for entry signals.

    Coordinates advanced order checking and entry signal processing.
    """
    # Check advanced orders first
    if position_manager.has_position(ticker):
        triggered = check_advanced_orders(
            ticker,
            current_price,
            position_manager,
            order_manager,
            advanced_order_manager,
            signal_handler,
            trading_config,
        )
        if triggered:
            return

    # Skip if already holding (after advanced order check)
    if position_manager.has_position(ticker):
        return

    # Check entry conditions
    metrics = target_info.get(ticker)
    target_price = metrics.get("target") if metrics else None
    if not signal_handler.check_entry_signal(ticker, current_price, target_price):
        return

    # Calculate and validate buy amount
    buy_amount = calculate_buy_amount_fn()
    if buy_amount <= 0:
        return

    # Execute buy order
    execute_buy_fn(ticker, current_price, buy_amount)
