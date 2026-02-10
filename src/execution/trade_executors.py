"""
Trade execution handlers.

Each executor handles a single trade direction (SRP).
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

from src.exchange import ExchangeError
from src.execution.trade_executor_orders import create_advanced_orders
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.exchange import Exchange
    from src.execution.order_manager import OrderManager
    from src.execution.orders.advanced_orders import AdvancedOrderManager
    from src.execution.position_manager import PositionManager
    from src.utils.telegram import TelegramNotifier

logger = get_logger(__name__)


class BuyExecutor:
    """Handles buy order execution."""

    def __init__(
        self,
        order_manager: OrderManager,
        position_manager: PositionManager,
        advanced_order_manager: AdvancedOrderManager,
        telegram: TelegramNotifier,
    ) -> None:
        """Initialize buy executor."""
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.advanced_order_manager = advanced_order_manager
        self.telegram = telegram

    def execute(
        self,
        ticker: str,
        current_price: float,
        buy_amount: float,
        trading_config: dict[str, Any],
        target_info: dict[str, dict[str, float]],
        min_amount: float,
    ) -> bool:
        """
        Execute buy market order.

        Args:
            ticker: Trading pair ticker
            current_price: Current market price
            buy_amount: Amount to buy in KRW
            trading_config: Trading configuration
            target_info: Target information dict
            min_amount: Minimum order amount

        Returns:
            True if order executed successfully
        """
        try:
            order = self.order_manager.place_buy_order(ticker, buy_amount, min_amount)

            if order and order.order_id:
                estimated_amount = buy_amount / current_price if current_price > 0 else 0.0
                self.position_manager.add_position(
                    ticker=ticker,
                    entry_price=current_price,
                    amount=estimated_amount,
                )

                self._create_advanced_orders(
                    ticker, current_price, estimated_amount, trading_config
                )

                self._send_notification(ticker, current_price, target_info)
                return True
            else:
                logger.warning(f"Buy failed for {ticker}: order not created")
                return False
        except (ExchangeError, ConnectionError, OSError, ValueError) as e:
            logger.error(f"Buy error for {ticker}: {e}", exc_info=True)
            return False

    def _create_advanced_orders(
        self,
        ticker: str,
        current_price: float,
        estimated_amount: float,
        trading_config: dict[str, Any],
    ) -> None:
        """Create advanced orders (stop loss, take profit, trailing stop)."""
        create_advanced_orders(
            ticker,
            current_price,
            estimated_amount,
            self.advanced_order_manager,
            trading_config,
        )

    def _send_notification(
        self,
        ticker: str,
        current_price: float,
        target_info: dict[str, dict[str, float]],
    ) -> None:
        """Send buy notification."""
        metrics = target_info.get(ticker, {})
        self.telegram.send_trade_signal(
            "BUY",
            ticker,
            current_price,
            target=metrics.get("target", 0),
            noise=f"{metrics.get('k', 0):.2f} < {metrics.get('long_noise', 0):.2f}",
        )
        logger.info(f"BUY {ticker} @ {current_price:.0f} | Target: {metrics.get('target', 0):.0f}")


class SellExecutor:
    """Handles sell order execution."""

    def __init__(
        self,
        order_manager: OrderManager,
        position_manager: PositionManager,
        exchange: Exchange,
        telegram: TelegramNotifier,
    ) -> None:
        """Initialize sell executor."""
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.exchange = exchange
        self.telegram = telegram

    def execute(self, ticker: str, min_amount: float) -> bool:
        """
        Sell all holdings for a ticker.

        Args:
            ticker: Trading pair ticker
            min_amount: Minimum order amount

        Returns:
            True if sold successfully
        """
        try:
            order = self.order_manager.sell_all(ticker, min_amount)

            if order and order.order_id:
                self.position_manager.remove_position(ticker)
                self._send_notification(ticker)
                logger.info(f"Sold all {ticker}")
                return True
        except (ExchangeError, ConnectionError, OSError) as e:
            logger.error(f"Sell error for {ticker}: {e}", exc_info=True)
        return False

    def _send_notification(self, ticker: str) -> None:
        """Send sell notification (skipped during testing)."""
        if self._is_testing():
            return

        try:
            curr_price = self.exchange.get_current_price(ticker)
            currency = ticker.split("-")[1]
            balance = self.exchange.get_balance(currency)
            self.telegram.send_trade_signal(
                "SELL",
                ticker,
                curr_price,
                amount=balance.available,
            )
        except (ExchangeError, ConnectionError, OSError) as e:
            logger.debug(f"Sell notification skipped for {ticker}: {e}")

    @staticmethod
    def _is_testing() -> bool:
        """Check if running in test environment."""
        return (
            "pytest" in sys.modules
            or "unittest" in sys.modules
            or "PYTEST_CURRENT_TEST" in os.environ
            or any("test" in arg.lower() for arg in sys.argv)
        )
