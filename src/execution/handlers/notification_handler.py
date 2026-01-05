"""
Notification event handler.

Handles events for sending notifications (e.g., Telegram).
"""

from typing import TYPE_CHECKING

from src.execution.event_bus import get_event_bus
from src.execution.events import (
    ErrorEvent,
    Event,
    EventType,
    OrderEvent,
    PositionEvent,
    SignalEvent,
    SystemEvent,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.telegram import TelegramNotifier

logger = get_logger(__name__)


class NotificationHandler:
    """
    Handler for notification events.

    Sends notifications via Telegram or other channels based on events.
    """

    def __init__(self, telegram_notifier: "TelegramNotifier | None" = None) -> None:
        """
        Initialize notification handler.

        Args:
            telegram_notifier: Telegram notifier instance (optional)
        """
        self.telegram = telegram_notifier
        self.event_bus = get_event_bus()
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register event handlers."""
        self.event_bus.subscribe(EventType.ENTRY_SIGNAL, self._handle_entry_signal)
        self.event_bus.subscribe(EventType.EXIT_SIGNAL, self._handle_exit_signal)
        self.event_bus.subscribe(EventType.ORDER_PLACED, self._handle_order_placed)
        self.event_bus.subscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self.event_bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self.event_bus.subscribe(EventType.ERROR, self._handle_error)
        self.event_bus.subscribe(EventType.DAILY_RESET, self._handle_daily_reset)

    def _handle_entry_signal(self, event: Event) -> None:
        """Handle entry signal event."""
        if isinstance(event, SignalEvent) and self.telegram:
            try:
                self.telegram.send_trade_signal(
                    "ENTRY_SIGNAL",
                    event.ticker,
                    event.price,
                    target=event.target_price,
                )
            except Exception as e:
                logger.error(f"Error sending entry signal notification: {e}", exc_info=True)

    def _handle_exit_signal(self, event: Event) -> None:
        """Handle exit signal event."""
        if isinstance(event, SignalEvent) and self.telegram:
            try:
                self.telegram.send_trade_signal(
                    "EXIT_SIGNAL",
                    event.ticker,
                    event.price,
                )
            except Exception as e:
                logger.error(f"Error sending exit signal notification: {e}", exc_info=True)

    def _handle_order_placed(self, event: Event) -> None:
        """Handle order placed event."""
        # Safety check: prevent notifications during tests
        import os
        import sys
        
        is_testing = (
            "pytest" in sys.modules
            or "unittest" in sys.modules
            or "PYTEST_CURRENT_TEST" in os.environ
            or any("test" in arg.lower() for arg in sys.argv)
        )
        
        if is_testing:
            logger.debug(f"Order notification blocked during testing: {event.ticker}")
            return
        
        if isinstance(event, OrderEvent) and self.telegram:
            try:
                side = "BUY" if event.side == "buy" else "SELL"
                self.telegram.send_trade_signal(
                    side,
                    event.ticker,
                    event.price,
                    amount=event.amount,
                )
            except Exception as e:
                logger.error(f"Error sending order notification: {e}", exc_info=True)

    def _handle_position_opened(self, event: Event) -> None:
        """Handle position opened event."""
        if isinstance(event, PositionEvent) and self.telegram:
            try:
                self.telegram.send_trade_signal(
                    "POSITION_OPENED",
                    event.ticker,
                    event.entry_price,
                    amount=event.amount,
                )
            except Exception as e:
                logger.error(f"Error sending position opened notification: {e}", exc_info=True)

    def _handle_position_closed(self, event: Event) -> None:
        """Handle position closed event."""
        if isinstance(event, PositionEvent) and self.telegram:
            try:
                msg = (
                    f"Position closed: {event.ticker}\n"
                    f"PnL: {event.pnl:.0f} KRW ({event.pnl_pct:.2f}%)"
                )
                self.telegram.send(msg)
            except Exception as e:
                logger.error(f"Error sending position closed notification: {e}", exc_info=True)

    def _handle_error(self, event: Event) -> None:
        """Handle error event."""
        if isinstance(event, ErrorEvent) and self.telegram:
            try:
                msg = f"Error: {event.error_type}\n{event.error_message}"
                self.telegram.send(msg)
            except Exception as e:
                logger.error(f"Error sending error notification: {e}", exc_info=True)

    def _handle_daily_reset(self, event: Event) -> None:
        """Handle daily reset event."""
        if isinstance(event, SystemEvent) and self.telegram:
            try:
                self.telegram.send("Daily reset completed")
            except Exception as e:
                logger.error(f"Error sending daily reset notification: {e}", exc_info=True)
