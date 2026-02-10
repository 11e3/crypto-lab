"""
Integration tests for EventBus, TradeHandler, and NotificationHandler.

Tests event publishing, subscription, routing, and handler auto-registration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.execution.event_bus import EventBus, get_event_bus, set_event_bus
from src.execution.events import (
    ErrorEvent,
    Event,
    EventType,
    OrderEvent,
    PositionEvent,
    SignalEvent,
    SystemEvent,
)
from src.execution.handlers.notification_handler import NotificationHandler
from src.execution.handlers.trade_handler import TradeHandler


class TestEventBusIntegration:
    """Test EventBus publish/subscribe mechanics."""

    def test_publish_reaches_typed_subscriber(self) -> None:
        """Publishing event should reach typed subscriber."""
        bus = get_event_bus()
        received: list[Event] = []
        bus.subscribe(EventType.ORDER_PLACED, lambda e: received.append(e))

        event = OrderEvent(
            event_type=EventType.ORDER_PLACED,
            source="test",
            order_id="ord-1",
            ticker="KRW-BTC",
            side="buy",
            amount=0.01,
            price=50_000_000.0,
            status="pending",
        )
        bus.publish(event)

        assert len(received) == 1
        assert received[0].event_type == EventType.ORDER_PLACED

    def test_publish_does_not_reach_wrong_type(self) -> None:
        """Publishing event should not reach subscriber of different type."""
        bus = get_event_bus()
        received: list[Event] = []
        bus.subscribe(EventType.POSITION_OPENED, lambda e: received.append(e))

        event = OrderEvent(
            event_type=EventType.ORDER_PLACED,
            source="test",
            order_id="ord-1",
            ticker="KRW-BTC",
            side="buy",
        )
        bus.publish(event)

        assert len(received) == 0

    def test_multiple_subscribers_all_receive(self) -> None:
        """Multiple subscribers to same event type should all receive."""
        bus = get_event_bus()
        received_a: list[Event] = []
        received_b: list[Event] = []
        bus.subscribe(EventType.ORDER_PLACED, lambda e: received_a.append(e))
        bus.subscribe(EventType.ORDER_PLACED, lambda e: received_b.append(e))

        event = OrderEvent(event_type=EventType.ORDER_PLACED, source="test")
        bus.publish(event)

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_global_subscriber_receives_all_events(self) -> None:
        """Global subscriber should receive events of all types."""
        bus = get_event_bus()
        received: list[Event] = []
        bus.subscribe(handler=lambda e: received.append(e))

        bus.publish(OrderEvent(event_type=EventType.ORDER_PLACED, source="test"))
        bus.publish(PositionEvent(event_type=EventType.POSITION_OPENED, source="test"))
        bus.publish(SignalEvent(event_type=EventType.ENTRY_SIGNAL, source="test"))

        assert len(received) == 3

    def test_unsubscribe_stops_delivery(self) -> None:
        """Unsubscribing should stop event delivery."""
        bus = get_event_bus()
        received: list[Event] = []
        handler = lambda e: received.append(e)  # noqa: E731
        bus.subscribe(EventType.ORDER_PLACED, handler)

        bus.publish(OrderEvent(event_type=EventType.ORDER_PLACED, source="test"))
        assert len(received) == 1

        bus.unsubscribe(EventType.ORDER_PLACED, handler)
        bus.publish(OrderEvent(event_type=EventType.ORDER_PLACED, source="test"))
        assert len(received) == 1  # No new events

    def test_decorator_subscribe(self) -> None:
        """Subscribe can be used as a decorator."""
        bus = get_event_bus()
        received: list[Event] = []

        @bus.subscribe(EventType.ORDER_FILLED)
        def handle_filled(event: Event) -> None:
            received.append(event)

        bus.publish(OrderEvent(event_type=EventType.ORDER_FILLED, source="test"))
        assert len(received) == 1

    def test_subscriber_count(self) -> None:
        """get_subscriber_count should return correct counts."""
        bus = get_event_bus()
        assert bus.get_subscriber_count(EventType.ORDER_PLACED) == 0

        bus.subscribe(EventType.ORDER_PLACED, lambda e: None)
        bus.subscribe(EventType.ORDER_PLACED, lambda e: None)
        assert bus.get_subscriber_count(EventType.ORDER_PLACED) == 2

    def test_clear_removes_all_subscribers(self) -> None:
        """clear() should remove all subscribers."""
        bus = get_event_bus()
        bus.subscribe(EventType.ORDER_PLACED, lambda e: None)
        bus.subscribe(handler=lambda e: None)

        bus.clear()

        assert bus.get_subscriber_count(EventType.ORDER_PLACED) == 0
        assert bus.get_subscriber_count(None) == 0

    def test_handler_error_does_not_stop_others(self) -> None:
        """Error in one handler should not prevent other handlers from running."""
        bus = get_event_bus()
        received: list[Event] = []

        def failing_handler(event: Event) -> None:
            raise RuntimeError("Handler error")

        bus.subscribe(EventType.ORDER_PLACED, failing_handler)
        bus.subscribe(EventType.ORDER_PLACED, lambda e: received.append(e))

        bus.publish(OrderEvent(event_type=EventType.ORDER_PLACED, source="test"))

        # Second handler should still receive the event
        assert len(received) == 1


class TestHandlerAutoSubscription:
    """Test TradeHandler and NotificationHandler auto-subscribe during init."""

    def test_trade_handler_subscribes_on_init(self) -> None:
        """TradeHandler should auto-subscribe to order and position events."""
        bus = get_event_bus()
        TradeHandler()

        assert bus.get_subscriber_count(EventType.ORDER_PLACED) >= 1
        assert bus.get_subscriber_count(EventType.ORDER_FILLED) >= 1
        assert bus.get_subscriber_count(EventType.ORDER_FAILED) >= 1
        assert bus.get_subscriber_count(EventType.POSITION_OPENED) >= 1
        assert bus.get_subscriber_count(EventType.POSITION_CLOSED) >= 1

    def test_notification_handler_subscribes_on_init(self) -> None:
        """NotificationHandler should auto-subscribe to signal and system events."""
        bus = get_event_bus()
        NotificationHandler(telegram_notifier=MagicMock())

        assert bus.get_subscriber_count(EventType.ENTRY_SIGNAL) >= 1
        assert bus.get_subscriber_count(EventType.EXIT_SIGNAL) >= 1
        assert bus.get_subscriber_count(EventType.ORDER_PLACED) >= 1
        assert bus.get_subscriber_count(EventType.POSITION_OPENED) >= 1
        assert bus.get_subscriber_count(EventType.POSITION_CLOSED) >= 1
        assert bus.get_subscriber_count(EventType.ERROR) >= 1
        assert bus.get_subscriber_count(EventType.DAILY_RESET) >= 1

    def test_trade_handler_logs_order_placed(self) -> None:
        """TradeHandler should handle ORDER_PLACED event without error."""
        bus = get_event_bus()
        TradeHandler()

        event = OrderEvent(
            event_type=EventType.ORDER_PLACED,
            source="test",
            order_id="ord-1",
            ticker="KRW-BTC",
            side="buy",
            amount=0.01,
            price=50_000_000.0,
            status="pending",
        )
        bus.publish(event)  # Should not raise

    def test_trade_handler_logs_position_opened(self) -> None:
        """TradeHandler should handle POSITION_OPENED event without error."""
        bus = get_event_bus()
        TradeHandler()

        event = PositionEvent(
            event_type=EventType.POSITION_OPENED,
            source="test",
            ticker="KRW-BTC",
            action="opened",
            entry_price=50_000_000.0,
            amount=0.02,
        )
        bus.publish(event)  # Should not raise

    def test_trade_handler_logs_position_closed(self) -> None:
        """TradeHandler should handle POSITION_CLOSED event without error."""
        bus = get_event_bus()
        TradeHandler()

        event = PositionEvent(
            event_type=EventType.POSITION_CLOSED,
            source="test",
            ticker="KRW-BTC",
            action="closed",
            entry_price=50_000_000.0,
            amount=0.02,
            pnl=100_000.0,
            pnl_pct=2.0,
        )
        bus.publish(event)  # Should not raise

    def test_notification_handler_sends_entry_signal(self) -> None:
        """NotificationHandler should forward ENTRY_SIGNAL to telegram."""
        telegram = MagicMock()
        bus = get_event_bus()
        NotificationHandler(telegram_notifier=telegram)

        event = SignalEvent(
            event_type=EventType.ENTRY_SIGNAL,
            source="test",
            ticker="KRW-BTC",
            signal_type="entry",
            price=50_000_000.0,
            target_price=49_000_000.0,
        )
        bus.publish(event)

        telegram.send_trade_signal.assert_called_once()

    def test_notification_handler_sends_error(self) -> None:
        """NotificationHandler should forward ERROR event to telegram."""
        telegram = MagicMock()
        bus = get_event_bus()
        NotificationHandler(telegram_notifier=telegram)

        event = ErrorEvent(
            event_type=EventType.ERROR,
            source="test",
            error_type="ConnectionError",
            error_message="API unreachable",
        )
        bus.publish(event)

        telegram.send.assert_called_once()

    def test_notification_handler_sends_daily_reset(self) -> None:
        """NotificationHandler should forward DAILY_RESET event to telegram."""
        telegram = MagicMock()
        bus = get_event_bus()
        NotificationHandler(telegram_notifier=telegram)

        event = SystemEvent(
            event_type=EventType.DAILY_RESET,
            source="test",
            action="daily_reset",
        )
        bus.publish(event)

        telegram.send.assert_called_once_with("Daily reset completed")


class TestEventIsolation:
    """Test that EventBus is properly isolated between tests."""

    def test_fresh_event_bus_has_no_subscribers(self) -> None:
        """After reset, new EventBus should have no subscribers."""
        bus = get_event_bus()
        assert bus.get_subscriber_count(EventType.ORDER_PLACED) == 0
        assert bus.get_subscriber_count(EventType.POSITION_OPENED) == 0
        assert bus.get_subscriber_count(None) == 0

    def test_set_event_bus_none_creates_new_on_get(self) -> None:
        """set_event_bus(None) then get_event_bus() should create new instance."""
        bus1 = get_event_bus()
        bus1.subscribe(EventType.ORDER_PLACED, lambda e: None)

        set_event_bus(None)
        bus2 = get_event_bus()

        assert bus2 is not bus1
        assert bus2.get_subscriber_count(EventType.ORDER_PLACED) == 0

    def test_custom_event_bus_injection(self) -> None:
        """Setting a custom EventBus should be returned by get_event_bus."""
        custom_bus = EventBus()
        set_event_bus(custom_bus)

        assert get_event_bus() is custom_bus

    def test_handler_error_isolates_from_other_handlers(self) -> None:
        """A failing handler should not prevent other handlers from executing."""
        bus = get_event_bus()
        results: list[str] = []

        def handler_a(event: Event) -> None:
            raise ValueError("Handler A failure")

        def handler_b(event: Event) -> None:
            results.append("B ran")

        def handler_c(event: Event) -> None:
            results.append("C ran")

        bus.subscribe(EventType.ORDER_PLACED, handler_a)
        bus.subscribe(EventType.ORDER_PLACED, handler_b)
        bus.subscribe(EventType.ORDER_PLACED, handler_c)

        bus.publish(OrderEvent(event_type=EventType.ORDER_PLACED, source="test"))

        assert "B ran" in results
        assert "C ran" in results

    def test_global_handler_error_isolates(self) -> None:
        """A failing global handler should not prevent typed handlers from running."""
        bus = get_event_bus()
        results: list[str] = []

        def global_fail(event: Event) -> None:
            raise RuntimeError("Global fail")

        def typed_handler(event: Event) -> None:
            results.append("typed ran")

        bus.subscribe(handler=global_fail)
        bus.subscribe(EventType.ORDER_PLACED, typed_handler)

        bus.publish(OrderEvent(event_type=EventType.ORDER_PLACED, source="test"))

        assert "typed ran" in results
