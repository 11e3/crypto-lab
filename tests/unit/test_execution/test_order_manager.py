"""
Unit tests for OrderManager.
"""

from unittest.mock import MagicMock, patch

from src.exchange import ExchangeOrderError
from src.exchange.types import Order, OrderSide, OrderStatus, OrderType
from src.execution.event_bus import EventBus, set_event_bus
from src.execution.events import EventType, OrderEvent
from src.execution.order_manager import OrderManager
from tests.fixtures.mock_exchange import MockExchange


class TestOrderManager:
    """Test cases for OrderManager."""

    def test_initialization(self, mock_exchange: MockExchange) -> None:
        """Test OrderManager initialization."""
        manager = OrderManager(mock_exchange)
        assert manager.exchange == mock_exchange
        assert len(manager.active_orders) == 0
        assert manager.publish_events is True

    def test_initialization_publish_events_false(self, mock_exchange: MockExchange) -> None:
        """Test OrderManager initialization with publish_events=False."""
        manager = OrderManager(mock_exchange, publish_events=False)
        assert manager.publish_events is False
        assert manager.event_bus is None

    def test_place_buy_order(self, mock_exchange: MockExchange) -> None:
        """Test placing a buy order."""
        manager = OrderManager(mock_exchange, publish_events=False)
        order = manager.place_buy_order("KRW-BTC", 50000.0)

        assert order is not None
        assert order.order_id in manager.active_orders
        assert manager.active_orders[order.order_id] == order
        assert order.side == OrderSide.BUY

    def test_place_buy_order_with_events(self, mock_exchange: MockExchange) -> None:
        """Test placing a buy order with event publishing."""
        event_bus = EventBus()
        set_event_bus(event_bus)
        handler = MagicMock()
        event_bus.subscribe(EventType.ORDER_PLACED, handler)

        manager = OrderManager(mock_exchange, publish_events=True)
        order = manager.place_buy_order("KRW-BTC", 50000.0)

        assert order is not None
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert isinstance(event, OrderEvent)
        assert event.event_type == EventType.ORDER_PLACED
        assert event.order_id == order.order_id
        assert event.ticker == "KRW-BTC"
        assert event.side == "buy"

    def test_place_buy_order_below_minimum(self, mock_exchange: MockExchange) -> None:
        """Test buy order below minimum amount."""
        manager = OrderManager(mock_exchange, publish_events=False)

        order = manager.place_buy_order("KRW-BTC", 1000.0, min_order_amount=50000.0)
        assert order is None

    def test_place_buy_order_insufficient_balance(self, mock_exchange: MockExchange) -> None:
        """Test buy order with insufficient balance."""
        mock_exchange.set_balance("KRW", 1000.0)
        manager = OrderManager(mock_exchange, publish_events=False)

        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is None

    def test_place_buy_order_exchange_error(self, mock_exchange: MockExchange) -> None:
        """Test buy order with exchange error."""
        mock_exchange.buy_market_order = MagicMock(side_effect=ExchangeOrderError("Exchange error"))
        manager = OrderManager(mock_exchange, publish_events=False)

        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is None

    def test_place_buy_order_unexpected_error(self, mock_exchange: MockExchange) -> None:
        """Test buy order with unexpected error."""
        mock_exchange.buy_market_order = MagicMock(side_effect=ConnectionError("Unexpected error"))
        manager = OrderManager(mock_exchange, publish_events=False)

        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is None

    def test_place_sell_order(self, mock_exchange: MockExchange) -> None:
        """Test placing a sell order."""
        manager = OrderManager(mock_exchange, publish_events=False)
        buy_order = manager.place_buy_order("KRW-BTC", 50_000.0)
        assert buy_order is not None

        order = manager.place_sell_order("KRW-BTC", 0.001)

        assert order is not None
        assert order.order_id in manager.active_orders
        assert order.side == OrderSide.SELL

    def test_place_sell_order_with_events(self, mock_exchange: MockExchange) -> None:
        """Test placing a sell order with event publishing."""
        event_bus = EventBus()
        set_event_bus(event_bus)
        handler = MagicMock()
        event_bus.subscribe(EventType.ORDER_PLACED, handler)

        manager = OrderManager(mock_exchange, publish_events=True)
        buy_order = manager.place_buy_order("KRW-BTC", 50_000.0)
        assert buy_order is not None

        handler.reset_mock()
        order = manager.place_sell_order("KRW-BTC", 0.001)

        assert order is not None
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert isinstance(event, OrderEvent)
        assert event.side == "sell"

    def test_place_sell_order_below_minimum(self, mock_exchange: MockExchange) -> None:
        """Test sell order below minimum order value."""
        manager = OrderManager(mock_exchange, publish_events=False)
        buy_order = manager.place_buy_order("KRW-BTC", 50_000.0)
        assert buy_order is not None

        # Try to sell very small amount (below minimum)
        order = manager.place_sell_order("KRW-BTC", 0.000001, min_order_amount=50000.0)
        assert order is None

    def test_place_sell_order_insufficient_balance(self, mock_exchange: MockExchange) -> None:
        """Test sell order with insufficient balance."""
        manager = OrderManager(mock_exchange, publish_events=False)

        order = manager.place_sell_order("KRW-BTC", 0.001)
        assert order is None

    def test_place_sell_order_exchange_error(self, mock_exchange: MockExchange) -> None:
        """Test sell order with exchange error."""
        manager = OrderManager(mock_exchange, publish_events=False)
        buy_order = manager.place_buy_order("KRW-BTC", 50_000.0)
        assert buy_order is not None

        mock_exchange.sell_market_order = MagicMock(
            side_effect=ExchangeOrderError("Exchange error")
        )
        order = manager.place_sell_order("KRW-BTC", 0.001)
        assert order is None

    def test_place_sell_order_unexpected_error(self, mock_exchange: MockExchange) -> None:
        """Test sell order with unexpected error."""
        manager = OrderManager(mock_exchange, publish_events=False)
        buy_order = manager.place_buy_order("KRW-BTC", 50_000.0)
        assert buy_order is not None

        mock_exchange.sell_market_order = MagicMock(side_effect=ConnectionError("Unexpected error"))
        order = manager.place_sell_order("KRW-BTC", 0.001)
        assert order is None

    def test_sell_all(self, mock_exchange: MockExchange) -> None:
        """Test selling all holdings."""
        manager = OrderManager(mock_exchange, publish_events=False)
        buy_order = manager.place_buy_order("KRW-BTC", 50_000.0)
        assert buy_order is not None

        order = manager.sell_all("KRW-BTC")

        assert order is not None
        assert order.side == OrderSide.SELL

    def test_sell_all_no_balance(self, mock_exchange: MockExchange) -> None:
        """Test sell_all when no balance exists."""
        manager = OrderManager(mock_exchange, publish_events=False)

        order = manager.sell_all("KRW-BTC")
        assert order is None

    def test_sell_all_with_minimum(self, mock_exchange: MockExchange) -> None:
        """Test sell_all with minimum order amount."""
        manager = OrderManager(mock_exchange, publish_events=False)
        buy_order = manager.place_buy_order("KRW-BTC", 50_000.0)
        assert buy_order is not None

        order = manager.sell_all("KRW-BTC", min_order_amount=0.0)
        assert order is not None

    def test_sell_all_error(self, mock_exchange: MockExchange) -> None:
        """Test sell_all when error occurs."""
        manager = OrderManager(mock_exchange, publish_events=False)
        buy_order = manager.place_buy_order("KRW-BTC", 50_000.0)
        assert buy_order is not None

        mock_exchange.get_balance = MagicMock(side_effect=ValueError("Error"))
        order = manager.sell_all("KRW-BTC")
        assert order is None

    def test_get_order_status(self, mock_exchange: MockExchange) -> None:
        """Test getting order status."""
        manager = OrderManager(mock_exchange, publish_events=False)
        order = manager.place_buy_order("KRW-BTC", 50000.0)

        assert order is not None
        status = manager.get_order_status(order.order_id)
        assert status is not None
        assert status.order_id == order.order_id

    def test_get_order_status_with_status_change(self, mock_exchange: MockExchange) -> None:
        """Test get_order_status when status changes and publishes ORDER_FILLED event (line 194)."""
        event_bus = EventBus()
        set_event_bus(event_bus)
        handler = MagicMock()
        event_bus.subscribe(EventType.ORDER_FILLED, handler)

        manager = OrderManager(mock_exchange, publish_events=True)
        # Create a PENDING order manually (MockExchange returns FILLED by default)
        pending_order = Order(
            order_id="test-order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.PENDING,
        )
        manager.active_orders["test-order-123"] = pending_order

        # Create a new order with filled status (different from pending)
        filled_order = Order(
            order_id="test-order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.FILLED,
            filled_price=50000.0,
        )
        mock_exchange.get_order_status = MagicMock(return_value=filled_order)

        handler.reset_mock()
        status = manager.get_order_status("test-order-123")

        assert status is not None
        assert status.status == OrderStatus.FILLED
        # Event should be published when status changes from PENDING to FILLED (line 194)
        handler.assert_called_once()

    def test_get_order_status_cancelled(self, mock_exchange: MockExchange) -> None:
        """Test get_order_status when order is cancelled."""
        event_bus = EventBus()
        set_event_bus(event_bus)
        handler = MagicMock()
        event_bus.subscribe(EventType.ORDER_CANCELLED, handler)

        manager = OrderManager(mock_exchange, publish_events=True)
        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is not None

        cancelled_order = Order(
            order_id=order.order_id,
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.CANCELLED,
        )
        with patch.object(mock_exchange, "get_order_status", return_value=cancelled_order):
            handler.reset_mock()
            status = manager.get_order_status(order.order_id)

            assert status is not None
            handler.assert_called_once()

    def test_get_order_status_failed(self, mock_exchange: MockExchange) -> None:
        """Test get_order_status when order failed."""
        event_bus = EventBus()
        set_event_bus(event_bus)
        handler = MagicMock()
        event_bus.subscribe(EventType.ORDER_FAILED, handler)

        manager = OrderManager(mock_exchange, publish_events=True)
        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is not None

        failed_order = Order(
            order_id=order.order_id,
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.FAILED,
        )
        with patch.object(mock_exchange, "get_order_status", return_value=failed_order):
            handler.reset_mock()
            status = manager.get_order_status(order.order_id)

            assert status is not None
            handler.assert_called_once()

    def test_get_order_status_error(self, mock_exchange: MockExchange) -> None:
        """Test get_order_status when error occurs."""
        manager = OrderManager(mock_exchange, publish_events=False)
        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is not None

        mock_exchange.get_order_status = MagicMock(side_effect=ConnectionError("Error"))
        status = manager.get_order_status(order.order_id)

        assert status is None

    def test_get_order_status_new_order(self, mock_exchange: MockExchange) -> None:
        """Test get_order_status for order not in cache."""
        manager = OrderManager(mock_exchange, publish_events=False)

        new_order = Order(
            order_id="new-order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.FILLED,
            filled_price=50000.0,
        )
        with patch.object(mock_exchange, "get_order_status", return_value=new_order):
            status = manager.get_order_status("new-order-123")
            assert status is not None
            assert status.order_id == "new-order-123"

    def test_get_order_status_placed_event(self, mock_exchange: MockExchange) -> None:
        """Test get_order_status publishes ORDER_PLACED event for other status changes (line 200)."""
        event_bus = EventBus()
        set_event_bus(event_bus)
        handler = MagicMock()
        event_bus.subscribe(EventType.ORDER_PLACED, handler)

        manager = OrderManager(mock_exchange, publish_events=True)
        # Create an order with PENDING status
        pending_order = Order(
            order_id="test-order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.PENDING,
        )
        manager.active_orders["test-order-123"] = pending_order

        # Create a new order with PARTIALLY_FILLED status (not filled, not cancelled, not failed)
        partially_filled_order = Order(
            order_id="test-order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_amount=0.0005,
        )
        mock_exchange.get_order_status = MagicMock(return_value=partially_filled_order)

        handler.reset_mock()
        status = manager.get_order_status("test-order-123")

        assert status is not None
        assert status.status == OrderStatus.PARTIALLY_FILLED
        # Event should be published with ORDER_PLACED for other status changes (line 200)
        handler.assert_called_once()

    def test_cancel_order(self, mock_exchange: MockExchange) -> None:
        """Test canceling an order."""
        manager = OrderManager(mock_exchange, publish_events=False)
        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is not None
        assert order.order_id in manager.active_orders

        mock_exchange.cancel_order = MagicMock(return_value=True)
        success = manager.cancel_order(order.order_id)

        assert success is True
        assert order.order_id not in manager.active_orders

    def test_cancel_order_failure(self, mock_exchange: MockExchange) -> None:
        """Test cancel_order when cancellation fails."""
        manager = OrderManager(mock_exchange, publish_events=False)
        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is not None

        mock_exchange.cancel_order = MagicMock(return_value=False)
        success = manager.cancel_order(order.order_id)

        assert success is False
        assert order.order_id in manager.active_orders  # Should still be in cache

    def test_cancel_order_error(self, mock_exchange: MockExchange) -> None:
        """Test cancel_order when error occurs."""
        manager = OrderManager(mock_exchange, publish_events=False)
        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is not None

        mock_exchange.cancel_order = MagicMock(side_effect=ConnectionError("Error"))
        success = manager.cancel_order(order.order_id)

        assert success is False

    def test_get_active_orders(self, mock_exchange: MockExchange) -> None:
        """Test getting all active orders."""
        manager = OrderManager(mock_exchange, publish_events=False)

        order1 = manager.place_buy_order("KRW-BTC", 50000.0)
        order2 = manager.place_buy_order("KRW-ETH", 30000.0)

        assert order1 is not None
        assert order2 is not None

        active_orders = manager.get_active_orders()
        assert len(active_orders) == 2
        assert order1.order_id in active_orders
        assert order2.order_id in active_orders

        # Should return a copy
        active_orders.clear()
        assert len(manager.active_orders) == 2

    def test_clear_filled_orders(self, mock_exchange: MockExchange) -> None:
        """Test clearing filled orders."""
        manager = OrderManager(mock_exchange, publish_events=False)
        order = manager.place_buy_order("KRW-BTC", 50000.0)
        assert order is not None

        # Mark order as filled by updating status
        filled_order = Order(
            order_id=order.order_id,
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.FILLED,
            filled_price=50000.0,
        )
        manager.active_orders[order.order_id] = filled_order

        manager.clear_filled_orders()

        assert order.order_id not in manager.active_orders

    def test_clear_filled_orders_multiple(self, mock_exchange: MockExchange) -> None:
        """Test clearing multiple filled orders."""
        manager = OrderManager(mock_exchange, publish_events=False)
        order1 = manager.place_buy_order("KRW-BTC", 50000.0)
        order2 = manager.place_buy_order("KRW-ETH", 30000.0)
        assert order1 is not None
        assert order2 is not None

        # Mark both as filled
        filled_order1 = Order(
            order_id=order1.order_id,
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.FILLED,
            filled_price=50000.0,
        )
        filled_order2 = Order(
            order_id=order2.order_id,
            symbol="KRW-ETH",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.01,
            price=3000.0,
            status=OrderStatus.FILLED,
            filled_price=3000.0,
        )
        manager.active_orders[order1.order_id] = filled_order1
        manager.active_orders[order2.order_id] = filled_order2

        manager.clear_filled_orders()

        assert order1.order_id not in manager.active_orders
        assert order2.order_id not in manager.active_orders

    def test_clear_filled_orders_no_filled(self, mock_exchange: MockExchange) -> None:
        """Test clear_filled_orders when no filled orders exist."""
        manager = OrderManager(mock_exchange, publish_events=False)
        # MockExchange returns FILLED orders by default, so we need to create a PENDING order manually
        pending_order = Order(
            order_id="pending-order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status=OrderStatus.PENDING,
        )
        manager.active_orders["pending-order-123"] = pending_order

        manager.clear_filled_orders()

        # PENDING order should still be in cache (not filled)
        assert "pending-order-123" in manager.active_orders
