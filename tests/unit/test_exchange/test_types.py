"""
Unit tests for exchange types.
"""

from datetime import datetime

from src.exchange.types import (
    Balance,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
)


class TestOrderSide:
    """Test cases for OrderSide enum."""

    def test_order_side_values(self) -> None:
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestOrderType:
    """Test cases for OrderType enum."""

    def test_order_type_values(self) -> None:
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"


class TestOrderStatus:
    """Test cases for OrderStatus enum."""

    def test_order_status_values(self) -> None:
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.FAILED.value == "failed"


class TestBalance:
    """Test cases for Balance dataclass."""

    def test_balance_initialization(self) -> None:
        """Test Balance initialization."""
        balance = Balance(currency="KRW", balance=1000.0, locked=100.0)
        assert balance.currency == "KRW"
        assert balance.balance == 1000.0
        assert balance.locked == 100.0

    def test_balance_available(self) -> None:
        """Test Balance available property."""
        balance = Balance(currency="KRW", balance=1000.0, locked=100.0)
        assert balance.available == 900.0

    def test_balance_available_no_locked(self) -> None:
        """Test Balance available property when locked is 0."""
        balance = Balance(currency="KRW", balance=1000.0)
        assert balance.available == 1000.0


class TestTicker:
    """Test cases for Ticker dataclass."""

    def test_ticker_initialization(self) -> None:
        """Test Ticker initialization."""
        ticker = Ticker(symbol="KRW-BTC", price=50000.0, volume=100.0)
        assert ticker.symbol == "KRW-BTC"
        assert ticker.price == 50000.0
        assert ticker.volume == 100.0

    def test_ticker_with_timestamp(self) -> None:
        """Test Ticker with timestamp."""
        timestamp = datetime.now()
        ticker = Ticker(symbol="KRW-BTC", price=50000.0, timestamp=timestamp)
        assert ticker.timestamp == timestamp


class TestOrder:
    """Test cases for Order dataclass."""

    def test_order_initialization(self) -> None:
        """Test Order initialization."""
        order = Order(
            order_id="order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
        )
        assert order.order_id == "order-123"
        assert order.symbol == "KRW-BTC"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.amount == 0.001
        assert order.status == OrderStatus.PENDING

    def test_order_is_filled_true(self) -> None:
        """Test Order is_filled property when status is FILLED."""
        order = Order(
            order_id="order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            status=OrderStatus.FILLED,
        )
        assert order.is_filled is True

    def test_order_is_filled_false(self) -> None:
        """Test Order is_filled property when status is not FILLED."""
        order = Order(
            order_id="order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            status=OrderStatus.PENDING,
        )
        assert order.is_filled is False

    def test_order_is_active_pending(self) -> None:
        """Test Order is_active property when status is PENDING (line 84)."""
        order = Order(
            order_id="order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            status=OrderStatus.PENDING,
        )
        assert order.is_active is True

    def test_order_is_active_partially_filled(self) -> None:
        """Test Order is_active property when status is PARTIALLY_FILLED (line 84)."""
        order = Order(
            order_id="order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            status=OrderStatus.PARTIALLY_FILLED,
        )
        assert order.is_active is True

    def test_order_is_active_filled(self) -> None:
        """Test Order is_active property when status is FILLED (line 84)."""
        order = Order(
            order_id="order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            status=OrderStatus.FILLED,
        )
        assert order.is_active is False

    def test_order_is_active_cancelled(self) -> None:
        """Test Order is_active property when status is CANCELLED (line 84)."""
        order = Order(
            order_id="order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            status=OrderStatus.CANCELLED,
        )
        assert order.is_active is False

    def test_order_is_active_failed(self) -> None:
        """Test Order is_active property when status is FAILED (line 84)."""
        order = Order(
            order_id="order-123",
            symbol="KRW-BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001,
            status=OrderStatus.FAILED,
        )
        assert order.is_active is False
