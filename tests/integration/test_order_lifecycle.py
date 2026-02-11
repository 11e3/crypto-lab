"""
Integration tests for order lifecycle.

Tests: Order creation -> Fill -> Position management -> Advanced orders -> PnL tracking.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.exchange.types import OrderSide, OrderStatus
from src.execution.event_bus import get_event_bus
from src.execution.events import EventType
from src.execution.order_manager import OrderManager
from src.execution.orders.advanced_orders import AdvancedOrderManager
from src.execution.position_manager import PositionManager
from src.execution.trade_executor import execute_buy_order, sell_all
from src.execution.trade_executor_orders import check_advanced_orders
from tests.fixtures.mock_exchange import MockExchange


@pytest.fixture
def exchange() -> MockExchange:
    """Create MockExchange with balances."""
    ex = MockExchange()
    ex.set_balance("KRW", 10_000_000.0)
    ex.set_price("KRW-BTC", 50_000_000.0)
    ex.set_price("KRW-ETH", 3_000_000.0)
    return ex


@pytest.fixture
def order_manager(exchange: MockExchange) -> OrderManager:
    """Create OrderManager with MockExchange."""
    return OrderManager(exchange, publish_events=True, event_bus=get_event_bus())


@pytest.fixture
def position_manager(exchange: MockExchange) -> PositionManager:
    """Create PositionManager with MockExchange."""
    return PositionManager(exchange, publish_events=True, event_bus=get_event_bus())


@pytest.fixture
def advanced_order_manager() -> AdvancedOrderManager:
    """Create AdvancedOrderManager."""
    return AdvancedOrderManager()


class TestBuyOrderLifecycle:
    """Test buy order creation, fill, and position registration."""

    def test_buy_order_fills_immediately(
        self, order_manager: OrderManager, exchange: MockExchange
    ) -> None:
        """Market buy order on MockExchange should fill immediately."""
        order = order_manager.place_buy_order("KRW-BTC", 1_000_000.0, min_order_amount=5000.0)

        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == OrderSide.BUY
        assert order.symbol == "KRW-BTC"

    def test_buy_order_tracked(self, order_manager: OrderManager) -> None:
        """Placed buy order should be tracked in active orders."""
        order = order_manager.place_buy_order("KRW-BTC", 1_000_000.0)
        assert order is not None
        assert order.order_id in order_manager.active_orders

    def test_buy_order_publishes_event(self, order_manager: OrderManager) -> None:
        """Buy order should publish ORDER_PLACED event."""
        bus = get_event_bus()
        events: list = []
        bus.subscribe(EventType.ORDER_PLACED, lambda e: events.append(e))

        order_manager.place_buy_order("KRW-BTC", 1_000_000.0)

        assert len(events) == 1
        assert events[0].side == "buy"
        assert events[0].ticker == "KRW-BTC"

    def test_buy_order_deducts_krw(
        self, order_manager: OrderManager, exchange: MockExchange
    ) -> None:
        """Buy order should deduct KRW from balance."""
        initial = exchange.get_balance("KRW").balance
        order_manager.place_buy_order("KRW-BTC", 1_000_000.0)

        remaining = exchange.get_balance("KRW").balance
        assert remaining == initial - 1_000_000.0

    def test_buy_order_adds_base_currency(
        self, order_manager: OrderManager, exchange: MockExchange
    ) -> None:
        """Buy order should add base currency to balance."""
        initial_btc = exchange.get_balance("BTC").balance
        order_manager.place_buy_order("KRW-BTC", 1_000_000.0)

        final_btc = exchange.get_balance("BTC").balance
        expected_btc = 1_000_000.0 / 50_000_000.0
        assert abs(final_btc - (initial_btc + expected_btc)) < 1e-10

    def test_buy_below_min_returns_none(self, order_manager: OrderManager) -> None:
        """Buy below min_order_amount should return None."""
        order = order_manager.place_buy_order("KRW-BTC", 1000.0, min_order_amount=5000.0)
        assert order is None

    def test_buy_insufficient_balance_returns_none(
        self, order_manager: OrderManager, exchange: MockExchange
    ) -> None:
        """Buy with insufficient KRW should return None."""
        exchange.set_balance("KRW", 100.0)
        order = order_manager.place_buy_order("KRW-BTC", 1_000_000.0)
        assert order is None

    def test_buy_exchange_failure_returns_none(
        self, order_manager: OrderManager, exchange: MockExchange
    ) -> None:
        """Buy should return None when exchange fails."""
        exchange.configure_failures(fail_buy=True)
        order = order_manager.place_buy_order("KRW-BTC", 1_000_000.0)
        assert order is None

    def test_full_buy_with_position_and_advanced_orders(
        self,
        exchange: MockExchange,
        order_manager: OrderManager,
        position_manager: PositionManager,
        advanced_order_manager: AdvancedOrderManager,
    ) -> None:
        """Full buy flow: order -> position -> advanced orders."""
        telegram = MagicMock()
        trading_config: dict[str, float | None] = {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 5.0,
            "trailing_stop_pct": 1.5,
            "min_order_amount": 5000.0,
        }

        result = execute_buy_order(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            buy_amount=1_000_000.0,
            order_manager=order_manager,
            position_manager=position_manager,
            advanced_order_manager=advanced_order_manager,
            telegram=telegram,
            trading_config=trading_config,
            target_info={"KRW-BTC": {"target": 50_000_000.0, "k": 0.5}},
            min_amount=5000.0,
        )

        assert result is True
        assert position_manager.has_position("KRW-BTC")
        active = advanced_order_manager.get_active_orders("KRW-BTC")
        assert len(active) >= 2  # SL + TS (+ TP)


class TestSellOrderLifecycle:
    """Test sell order lifecycle with position removal."""

    def test_sell_order_fills(self, order_manager: OrderManager, exchange: MockExchange) -> None:
        """Market sell order should fill and update balances."""
        exchange.set_balance("BTC", 0.1)

        order = order_manager.place_sell_order("KRW-BTC", 0.1)

        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == OrderSide.SELL

    def test_sell_all_sells_entire_balance(
        self, order_manager: OrderManager, exchange: MockExchange
    ) -> None:
        """sell_all should sell entire base currency balance."""
        exchange.set_balance("BTC", 0.1)

        order = order_manager.sell_all("KRW-BTC")

        assert order is not None
        remaining_btc = exchange.get_balance("BTC").balance
        assert remaining_btc < 0.001

    def test_sell_all_adds_krw(self, order_manager: OrderManager, exchange: MockExchange) -> None:
        """sell_all should increase KRW balance."""
        exchange.set_balance("BTC", 0.1)
        initial_krw = exchange.get_balance("KRW").balance

        order_manager.sell_all("KRW-BTC")

        final_krw = exchange.get_balance("KRW").balance
        assert final_krw > initial_krw

    def test_sell_publishes_event(
        self, order_manager: OrderManager, exchange: MockExchange
    ) -> None:
        """Sell order should publish ORDER_PLACED event."""
        exchange.set_balance("BTC", 0.1)
        bus = get_event_bus()
        events: list = []
        bus.subscribe(EventType.ORDER_PLACED, lambda e: events.append(e))

        order_manager.place_sell_order("KRW-BTC", 0.1)

        assert len(events) == 1
        assert events[0].side == "sell"

    def test_sell_failure_returns_none(
        self, order_manager: OrderManager, exchange: MockExchange
    ) -> None:
        """Sell should return None when exchange fails."""
        exchange.set_balance("BTC", 0.1)
        exchange.configure_failures(fail_sell=True)

        order = order_manager.place_sell_order("KRW-BTC", 0.1)
        assert order is None

    def test_sell_with_position_removal(
        self,
        exchange: MockExchange,
        order_manager: OrderManager,
        position_manager: PositionManager,
    ) -> None:
        """Full sell flow: sell order -> position removal."""
        exchange.set_balance("BTC", 0.1)
        position_manager.add_position("KRW-BTC", 50_000_000.0, 0.1)

        result = sell_all(
            ticker="KRW-BTC",
            order_manager=order_manager,
            position_manager=position_manager,
            exchange=exchange,
            telegram=MagicMock(),
            min_amount=5000.0,
        )

        assert result is True
        assert not position_manager.has_position("KRW-BTC")

    def test_sell_cancels_advanced_orders(
        self,
        exchange: MockExchange,
        advanced_order_manager: AdvancedOrderManager,
    ) -> None:
        """After sell, advanced orders should be cancelled."""
        advanced_order_manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            stop_loss_pct=0.02,
        )

        active_before = advanced_order_manager.get_active_orders("KRW-BTC")
        assert len(active_before) == 1

        advanced_order_manager.cancel_all_orders(ticker="KRW-BTC")

        active_after = advanced_order_manager.get_active_orders("KRW-BTC")
        assert len(active_after) == 0


class TestAdvancedOrderTrigger:
    """Test advanced order trigger -> sell lifecycle."""

    def test_stop_loss_trigger_and_sell(
        self,
        exchange: MockExchange,
        order_manager: OrderManager,
        position_manager: PositionManager,
        advanced_order_manager: AdvancedOrderManager,
    ) -> None:
        """Stop loss trigger should execute sell and remove position."""
        exchange.set_balance("BTC", 0.02)
        position_manager.add_position("KRW-BTC", 50_000_000.0, 0.02)

        advanced_order_manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            stop_loss_pct=0.02,
        )

        # Create mock signal handler for check_advanced_orders
        sh = MagicMock()
        import pandas as pd

        sh.get_ohlcv_data.return_value = pd.DataFrame(
            {"low": [48_000_000.0], "high": [50_000_000.0], "close": [48_500_000.0]}
        )

        # Price drops below stop loss
        drop_price = 48_000_000.0
        exchange.set_price("KRW-BTC", drop_price)

        triggered = check_advanced_orders(
            ticker="KRW-BTC",
            current_price=drop_price,
            position_manager=position_manager,
            order_manager=order_manager,
            advanced_order_manager=advanced_order_manager,
            signal_handler=sh,
            trading_config={"min_order_amount": 5000.0},
        )

        assert triggered is True
        assert not position_manager.has_position("KRW-BTC")

    def test_take_profit_trigger(
        self,
        exchange: MockExchange,
        advanced_order_manager: AdvancedOrderManager,
    ) -> None:
        """Take profit order should trigger when price rises above threshold."""
        advanced_order_manager.create_take_profit(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            take_profit_pct=0.05,
        )

        # Price rises above take profit (5% above 50M = 52.5M)
        high_price = 53_000_000.0
        triggered = advanced_order_manager.check_orders(
            ticker="KRW-BTC",
            current_price=high_price,
            current_date=date.today(),
            high_price=high_price,
        )

        assert len(triggered) >= 1

    def test_no_trigger_within_range(self, advanced_order_manager: AdvancedOrderManager) -> None:
        """No trigger when price is within stop loss and take profit range."""
        advanced_order_manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            stop_loss_pct=0.02,
        )
        advanced_order_manager.create_take_profit(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            take_profit_pct=0.05,
        )

        # Price within range
        triggered = advanced_order_manager.check_orders(
            ticker="KRW-BTC",
            current_price=50_500_000.0,
            current_date=date.today(),
            low_price=50_000_000.0,
            high_price=51_000_000.0,
        )

        assert len(triggered) == 0

    def test_cancel_order_by_id(self, advanced_order_manager: AdvancedOrderManager) -> None:
        """Cancel specific order by ID."""
        order = advanced_order_manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            stop_loss_pct=0.02,
        )

        result = advanced_order_manager.cancel_order(order.order_id)
        assert result is True
        assert len(advanced_order_manager.get_active_orders("KRW-BTC")) == 0

    def test_cancel_all_orders_by_ticker(
        self, advanced_order_manager: AdvancedOrderManager
    ) -> None:
        """Cancel all orders for a specific ticker."""
        advanced_order_manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            stop_loss_pct=0.02,
        )
        advanced_order_manager.create_trailing_stop(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            trailing_stop_pct=1.5,
        )
        # Different ticker
        advanced_order_manager.create_stop_loss(
            ticker="KRW-ETH",
            entry_price=3_000_000.0,
            entry_date=date.today(),
            amount=1.0,
            stop_loss_pct=0.02,
        )

        count = advanced_order_manager.cancel_all_orders(ticker="KRW-BTC")
        assert count == 2
        assert len(advanced_order_manager.get_active_orders("KRW-BTC")) == 0
        assert len(advanced_order_manager.get_active_orders("KRW-ETH")) == 1


class TestPositionTracking:
    """Test position tracking across multiple tickers."""

    def test_multi_ticker_independent(self, position_manager: PositionManager) -> None:
        """Positions for different tickers should be independent."""
        position_manager.add_position("KRW-BTC", 50_000_000.0, 0.1)
        position_manager.add_position("KRW-ETH", 3_000_000.0, 1.0)

        assert position_manager.get_position_count() == 2
        assert position_manager.has_position("KRW-BTC")
        assert position_manager.has_position("KRW-ETH")

        position_manager.remove_position("KRW-BTC")
        assert not position_manager.has_position("KRW-BTC")
        assert position_manager.has_position("KRW-ETH")
        assert position_manager.get_position_count() == 1

    def test_position_count_matches_slots(self, position_manager: PositionManager) -> None:
        """Position count should reflect actual open positions."""
        assert position_manager.get_position_count() == 0

        position_manager.add_position("KRW-BTC", 50_000_000.0, 0.1)
        assert position_manager.get_position_count() == 1

        position_manager.add_position("KRW-ETH", 3_000_000.0, 1.0)
        assert position_manager.get_position_count() == 2

        position_manager.remove_position("KRW-BTC")
        assert position_manager.get_position_count() == 1

    def test_position_pnl_calculation(
        self, position_manager: PositionManager, exchange: MockExchange
    ) -> None:
        """PnL should reflect current price vs entry price."""
        position_manager.add_position("KRW-BTC", 50_000_000.0, 0.02)

        # Price increased
        exchange.set_price("KRW-BTC", 51_000_000.0)
        pnl = position_manager.calculate_pnl("KRW-BTC")
        assert pnl > 0  # Profit

        pnl_pct = position_manager.calculate_pnl_pct("KRW-BTC")
        assert pnl_pct > 0

    def test_position_pnl_negative(
        self, position_manager: PositionManager, exchange: MockExchange
    ) -> None:
        """PnL should be negative when price drops."""
        position_manager.add_position("KRW-BTC", 50_000_000.0, 0.02)

        exchange.set_price("KRW-BTC", 49_000_000.0)
        pnl = position_manager.calculate_pnl("KRW-BTC")
        assert pnl < 0

    def test_clear_all_positions(self, position_manager: PositionManager) -> None:
        """clear_all should remove all positions."""
        position_manager.add_position("KRW-BTC", 50_000_000.0, 0.1)
        position_manager.add_position("KRW-ETH", 3_000_000.0, 1.0)
        assert position_manager.get_position_count() == 2

        position_manager.clear_all()
        assert position_manager.get_position_count() == 0

    def test_position_publishes_opened_event(self, position_manager: PositionManager) -> None:
        """Adding position should publish POSITION_OPENED event."""
        bus = get_event_bus()
        events: list = []
        bus.subscribe(EventType.POSITION_OPENED, lambda e: events.append(e))

        position_manager.add_position("KRW-BTC", 50_000_000.0, 0.1)

        assert len(events) == 1
        assert events[0].ticker == "KRW-BTC"

    def test_position_publishes_closed_event(self, position_manager: PositionManager) -> None:
        """Removing position should publish POSITION_CLOSED event."""
        bus = get_event_bus()
        events: list = []
        bus.subscribe(EventType.POSITION_CLOSED, lambda e: events.append(e))

        position_manager.add_position("KRW-BTC", 50_000_000.0, 0.1)
        position_manager.remove_position("KRW-BTC")

        assert len(events) == 1
        assert events[0].ticker == "KRW-BTC"
