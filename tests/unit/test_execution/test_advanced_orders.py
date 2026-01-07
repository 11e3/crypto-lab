"""
Tests for advanced order types (stop loss, take profit, trailing stop).
"""

from datetime import date

import pytest

from src.execution.advanced_orders import (
    AdvancedOrder,
    AdvancedOrderManager,
    OrderType,
)

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def manager() -> AdvancedOrderManager:
    """Create an AdvancedOrderManager instance."""
    return AdvancedOrderManager()


@pytest.fixture
def sample_date() -> date:
    """Create a sample date."""
    return date(2023, 1, 1)


# -------------------------------------------------------------------------
# AdvancedOrder Tests
# -------------------------------------------------------------------------


class TestAdvancedOrder:
    """Test AdvancedOrder dataclass."""

    def test_advanced_order_creation_stop_loss(self, sample_date: date) -> None:
        """Test creating a stop loss order."""
        order = AdvancedOrder(
            order_id="sl_001",
            ticker="KRW-BTC",
            order_type=OrderType.STOP_LOSS,
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        assert order.order_id == "sl_001"
        assert order.ticker == "KRW-BTC"
        assert order.order_type == OrderType.STOP_LOSS
        assert order.entry_price == 50000.0
        assert order.stop_loss_price == 45000.0
        assert order.is_active is True
        assert order.is_triggered is False

    def test_advanced_order_creation_take_profit(self, sample_date: date) -> None:
        """Test creating a take profit order."""
        order = AdvancedOrder(
            order_id="tp_001",
            ticker="KRW-ETH",
            order_type=OrderType.TAKE_PROFIT,
            entry_price=2000.0,
            entry_date=sample_date,
            amount=10.0,
            take_profit_price=2200.0,
        )

        assert order.order_type == OrderType.TAKE_PROFIT
        assert order.take_profit_price == 2200.0
        assert order.is_active is True

    def test_advanced_order_creation_trailing_stop(self, sample_date: date) -> None:
        """Test creating a trailing stop order."""
        order = AdvancedOrder(
            order_id="ts_001",
            ticker="KRW-XRP",
            order_type=OrderType.TRAILING_STOP,
            entry_price=1000.0,
            entry_date=sample_date,
            amount=100.0,
            trailing_stop_pct=0.05,
            highest_price=1000.0,
        )

        assert order.order_type == OrderType.TRAILING_STOP
        assert order.trailing_stop_pct == 0.05
        assert order.highest_price == 1000.0

    def test_advanced_order_repr(self, sample_date: date) -> None:
        """Test string representation of order."""
        order = AdvancedOrder(
            order_id="sl_001",
            ticker="KRW-BTC",
            order_type=OrderType.STOP_LOSS,
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        repr_str = repr(order)
        assert "STOP_LOSS" in repr_str or "stop_loss" in repr_str
        assert "KRW-BTC" in repr_str
        assert "50000" in repr_str


# -------------------------------------------------------------------------
# Create Stop Loss Tests
# -------------------------------------------------------------------------


class TestCreateStopLoss:
    """Test creating stop loss orders."""

    def test_create_stop_loss_with_price(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test creating stop loss with absolute price."""
        order = manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        assert order.order_type == OrderType.STOP_LOSS
        assert order.stop_loss_price == 45000.0
        assert order.is_active is True
        assert order.order_id in manager.orders

    def test_create_stop_loss_with_percentage(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test creating stop loss with percentage."""
        order = manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_pct=0.1,  # 10% below entry
        )

        assert order.stop_loss_price == 45000.0  # 50000 * (1 - 0.1)
        assert order.stop_loss_pct == 0.1

    def test_create_stop_loss_no_params_raises_error(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test that creating stop loss without params raises error."""
        with pytest.raises(ValueError):
            manager.create_stop_loss(
                ticker="KRW-BTC",
                entry_price=50000.0,
                entry_date=sample_date,
                amount=1.0,
            )

    def test_create_multiple_stop_losses(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test creating multiple stop loss orders."""
        order1 = manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        order2 = manager.create_stop_loss(
            ticker="KRW-ETH",
            entry_price=2000.0,
            entry_date=sample_date,
            amount=10.0,
            stop_loss_price=1800.0,
        )

        assert len(manager.orders) == 2
        assert order1.order_id != order2.order_id


# -------------------------------------------------------------------------
# Create Take Profit Tests
# -------------------------------------------------------------------------


class TestCreateTakeProfit:
    """Test creating take profit orders."""

    def test_create_take_profit_with_price(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test creating take profit with absolute price."""
        order = manager.create_take_profit(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            take_profit_price=55000.0,
        )

        assert order.order_type == OrderType.TAKE_PROFIT
        assert order.take_profit_price == 55000.0
        assert order.is_active is True

    def test_create_take_profit_with_percentage(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test creating take profit with percentage."""
        order = manager.create_take_profit(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            take_profit_pct=0.1,  # 10% above entry
        )

        assert order.take_profit_price is not None
        assert abs(order.take_profit_price - 55000.0) < 0.01  # 50000 * (1 + 0.1)
        assert order.take_profit_pct == 0.1

    def test_create_take_profit_no_params_raises_error(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test that creating take profit without params raises error."""
        with pytest.raises(ValueError):
            manager.create_take_profit(
                ticker="KRW-BTC",
                entry_price=50000.0,
                entry_date=sample_date,
                amount=1.0,
            )


# -------------------------------------------------------------------------
# Create Trailing Stop Tests
# -------------------------------------------------------------------------


class TestCreateTrailingStop:
    """Test creating trailing stop orders."""

    def test_create_trailing_stop_basic(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test creating basic trailing stop."""
        order = manager.create_trailing_stop(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            trailing_stop_pct=0.05,  # Trail 5% from peak
        )

        assert order.order_type == OrderType.TRAILING_STOP
        assert order.trailing_stop_pct == 0.05
        assert order.highest_price == 50000.0
        assert order.stop_loss_price == 47500.0  # 50000 * (1 - 0.05)

    def test_create_trailing_stop_with_initial_sl(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test creating trailing stop with custom initial stop loss."""
        order = manager.create_trailing_stop(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            trailing_stop_pct=0.05,
            initial_stop_loss_pct=0.1,  # Initial 10% stop loss
        )

        assert order.trailing_stop_pct == 0.05
        assert order.stop_loss_price == 45000.0  # 50000 * (1 - 0.1)


# -------------------------------------------------------------------------
# Check Orders Tests
# -------------------------------------------------------------------------


class TestCheckOrders:
    """Test checking and triggering orders."""

    def test_check_stop_loss_not_triggered(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test stop loss not triggered when price is above threshold."""
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        triggered = manager.check_orders(
            ticker="KRW-BTC",
            current_price=48000.0,
            current_date=sample_date,
            low_price=48000.0,
        )

        assert len(triggered) == 0

    def test_check_stop_loss_triggered(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test stop loss triggered when price touches threshold."""
        order = manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        triggered = manager.check_orders(
            ticker="KRW-BTC",
            current_price=44000.0,
            current_date=sample_date,
            low_price=44000.0,
        )

        assert len(triggered) == 1
        assert triggered[0].order_id == order.order_id
        assert triggered[0].is_triggered is True
        assert triggered[0].is_active is False

    def test_check_take_profit_triggered(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test take profit triggered when price reaches target."""
        manager.create_take_profit(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            take_profit_price=55000.0,
        )

        triggered = manager.check_orders(
            ticker="KRW-BTC",
            current_price=56000.0,
            current_date=sample_date,
            high_price=56000.0,
        )

        assert len(triggered) == 1
        assert triggered[0].is_triggered is True

    def test_check_trailing_stop_updates_high(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test trailing stop updates highest price."""
        order = manager.create_trailing_stop(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            trailing_stop_pct=0.05,
        )

        # Price goes up to 55000
        manager.check_orders(
            ticker="KRW-BTC",
            current_price=55000.0,
            current_date=sample_date,
            high_price=55000.0,
        )

        # Highest price should be updated
        assert order.highest_price == 55000.0
        # Stop loss should trail from new high: 55000 * (1 - 0.05) = 52250
        assert order.stop_loss_price == 52250.0

    def test_check_trailing_stop_triggered(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test trailing stop triggered when price drops below trailing threshold."""
        manager.create_trailing_stop(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            trailing_stop_pct=0.05,
        )

        # Price goes up
        manager.check_orders(
            ticker="KRW-BTC",
            current_price=55000.0,
            current_date=sample_date,
            high_price=55000.0,
        )

        # Price drops below trailing stop
        triggered = manager.check_orders(
            ticker="KRW-BTC",
            current_price=52000.0,
            current_date=sample_date,
            low_price=52000.0,
        )

        assert len(triggered) == 1
        assert triggered[0].is_triggered is True

    def test_check_orders_multiple_tickers(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test checking orders for multiple tickers."""
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        manager.create_stop_loss(
            ticker="KRW-ETH",
            entry_price=2000.0,
            entry_date=sample_date,
            amount=10.0,
            stop_loss_price=1800.0,
        )

        # Check only BTC
        triggered = manager.check_orders(
            ticker="KRW-BTC",
            current_price=44000.0,
            current_date=sample_date,
            low_price=44000.0,
        )

        assert len(triggered) == 1
        assert triggered[0].ticker == "KRW-BTC"

    def test_check_orders_with_default_prices(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test check orders uses current_price when low/high not provided."""
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        # Only provide current_price
        triggered = manager.check_orders(
            ticker="KRW-BTC",
            current_price=44000.0,
            current_date=sample_date,
        )

        assert len(triggered) == 1


# -------------------------------------------------------------------------
# Get Active Orders Tests
# -------------------------------------------------------------------------


class TestGetActiveOrders:
    """Test retrieving active orders."""

    def test_get_active_orders_all(self, manager: AdvancedOrderManager, sample_date: date) -> None:
        """Test getting all active orders."""
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        manager.create_take_profit(
            ticker="KRW-ETH",
            entry_price=2000.0,
            entry_date=sample_date,
            amount=10.0,
            take_profit_price=2200.0,
        )

        active = manager.get_active_orders()

        assert len(active) == 2

    def test_get_active_orders_by_ticker(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test getting active orders filtered by ticker."""
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        manager.create_take_profit(
            ticker="KRW-ETH",
            entry_price=2000.0,
            entry_date=sample_date,
            amount=10.0,
            take_profit_price=2200.0,
        )

        active_btc = manager.get_active_orders(ticker="KRW-BTC")

        assert len(active_btc) == 2
        assert all(o.ticker == "KRW-BTC" for o in active_btc)

    def test_get_active_orders_excludes_triggered(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test that triggered orders are not in active list."""
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        # Trigger the order
        manager.check_orders(
            ticker="KRW-BTC",
            current_price=44000.0,
            current_date=sample_date,
            low_price=44000.0,
        )

        active = manager.get_active_orders()

        assert len(active) == 0


# -------------------------------------------------------------------------
# Cancel Order Tests
# -------------------------------------------------------------------------


class TestCancelOrder:
    """Test canceling orders."""

    def test_cancel_order_success(self, manager: AdvancedOrderManager, sample_date: date) -> None:
        """Test successfully canceling an order."""
        order = manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        result = manager.cancel_order(order.order_id)

        assert result is True
        assert order.is_active is False

    def test_cancel_order_not_found(self, manager: AdvancedOrderManager) -> None:
        """Test canceling non-existent order."""
        result = manager.cancel_order("non_existent_id")

        assert result is False

    def test_cancel_all_orders_all_tickers(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test canceling all orders across all tickers."""
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        manager.create_stop_loss(
            ticker="KRW-ETH",
            entry_price=2000.0,
            entry_date=sample_date,
            amount=10.0,
            stop_loss_price=1800.0,
        )

        count = manager.cancel_all_orders()

        assert count == 2

    def test_cancel_all_orders_by_ticker(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test canceling all orders for a specific ticker."""
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        manager.create_stop_loss(
            ticker="KRW-ETH",
            entry_price=2000.0,
            entry_date=sample_date,
            amount=10.0,
            stop_loss_price=1800.0,
        )

        count = manager.cancel_all_orders(ticker="KRW-BTC")

        assert count == 2
        btc_orders = manager.get_active_orders(ticker="KRW-BTC")
        eth_orders = manager.get_active_orders(ticker="KRW-ETH")
        assert len(btc_orders) == 0
        assert len(eth_orders) == 1

    def test_cancel_all_orders_empty(self, manager: AdvancedOrderManager) -> None:
        """Test canceling all orders when none exist."""
        count = manager.cancel_all_orders()

        assert count == 0


# -------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------


class TestAdvancedOrdersIntegration:
    """Integration tests for advanced order management."""

    def test_complete_stop_loss_workflow(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test complete stop loss workflow."""
        # Create order
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        # Order should be active
        active = manager.get_active_orders()
        assert len(active) == 1

        # Price drops, trigger order
        triggered = manager.check_orders(
            ticker="KRW-BTC",
            current_price=44000.0,
            current_date=sample_date,
            low_price=44000.0,
        )

        assert len(triggered) == 1
        # Order should no longer be active
        active = manager.get_active_orders()
        assert len(active) == 0

    def test_multiple_orders_same_ticker(
        self, manager: AdvancedOrderManager, sample_date: date
    ) -> None:
        """Test managing multiple order types for same ticker."""
        manager.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            stop_loss_price=45000.0,
        )

        manager.create_take_profit(
            ticker="KRW-BTC",
            entry_price=50000.0,
            entry_date=sample_date,
            amount=1.0,
            take_profit_price=55000.0,
        )

        # Both should be active
        active = manager.get_active_orders(ticker="KRW-BTC")
        assert len(active) == 2

        # Trigger stop loss
        manager.check_orders(
            ticker="KRW-BTC",
            current_price=44000.0,
            current_date=sample_date,
            low_price=44000.0,
        )

        # Only take profit should be active now
        active = manager.get_active_orders(ticker="KRW-BTC")
        assert len(active) == 1
        assert active[0].order_type == OrderType.TAKE_PROFIT
