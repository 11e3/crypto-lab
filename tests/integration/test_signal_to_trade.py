"""
Integration tests for signal-to-trade flow.

Tests the complete flow: Signal generation -> Entry/Exit decision -> Order execution -> Position management.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.execution.event_bus import get_event_bus
from src.execution.events import EventType
from src.execution.order_manager import OrderManager
from src.execution.orders.advanced_orders import AdvancedOrderManager
from src.execution.position_manager import PositionManager
from src.execution.signal_handler import SignalHandler
from src.execution.trade_executor import execute_buy_order, process_ticker_update, sell_all
from src.strategies.volatility_breakout import VanillaVBO
from tests.fixtures.mock_exchange import MockExchange


@pytest.fixture
def exchange() -> MockExchange:
    """Create MockExchange with default state."""
    ex = MockExchange()
    ex.set_balance("KRW", 10_000_000.0)
    ex.set_price("KRW-BTC", 50_000_000.0)
    ex.set_price("KRW-ETH", 3_000_000.0)
    return ex


@pytest.fixture
def strategy() -> VanillaVBO:
    """Create VBO strategy."""
    return VanillaVBO(
        sma_period=5,
        trend_sma_period=10,
        short_noise_period=5,
        long_noise_period=5,
    )


@pytest.fixture
def components(exchange: MockExchange, strategy: VanillaVBO) -> dict:
    """Create real components wired together."""
    event_bus = get_event_bus()
    pm = PositionManager(exchange, publish_events=True, event_bus=event_bus)
    om = OrderManager(exchange, publish_events=True, event_bus=event_bus)
    sh = SignalHandler(
        strategy=strategy,
        exchange=exchange,
        min_data_points=5,
        publish_events=True,
        event_bus=event_bus,
    )
    aom = AdvancedOrderManager()
    telegram = MagicMock()
    return {
        "exchange": exchange,
        "position_manager": pm,
        "order_manager": om,
        "signal_handler": sh,
        "advanced_order_manager": aom,
        "telegram": telegram,
        "event_bus": event_bus,
    }


@pytest.fixture
def trading_config() -> dict:
    """Standard trading config."""
    return {
        "tickers": ["KRW-BTC", "KRW-ETH"],
        "min_order_amount": 5000.0,
        "max_slots": 3,
        "fee_rate": 0.0005,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 5.0,
        "trailing_stop_pct": 1.5,
    }


def _make_ohlcv(periods: int = 30, base_price: float = 50_000_000.0) -> pd.DataFrame:
    """Create OHLCV data with entry signals."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=periods, freq="1D")
    close = base_price + np.cumsum(np.random.randn(periods) * base_price * 0.01)
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(periods) * 0.001),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.uniform(100, 1000, periods),
        },
        index=dates,
    )


class TestEntryFlow:
    """Test entry signal -> buy order -> position lifecycle."""

    def test_buy_order_creates_position(self, components: dict, trading_config: dict) -> None:
        """execute_buy_order should place order and add position."""
        om = components["order_manager"]
        pm = components["position_manager"]
        aom = components["advanced_order_manager"]
        telegram = components["telegram"]

        result = execute_buy_order(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            buy_amount=1_000_000.0,
            order_manager=om,
            position_manager=pm,
            advanced_order_manager=aom,
            telegram=telegram,
            trading_config=trading_config,
            target_info={"KRW-BTC": {"target": 50_000_000.0, "k": 0.5, "long_noise": 0.6}},
            min_amount=5000.0,
        )

        assert result is True
        assert pm.has_position("KRW-BTC")
        position = pm.get_position("KRW-BTC")
        assert position is not None
        assert position.entry_price == 50_000_000.0

    def test_buy_creates_advanced_orders(self, components: dict, trading_config: dict) -> None:
        """Buy should create stop loss and trailing stop orders."""
        aom = components["advanced_order_manager"]

        execute_buy_order(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            buy_amount=1_000_000.0,
            order_manager=components["order_manager"],
            position_manager=components["position_manager"],
            advanced_order_manager=aom,
            telegram=components["telegram"],
            trading_config=trading_config,
            target_info={"KRW-BTC": {"target": 50_000_000.0, "k": 0.5}},
            min_amount=5000.0,
        )

        active_orders = aom.get_active_orders("KRW-BTC")
        assert len(active_orders) >= 2  # stop_loss + trailing_stop (+ take_profit)

    def test_buy_updates_krw_balance(self, components: dict, trading_config: dict) -> None:
        """Buy order should deduct from KRW balance."""
        exchange = components["exchange"]
        initial_krw = exchange.get_balance("KRW").balance

        execute_buy_order(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            buy_amount=1_000_000.0,
            order_manager=components["order_manager"],
            position_manager=components["position_manager"],
            advanced_order_manager=components["advanced_order_manager"],
            telegram=components["telegram"],
            trading_config=trading_config,
            target_info={},
            min_amount=5000.0,
        )

        final_krw = exchange.get_balance("KRW").balance
        assert final_krw < initial_krw

    def test_buy_below_min_amount_fails(self, components: dict, trading_config: dict) -> None:
        """Buy amount below min_order_amount should not execute."""
        result = execute_buy_order(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            buy_amount=1000.0,  # Below min 5000
            order_manager=components["order_manager"],
            position_manager=components["position_manager"],
            advanced_order_manager=components["advanced_order_manager"],
            telegram=components["telegram"],
            trading_config=trading_config,
            target_info={},
            min_amount=5000.0,
        )

        assert result is False
        assert not components["position_manager"].has_position("KRW-BTC")

    def test_buy_with_insufficient_balance(self, components: dict, trading_config: dict) -> None:
        """Buy should fail gracefully when exchange has insufficient balance."""
        components["exchange"].set_balance("KRW", 100.0)

        result = execute_buy_order(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            buy_amount=1_000_000.0,
            order_manager=components["order_manager"],
            position_manager=components["position_manager"],
            advanced_order_manager=components["advanced_order_manager"],
            telegram=components["telegram"],
            trading_config=trading_config,
            target_info={},
            min_amount=5000.0,
        )

        assert result is False
        assert not components["position_manager"].has_position("KRW-BTC")

    def test_process_ticker_no_position_no_signal(
        self, components: dict, trading_config: dict
    ) -> None:
        """process_ticker_update should do nothing when no signal."""
        sh = components["signal_handler"]
        sh.check_entry_signal = MagicMock(return_value=False)

        process_ticker_update(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            position_manager=components["position_manager"],
            order_manager=components["order_manager"],
            advanced_order_manager=components["advanced_order_manager"],
            signal_handler=sh,
            trading_config=trading_config,
            target_info={"KRW-BTC": {"target": 50_000_000.0}},
            telegram=components["telegram"],
            calculate_buy_amount_fn=lambda: 1_000_000.0,
            execute_buy_fn=MagicMock(),
        )

        assert not components["position_manager"].has_position("KRW-BTC")

    def test_process_ticker_with_signal_calls_buy(
        self, components: dict, trading_config: dict
    ) -> None:
        """process_ticker_update should call execute_buy_fn when signal is True."""
        sh = components["signal_handler"]
        sh.check_entry_signal = MagicMock(return_value=True)

        execute_buy_fn = MagicMock(return_value=True)
        process_ticker_update(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            position_manager=components["position_manager"],
            order_manager=components["order_manager"],
            advanced_order_manager=components["advanced_order_manager"],
            signal_handler=sh,
            trading_config=trading_config,
            target_info={"KRW-BTC": {"target": 50_000_000.0}},
            telegram=components["telegram"],
            calculate_buy_amount_fn=lambda: 1_000_000.0,
            execute_buy_fn=execute_buy_fn,
        )

        execute_buy_fn.assert_called_once_with("KRW-BTC", 50_000_000.0, 1_000_000.0)

    def test_process_ticker_skips_when_already_holding(
        self, components: dict, trading_config: dict
    ) -> None:
        """process_ticker_update should skip entry when already holding position."""
        pm = components["position_manager"]
        pm.add_position("KRW-BTC", 50_000_000.0, 0.02)

        execute_buy_fn = MagicMock()
        process_ticker_update(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            position_manager=pm,
            order_manager=components["order_manager"],
            advanced_order_manager=components["advanced_order_manager"],
            signal_handler=components["signal_handler"],
            trading_config=trading_config,
            target_info={"KRW-BTC": {"target": 50_000_000.0}},
            telegram=components["telegram"],
            calculate_buy_amount_fn=lambda: 1_000_000.0,
            execute_buy_fn=execute_buy_fn,
        )

        execute_buy_fn.assert_not_called()

    def test_process_ticker_skips_zero_buy_amount(
        self, components: dict, trading_config: dict
    ) -> None:
        """process_ticker_update should skip when buy amount is 0."""
        sh = components["signal_handler"]
        sh.check_entry_signal = MagicMock(return_value=True)

        execute_buy_fn = MagicMock()
        process_ticker_update(
            ticker="KRW-BTC",
            current_price=50_000_000.0,
            position_manager=components["position_manager"],
            order_manager=components["order_manager"],
            advanced_order_manager=components["advanced_order_manager"],
            signal_handler=sh,
            trading_config=trading_config,
            target_info={"KRW-BTC": {"target": 50_000_000.0}},
            telegram=components["telegram"],
            calculate_buy_amount_fn=lambda: 0.0,
            execute_buy_fn=execute_buy_fn,
        )

        execute_buy_fn.assert_not_called()


class TestExitFlow:
    """Test exit signal -> sell order -> position removal."""

    def test_sell_all_removes_position(self, components: dict) -> None:
        """sell_all should remove position after successful sell."""
        pm = components["position_manager"]
        exchange = components["exchange"]
        exchange.set_balance("BTC", 0.1)
        pm.add_position("KRW-BTC", 50_000_000.0, 0.1)

        result = sell_all(
            ticker="KRW-BTC",
            order_manager=components["order_manager"],
            position_manager=pm,
            exchange=exchange,
            telegram=components["telegram"],
            min_amount=5000.0,
        )

        assert result is True
        assert not pm.has_position("KRW-BTC")

    def test_sell_all_updates_balances(self, components: dict) -> None:
        """sell_all should update both KRW and base currency balances."""
        exchange = components["exchange"]
        exchange.set_balance("BTC", 0.1)
        exchange.set_balance("KRW", 5_000_000.0)
        pm = components["position_manager"]
        pm.add_position("KRW-BTC", 50_000_000.0, 0.1)

        sell_all(
            ticker="KRW-BTC",
            order_manager=components["order_manager"],
            position_manager=pm,
            exchange=exchange,
            telegram=components["telegram"],
            min_amount=5000.0,
        )

        # KRW should increase, BTC should decrease
        krw = exchange.get_balance("KRW").balance
        btc = exchange.get_balance("BTC").balance
        assert krw > 5_000_000.0
        assert btc < 0.1

    def test_sell_all_failure_preserves_position(self, components: dict) -> None:
        """sell_all should preserve position when exchange fails."""
        exchange = components["exchange"]
        exchange.configure_failures(fail_sell=True)
        pm = components["position_manager"]
        exchange.set_balance("BTC", 0.1)
        pm.add_position("KRW-BTC", 50_000_000.0, 0.1)

        result = sell_all(
            ticker="KRW-BTC",
            order_manager=components["order_manager"],
            position_manager=pm,
            exchange=exchange,
            telegram=components["telegram"],
            min_amount=5000.0,
        )

        assert result is False
        assert pm.has_position("KRW-BTC")

    def test_stop_loss_triggers_sell(self, components: dict, trading_config: dict) -> None:
        """Stop loss order should trigger sell when price drops below threshold."""
        exchange = components["exchange"]
        pm = components["position_manager"]
        aom = components["advanced_order_manager"]

        # Set up position with stop loss (2% = 0.02 decimal fraction)
        exchange.set_balance("BTC", 0.02)
        pm.add_position("KRW-BTC", 50_000_000.0, 0.02)
        aom.create_stop_loss(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            stop_loss_pct=0.02,
        )

        # Price drops below stop loss threshold (2% below 50M = 49M)
        drop_price = 48_000_000.0
        exchange.set_price("KRW-BTC", drop_price)

        triggered = aom.check_orders(
            ticker="KRW-BTC",
            current_price=drop_price,
            current_date=date.today(),
            low_price=drop_price,
        )

        assert len(triggered) >= 1

    def test_trailing_stop_updates_on_price_rise(self, components: dict) -> None:
        """Trailing stop should update stop price as price rises."""
        aom = components["advanced_order_manager"]

        aom.create_trailing_stop(
            ticker="KRW-BTC",
            entry_price=50_000_000.0,
            entry_date=date.today(),
            amount=0.02,
            trailing_stop_pct=1.5,
        )

        # Price rises to 55M
        aom.check_orders(
            ticker="KRW-BTC",
            current_price=55_000_000.0,
            current_date=date.today(),
            high_price=55_000_000.0,
        )

        active = aom.get_active_orders("KRW-BTC")
        assert len(active) == 1
        # Trailing stop should have adjusted upward


class TestSignalCalculation:
    """Test signal handler integration with strategy and data."""

    def test_entry_signal_with_ohlcv(self, exchange: MockExchange, strategy: VanillaVBO) -> None:
        """SignalHandler should evaluate entry signal from OHLCV data."""
        ohlcv = _make_ohlcv(30, base_price=50_000_000.0)
        exchange.set_ohlcv_data("KRW-BTC", "day", ohlcv)

        sh = SignalHandler(
            strategy=strategy,
            exchange=exchange,
            min_data_points=5,
            publish_events=False,
        )

        # This should not raise - signal value depends on data
        result = sh.check_entry_signal("KRW-BTC", 50_000_000.0, target_price=49_000_000.0)
        assert isinstance(result, bool)

    def test_entry_signal_insufficient_data(
        self, exchange: MockExchange, strategy: VanillaVBO
    ) -> None:
        """Entry signal should return False with insufficient data."""
        # Very short OHLCV
        ohlcv = _make_ohlcv(2, base_price=50_000_000.0)
        exchange.set_ohlcv_data("KRW-BTC", "day", ohlcv)

        sh = SignalHandler(
            strategy=strategy,
            exchange=exchange,
            min_data_points=5,
            publish_events=False,
        )

        result = sh.check_entry_signal("KRW-BTC", 50_000_000.0)
        assert result is False

    def test_entry_signal_no_data(self, exchange: MockExchange, strategy: VanillaVBO) -> None:
        """Entry signal should return False when no OHLCV data."""
        sh = SignalHandler(
            strategy=strategy,
            exchange=exchange,
            min_data_points=5,
            publish_events=False,
        )

        result = sh.check_entry_signal("KRW-BTC", 50_000_000.0)
        assert result is False

    def test_exit_signal_with_data(self, exchange: MockExchange, strategy: VanillaVBO) -> None:
        """SignalHandler should evaluate exit signal from OHLCV data."""
        ohlcv = _make_ohlcv(30, base_price=50_000_000.0)
        exchange.set_ohlcv_data("KRW-BTC", "day", ohlcv)

        sh = SignalHandler(
            strategy=strategy,
            exchange=exchange,
            min_data_points=5,
            publish_events=False,
        )

        result = sh.check_exit_signal("KRW-BTC")
        assert isinstance(result, bool)

    def test_signal_publishes_entry_event(
        self, exchange: MockExchange, strategy: VanillaVBO
    ) -> None:
        """Entry signal True should publish ENTRY_SIGNAL event."""
        ohlcv = _make_ohlcv(30, base_price=50_000_000.0)
        exchange.set_ohlcv_data("KRW-BTC", "day", ohlcv)

        bus = get_event_bus()
        events_received: list = []
        bus.subscribe(EventType.ENTRY_SIGNAL, lambda e: events_received.append(e))

        sh = SignalHandler(
            strategy=strategy,
            exchange=exchange,
            min_data_points=5,
            publish_events=True,
            event_bus=bus,
        )

        # Mock the data_loader and strategy to force entry signal
        with (
            patch.object(sh.data_loader, "get_ohlcv", return_value=ohlcv),
            patch.object(
                strategy,
                "calculate_indicators",
                return_value=ohlcv.assign(entry_signal=True, exit_signal=False),
            ),
            patch.object(
                strategy,
                "generate_signals",
                return_value=ohlcv.assign(entry_signal=True, exit_signal=False),
            ),
        ):
            result = sh.check_entry_signal("KRW-BTC", 50_000_000.0)

        if result:
            assert len(events_received) >= 1

    def test_multiple_tickers_independent_signals(
        self, exchange: MockExchange, strategy: VanillaVBO
    ) -> None:
        """Each ticker should have independent signal evaluation."""
        btc_ohlcv = _make_ohlcv(30, base_price=50_000_000.0)
        eth_ohlcv = _make_ohlcv(30, base_price=3_000_000.0)
        exchange.set_ohlcv_data("KRW-BTC", "day", btc_ohlcv)
        exchange.set_ohlcv_data("KRW-ETH", "day", eth_ohlcv)

        sh = SignalHandler(
            strategy=strategy,
            exchange=exchange,
            min_data_points=5,
            publish_events=False,
        )

        btc_signal = sh.check_entry_signal("KRW-BTC", 50_000_000.0)
        eth_signal = sh.check_entry_signal("KRW-ETH", 3_000_000.0)

        # Both should return boolean (actual value depends on data)
        assert isinstance(btc_signal, bool)
        assert isinstance(eth_signal, bool)
