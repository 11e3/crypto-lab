"""
Integration tests for execution component wiring and facade.

Tests:
- BotComponentFactory DI wiring
- TradingBotFacade integration with real components
- Error recovery and graceful degradation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.execution.bot.bot_facade import TradingBotFacade
from src.execution.bot.bot_factory import BotComponentFactory, BotComponents
from src.execution.bot.bot_init import create_bot_components
from src.execution.event_bus import get_event_bus
from src.execution.handlers.notification_handler import NotificationHandler
from src.execution.handlers.trade_handler import TradeHandler
from src.execution.order_manager import OrderManager
from src.execution.orders.advanced_orders import AdvancedOrderManager
from src.execution.position_manager import PositionManager
from src.execution.signal_handler import SignalHandler
from src.strategies.volatility_breakout import VanillaVBO
from tests.fixtures.mock_exchange import MockExchange


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock config for factory tests."""
    config = MagicMock()
    config.get_trading_config.return_value = {
        "tickers": ["KRW-BTC", "KRW-ETH"],
        "min_order_amount": 5000.0,
        "max_slots": 3,
        "fee_rate": 0.0005,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 5.0,
        "trailing_stop_pct": 1.5,
    }
    config.get_strategy_config.return_value = {
        "name": "VanillaVBO",
        "sma_period": 5,
        "trend_sma_period": 20,
        "short_noise_period": 20,
        "long_noise_period": 10,
        "exclude_current": True,
    }
    config.get_bot_config.return_value = {
        "api_retry_delay": 0.1,
        "websocket_reconnect_delay": 0.1,
        "daily_reset_hour": 9,
        "daily_reset_minute": 0,
    }
    config.get_telegram_config.return_value = {
        "token": "test_token",
        "chat_id": "test_chat_id",
        "enabled": False,
    }
    config.get_exchange_name.return_value = "upbit"
    return config


@pytest.fixture
def facade_with_mock_exchange(mock_config: MagicMock) -> TradingBotFacade:
    """Create a TradingBotFacade with MockExchange via DI."""
    exchange = MockExchange()
    exchange.set_balance("KRW", 10_000_000.0)

    with (
        patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
        patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
    ):
        return TradingBotFacade(exchange=exchange)


class TestComponentWiring:
    """Test BotComponentFactory creates and wires components correctly."""

    def test_factory_creates_all_components(self, mock_config: MagicMock) -> None:
        """Factory should create all 10 components."""
        exchange = MockExchange()
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            factory = BotComponentFactory()
            components = factory.create(exchange=exchange)

        assert isinstance(components, BotComponents)
        assert components.exchange is exchange
        assert isinstance(components.position_manager, PositionManager)
        assert isinstance(components.order_manager, OrderManager)
        assert isinstance(components.signal_handler, SignalHandler)
        assert isinstance(components.strategy, VanillaVBO)
        assert isinstance(components.advanced_order_manager, AdvancedOrderManager)
        assert isinstance(components.trade_handler, TradeHandler)
        assert isinstance(components.notification_handler, NotificationHandler)
        assert components.event_bus is get_event_bus()

    def test_di_exchange_preserved(self, mock_config: MagicMock) -> None:
        """DI-injected exchange should be the exact same instance in components."""
        exchange = MockExchange()
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            factory = BotComponentFactory()
            components = factory.create(exchange=exchange)

        assert components.exchange is exchange
        # OrderManager and PositionManager should use the same exchange
        assert components.order_manager.exchange is exchange
        assert components.position_manager.exchange is exchange

    def test_di_position_manager_preserved(self, mock_config: MagicMock) -> None:
        """DI-injected position_manager should be preserved."""
        exchange = MockExchange()
        pm = PositionManager(exchange, publish_events=False)
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            factory = BotComponentFactory()
            components = factory.create(exchange=exchange, position_manager=pm)

        assert components.position_manager is pm

    def test_di_order_manager_preserved(self, mock_config: MagicMock) -> None:
        """DI-injected order_manager should be preserved."""
        exchange = MockExchange()
        om = OrderManager(exchange, publish_events=False)
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            factory = BotComponentFactory()
            components = factory.create(exchange=exchange, order_manager=om)

        assert components.order_manager is om

    def test_di_strategy_preserved(self, mock_config: MagicMock) -> None:
        """DI-injected strategy should be preserved."""
        exchange = MockExchange()
        strategy = VanillaVBO(
            sma_period=3, trend_sma_period=10, short_noise_period=5, long_noise_period=5
        )
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            factory = BotComponentFactory()
            components = factory.create(exchange=exchange, strategy=strategy)

        assert components.strategy is strategy

    def test_config_propagated_to_tickers(self, mock_config: MagicMock) -> None:
        """Config tickers should be accessible from factory."""
        exchange = MockExchange()
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            factory = BotComponentFactory()
            factory.create(exchange=exchange)

        assert factory.get_tickers() == ["KRW-BTC", "KRW-ETH"]

    def test_create_bot_components_wrapper(self, mock_config: MagicMock) -> None:
        """create_bot_components should return correct tuple structure."""
        exchange = MockExchange()
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            components, trading_cfg, strategy_cfg, bot_cfg, tickers = create_bot_components(
                exchange=exchange
            )

        assert isinstance(components, BotComponents)
        assert trading_cfg["max_slots"] == 3
        assert strategy_cfg["sma_period"] == 5
        assert bot_cfg["daily_reset_hour"] == 9
        assert tickers == ["KRW-BTC", "KRW-ETH"]

    def test_event_bus_shared_across_components(self, mock_config: MagicMock) -> None:
        """All event-aware components should share the same EventBus."""
        exchange = MockExchange()
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            factory = BotComponentFactory()
            components = factory.create(exchange=exchange)

        bus = get_event_bus()
        assert components.event_bus is bus
        assert components.trade_handler.event_bus is bus
        assert components.notification_handler.event_bus is bus

    def test_handlers_auto_subscribe_to_event_bus(self, mock_config: MagicMock) -> None:
        """TradeHandler and NotificationHandler should auto-subscribe during init."""
        exchange = MockExchange()
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            factory = BotComponentFactory()
            factory.create(exchange=exchange)

        from src.execution.events import EventType

        bus = get_event_bus()
        # TradeHandler subscribes to 5 event types
        assert bus.get_subscriber_count(EventType.ORDER_PLACED) >= 1
        assert bus.get_subscriber_count(EventType.ORDER_FILLED) >= 1
        assert bus.get_subscriber_count(EventType.POSITION_OPENED) >= 1
        assert bus.get_subscriber_count(EventType.POSITION_CLOSED) >= 1
        # NotificationHandler subscribes to 7 event types
        assert bus.get_subscriber_count(EventType.ENTRY_SIGNAL) >= 1
        assert bus.get_subscriber_count(EventType.EXIT_SIGNAL) >= 1
        assert bus.get_subscriber_count(EventType.DAILY_RESET) >= 1


class TestFacadeIntegration:
    """Test TradingBotFacade with real components wired via factory."""

    def test_facade_has_correct_tickers(self, facade_with_mock_exchange: TradingBotFacade) -> None:
        """Facade should expose configured tickers."""
        assert facade_with_mock_exchange.tickers == ["KRW-BTC", "KRW-ETH"]

    def test_facade_exchange_is_mock(self, facade_with_mock_exchange: TradingBotFacade) -> None:
        """Facade should use the injected MockExchange."""
        assert isinstance(facade_with_mock_exchange.exchange, MockExchange)

    def test_get_krw_balance_success(self, facade_with_mock_exchange: TradingBotFacade) -> None:
        """get_krw_balance should return correct balance from MockExchange."""
        balance = facade_with_mock_exchange.get_krw_balance()
        assert balance == 10_000_000.0

    def test_get_krw_balance_after_change(
        self, facade_with_mock_exchange: TradingBotFacade
    ) -> None:
        """get_krw_balance should reflect updated balance."""
        facade_with_mock_exchange.exchange.set_balance("KRW", 5_000_000.0)  # type: ignore[attr-defined]
        assert facade_with_mock_exchange.get_krw_balance() == 5_000_000.0

    def test_get_krw_balance_exchange_error(self, mock_config: MagicMock) -> None:
        """get_krw_balance should return 0.0 when exchange raises error."""
        from src.exchange import ExchangeError

        exchange = MagicMock()
        exchange.get_balance.side_effect = ExchangeError("Connection failed")

        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier", return_value=MagicMock()),
        ):
            facade = TradingBotFacade(exchange=exchange)

        assert facade.get_krw_balance() == 0.0

    def test_calculate_buy_amount_normal(self, facade_with_mock_exchange: TradingBotFacade) -> None:
        """_calculate_buy_amount should divide balance by available slots."""
        bot = facade_with_mock_exchange
        # 10M KRW, 3 max slots, 0 positions -> 10M/3 * (1-0.0005)
        amount = bot._calculate_buy_amount()
        expected = (10_000_000.0 / 3) * (1 - 0.0005)
        assert abs(amount - expected) < 1.0

    def test_calculate_buy_amount_with_existing_position(
        self, facade_with_mock_exchange: TradingBotFacade
    ) -> None:
        """_calculate_buy_amount should account for existing positions."""
        bot = facade_with_mock_exchange
        bot.position_manager.add_position("KRW-BTC", 50_000_000.0, 0.1)
        # 1 position, 2 remaining slots
        amount = bot._calculate_buy_amount()
        expected = (10_000_000.0 / 2) * (1 - 0.0005)
        assert abs(amount - expected) < 1.0

    def test_calculate_buy_amount_insufficient_balance(
        self, facade_with_mock_exchange: TradingBotFacade
    ) -> None:
        """_calculate_buy_amount should return 0.0 when balance is below min."""
        bot = facade_with_mock_exchange
        bot.exchange.set_balance("KRW", 1000.0)  # type: ignore[attr-defined]  # Below min_order_amount
        assert bot._calculate_buy_amount() == 0.0

    def test_calculate_buy_amount_no_slots(
        self, facade_with_mock_exchange: TradingBotFacade
    ) -> None:
        """_calculate_buy_amount should return 0.0 when no slots available."""
        bot = facade_with_mock_exchange
        bot.position_manager.add_position("KRW-BTC", 50_000_000.0, 0.1)
        bot.position_manager.add_position("KRW-ETH", 3_000_000.0, 1.0)
        bot.position_manager.add_position("KRW-XRP", 500.0, 100.0)
        # 3 positions = max_slots, no room
        assert bot._calculate_buy_amount() == 0.0

    def test_initialize_targets_calls_signal_handler(
        self, facade_with_mock_exchange: TradingBotFacade
    ) -> None:
        """initialize_targets should call signal_handler.calculate_metrics for each ticker."""
        bot = facade_with_mock_exchange
        mock_metrics = {
            "target": 50_000_000.0,
            "k": 0.5,
            "sma": 48_000_000.0,
            "sma_trend": 49_000_000.0,
            "long_noise": 0.6,
        }
        bot.signal_handler.metrics_calculator = MagicMock()
        bot.signal_handler.metrics_calculator.calculate.return_value = mock_metrics

        bot.initialize_targets()

        assert len(bot.target_info) == 2
        assert bot.target_info["KRW-BTC"] == mock_metrics


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    def test_facade_survives_exchange_error_on_balance(
        self, facade_with_mock_exchange: TradingBotFacade
    ) -> None:
        """Facade should not crash when exchange fails on balance check."""
        from src.exchange import ExchangeError

        bot = facade_with_mock_exchange
        original_get_balance = bot.exchange.get_balance

        def failing_get_balance(currency: str) -> None:
            raise ExchangeError("Network error")

        bot.exchange.get_balance = failing_get_balance  # type: ignore[method-assign]
        assert bot.get_krw_balance() == 0.0
        # Restore
        bot.exchange.get_balance = original_get_balance  # type: ignore[method-assign]
        assert bot.get_krw_balance() == 10_000_000.0

    def test_all_slots_full_prevents_buy(self, facade_with_mock_exchange: TradingBotFacade) -> None:
        """No buy should happen when all slots are full."""
        bot = facade_with_mock_exchange
        for i, ticker in enumerate(["KRW-BTC", "KRW-ETH", "KRW-XRP"]):
            bot.position_manager.add_position(ticker, 50_000_000.0, 0.001 * (i + 1))
        assert bot._calculate_buy_amount() == 0.0

    def test_duplicate_position_raises(self, facade_with_mock_exchange: TradingBotFacade) -> None:
        """Adding duplicate position should raise ValueError."""
        bot = facade_with_mock_exchange
        bot.position_manager.add_position("KRW-BTC", 50_000_000.0, 0.1)
        with pytest.raises(ValueError, match="Position already exists"):
            bot.position_manager.add_position("KRW-BTC", 51_000_000.0, 0.2)

    def test_facade_components_not_none(self, facade_with_mock_exchange: TradingBotFacade) -> None:
        """All facade components should be initialized (not None)."""
        bot = facade_with_mock_exchange
        assert bot.exchange is not None
        assert bot.position_manager is not None
        assert bot.order_manager is not None
        assert bot.signal_handler is not None
        assert bot.strategy is not None
        assert bot.advanced_order_manager is not None
        assert bot.telegram is not None
        assert bot.trade_handler is not None
        assert bot.notification_handler is not None
        assert bot.event_bus is not None
