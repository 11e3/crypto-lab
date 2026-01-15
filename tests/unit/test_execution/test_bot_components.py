"""
Tests for Bot Components Module.
"""

from unittest.mock import MagicMock

from src.execution.bot.bot_init import BotComponents


class TestBotComponents:
    """Tests for BotComponents container."""

    def test_initialization(self) -> None:
        """Test BotComponents initialization."""
        mock_exchange = MagicMock()
        mock_position_manager = MagicMock()
        mock_order_manager = MagicMock()
        mock_signal_handler = MagicMock()
        mock_strategy = MagicMock()
        mock_advanced_order_manager = MagicMock()
        mock_telegram = MagicMock()
        mock_trade_handler = MagicMock()
        mock_notification_handler = MagicMock()
        mock_event_bus = MagicMock()

        components = BotComponents(
            exchange=mock_exchange,
            position_manager=mock_position_manager,
            order_manager=mock_order_manager,
            signal_handler=mock_signal_handler,
            strategy=mock_strategy,
            advanced_order_manager=mock_advanced_order_manager,
            telegram=mock_telegram,
            trade_handler=mock_trade_handler,
            notification_handler=mock_notification_handler,
            event_bus=mock_event_bus,
        )

        assert components.exchange == mock_exchange
        assert components.position_manager == mock_position_manager
        assert components.order_manager == mock_order_manager
        assert components.signal_handler == mock_signal_handler
        assert components.strategy == mock_strategy
        assert components.advanced_order_manager == mock_advanced_order_manager
        assert components.telegram == mock_telegram
        assert components.trade_handler == mock_trade_handler
        assert components.notification_handler == mock_notification_handler
        assert components.event_bus == mock_event_bus

    def test_components_accessible(self) -> None:
        """Test that all components are properly stored and accessible."""
        mock_exchange = MagicMock()
        mock_position_manager = MagicMock()
        mock_order_manager = MagicMock()
        mock_signal_handler = MagicMock()
        mock_strategy = MagicMock()
        mock_advanced_order_manager = MagicMock()
        mock_telegram = MagicMock()
        mock_trade_handler = MagicMock()
        mock_notification_handler = MagicMock()
        mock_event_bus = MagicMock()

        components = BotComponents(
            exchange=mock_exchange,
            position_manager=mock_position_manager,
            order_manager=mock_order_manager,
            signal_handler=mock_signal_handler,
            strategy=mock_strategy,
            advanced_order_manager=mock_advanced_order_manager,
            telegram=mock_telegram,
            trade_handler=mock_trade_handler,
            notification_handler=mock_notification_handler,
            event_bus=mock_event_bus,
        )

        # Verify all components are accessible
        assert hasattr(components, "exchange")
        assert hasattr(components, "position_manager")
        assert hasattr(components, "order_manager")
        assert hasattr(components, "signal_handler")
        assert hasattr(components, "strategy")
        assert hasattr(components, "advanced_order_manager")
        assert hasattr(components, "telegram")
        assert hasattr(components, "trade_handler")
        assert hasattr(components, "notification_handler")
        assert hasattr(components, "event_bus")
