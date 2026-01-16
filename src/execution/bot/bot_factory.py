"""
Bot component factory.

Separates component creation responsibility from bot initialization (SRP).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.loader import get_config
from src.exchange import Exchange, ExchangeFactory
from src.execution.event_bus import get_event_bus
from src.execution.handlers.notification_handler import NotificationHandler
from src.execution.handlers.trade_handler import TradeHandler
from src.execution.order_manager import OrderManager
from src.execution.orders.advanced_orders import AdvancedOrderManager
from src.execution.position_manager import PositionManager
from src.execution.signal_handler import SignalHandler
from src.strategies.volatility_breakout import VanillaVBO
from src.utils.logger import get_logger
from src.utils.telegram import TelegramNotifier, get_notifier

if TYPE_CHECKING:
    from src.execution.event_bus import EventBus

logger = get_logger(__name__)

MIN_DATA_POINTS = 10


class BotComponents:
    """Container for bot components (DTO)."""

    def __init__(
        self,
        exchange: Exchange,
        position_manager: PositionManager,
        order_manager: OrderManager,
        signal_handler: SignalHandler,
        strategy: VanillaVBO,
        advanced_order_manager: AdvancedOrderManager,
        telegram: TelegramNotifier,
        trade_handler: TradeHandler,
        notification_handler: NotificationHandler,
        event_bus: EventBus,
    ) -> None:
        """Initialize bot components."""
        self.exchange = exchange
        self.position_manager = position_manager
        self.order_manager = order_manager
        self.signal_handler = signal_handler
        self.strategy = strategy
        self.advanced_order_manager = advanced_order_manager
        self.telegram = telegram
        self.trade_handler = trade_handler
        self.notification_handler = notification_handler
        self.event_bus = event_bus


class BotComponentFactory:
    """
    Factory for creating bot components.

    Handles all component initialization with proper dependency injection.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Initialize factory.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.trading_config = self.config.get_trading_config()
        self.strategy_config = self.config.get_strategy_config()
        self.bot_config = self.config.get_bot_config()

    def create(
        self,
        exchange: Exchange | None = None,
        position_manager: PositionManager | None = None,
        order_manager: OrderManager | None = None,
        signal_handler: SignalHandler | None = None,
        strategy: VanillaVBO | None = None,
    ) -> BotComponents:
        """
        Create bot components.

        Args:
            exchange: Optional exchange instance for DI
            position_manager: Optional position manager for DI
            order_manager: Optional order manager for DI
            signal_handler: Optional signal handler for DI
            strategy: Optional strategy for DI

        Returns:
            BotComponents instance
        """
        exchange = self._create_exchange(exchange)
        telegram = self._create_telegram()
        strategy = self._create_strategy(strategy)
        event_bus = get_event_bus()

        trade_handler = TradeHandler()
        notification_handler = NotificationHandler(telegram)

        position_manager = position_manager or PositionManager(exchange, publish_events=True)
        order_manager = order_manager or OrderManager(exchange, publish_events=True)
        signal_handler = signal_handler or SignalHandler(
            strategy=strategy,
            exchange=exchange,
            min_data_points=MIN_DATA_POINTS,
            publish_events=True,
        )
        advanced_order_manager = AdvancedOrderManager()

        return BotComponents(
            exchange=exchange,
            position_manager=position_manager,
            order_manager=order_manager,
            signal_handler=signal_handler,
            strategy=strategy,
            advanced_order_manager=advanced_order_manager,
            telegram=telegram,
            trade_handler=trade_handler,
            notification_handler=notification_handler,
            event_bus=event_bus,
        )

    def _create_exchange(self, exchange: Exchange | None) -> Exchange:
        """Create or return exchange instance."""
        if exchange is not None:
            return exchange
        exchange_name = self.config.get_exchange_name()
        return ExchangeFactory.create(exchange_name)

    def _create_telegram(self) -> TelegramNotifier:
        """Create telegram notifier."""
        telegram_config = self.config.get_telegram_config()
        return get_notifier(
            token=telegram_config["token"],
            chat_id=telegram_config["chat_id"],
            enabled=telegram_config["enabled"],
        )

    def _create_strategy(self, strategy: VanillaVBO | None) -> VanillaVBO:
        """Create or return strategy instance."""
        if strategy is not None:
            return strategy
        return VanillaVBO(
            name=self.strategy_config["name"],
            sma_period=self.strategy_config["sma_period"],
            trend_sma_period=self.strategy_config["trend_sma_period"],
            short_noise_period=self.strategy_config["short_noise_period"],
            long_noise_period=self.strategy_config["long_noise_period"],
            exclude_current=self.strategy_config.get("exclude_current", True),
        )

    def get_tickers(self) -> list[str]:
        """Get configured tickers."""
        return list(self.trading_config["tickers"])

    def get_configs(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Get all configuration dictionaries."""
        return self.trading_config, self.strategy_config, self.bot_config
