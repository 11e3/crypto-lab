"""
Signal handler for processing trading signals from strategies.

Focuses on signal detection only (SRP). Data loading and metrics
calculation are delegated to specialized classes.
"""

import pandas as pd

from src.exchange import MarketDataService
from src.execution.event_bus import EventBus, get_event_bus
from src.execution.events import EventType, SignalEvent
from src.execution.signal_data import SignalDataLoader
from src.execution.signal_metrics import SignalMetricsCalculator
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalHandler:
    """
    Handles trading signals from strategies.

    Focuses on signal detection and event publishing.
    Delegates data loading and metrics calculation to specialized classes.
    """

    def __init__(
        self,
        strategy: Strategy,
        exchange: MarketDataService,
        min_data_points: int = 10,
        publish_events: bool = True,
        event_bus: EventBus | None = None,
        data_loader: SignalDataLoader | None = None,
        metrics_calculator: SignalMetricsCalculator | None = None,
    ) -> None:
        """
        Initialize signal handler.

        Args:
            strategy: Trading strategy instance
            exchange: Service implementing MarketDataService protocol
            min_data_points: Minimum data points required for signal generation
            publish_events: Whether to publish events (default: True)
            event_bus: Optional EventBus instance (uses global if not provided)
            data_loader: Optional SignalDataLoader (creates default if not provided)
            metrics_calculator: Optional SignalMetricsCalculator (creates default if not provided)
        """
        self.strategy = strategy
        self.exchange = exchange
        self.min_data_points = min_data_points
        self.publish_events = publish_events
        self.event_bus = event_bus if event_bus else (get_event_bus() if publish_events else None)

        self.data_loader = data_loader or SignalDataLoader(exchange, min_data_points)
        self.metrics_calculator = metrics_calculator or SignalMetricsCalculator(
            strategy, self.data_loader
        )

    def get_ohlcv_data(
        self,
        ticker: str,
        interval: str = "day",
        count: int | None = None,
    ) -> pd.DataFrame | None:
        """Get OHLCV data for signal generation. Delegates to SignalDataLoader."""
        return self.data_loader.get_ohlcv(ticker, interval=interval, count=count)

    def check_entry_signal(
        self,
        ticker: str,
        current_price: float,
        target_price: float | None = None,
    ) -> bool:
        """
        Check if entry signal is present.

        Args:
            ticker: Trading pair symbol
            current_price: Current market price
            target_price: Target price for breakout (optional)

        Returns:
            True if entry conditions are met
        """
        try:
            df = self.data_loader.get_ohlcv(ticker)
            if df is None:
                return False

            df = self.strategy.calculate_indicators(df)
            df = self.strategy.generate_signals(df)

            if len(df) < 2:
                return False

            yesterday_signal = df.iloc[-2]["entry_signal"]
            entry_signal = bool(yesterday_signal)

            if target_price is not None:
                entry_signal = entry_signal and current_price >= target_price

            if entry_signal and self.event_bus:
                event = SignalEvent(
                    event_type=EventType.ENTRY_SIGNAL,
                    source="SignalHandler",
                    ticker=ticker,
                    signal_type="entry",
                    price=current_price,
                    target_price=target_price,
                )
                self.event_bus.publish(event)

            return entry_signal
        except Exception as e:
            logger.error(f"Error checking entry signal for {ticker}: {e}", exc_info=True)
            return False

    def check_exit_signal(self, ticker: str) -> bool:
        """
        Check if exit signal is present.

        Args:
            ticker: Trading pair symbol

        Returns:
            True if exit conditions are met
        """
        try:
            df = self.data_loader.get_ohlcv(ticker)
            if df is None:
                return False

            df = self.strategy.calculate_indicators(df)
            df = self.strategy.generate_signals(df)

            if len(df) < 2:
                return False

            yesterday_signal = df.iloc[-2]["exit_signal"]
            exit_signal = bool(yesterday_signal)

            if exit_signal and self.event_bus:
                try:
                    current_price = self.exchange.get_current_price(ticker)
                    event = SignalEvent(
                        event_type=EventType.EXIT_SIGNAL,
                        source="SignalHandler",
                        ticker=ticker,
                        signal_type="exit",
                        price=current_price,
                    )
                    self.event_bus.publish(event)
                except Exception as e:
                    logger.error(f"Error getting price for exit signal event: {e}", exc_info=True)

            return exit_signal
        except Exception as e:
            logger.error(f"Error checking exit signal for {ticker}: {e}", exc_info=True)
            return False

    def calculate_metrics(
        self,
        ticker: str,
        required_period: int | None = None,
    ) -> dict[str, float] | None:
        """Calculate strategy metrics. Delegates to SignalMetricsCalculator."""
        return self.metrics_calculator.calculate(ticker, required_period)
