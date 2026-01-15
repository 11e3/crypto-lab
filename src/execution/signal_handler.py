"""
Signal handler for processing trading signals from strategies.
"""

from typing import TYPE_CHECKING

import pandas as pd

from src.execution.event_bus import EventBus, get_event_bus
from src.execution.events import EventType, SignalEvent
from src.strategies.base import Strategy
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.exchange import Exchange

logger = get_logger(__name__)


class SignalHandler:
    """
    Handles trading signals from strategies.

    Processes entry/exit signals and validates conditions.
    """

    def __init__(
        self,
        strategy: Strategy,
        exchange: "Exchange",
        min_data_points: int = 10,
        publish_events: bool = True,
        event_bus: EventBus | None = None,
    ) -> None:
        """
        Initialize signal handler.

        Args:
            strategy: Trading strategy instance
            exchange: Exchange instance for data fetching
            min_data_points: Minimum data points required for signal generation
            publish_events: Whether to publish events (default: True)
            event_bus: Optional EventBus instance (uses global if not provided)
        """
        self.strategy = strategy
        self.exchange = exchange
        self.min_data_points = min_data_points
        self.publish_events = publish_events
        self.event_bus = event_bus if event_bus else (get_event_bus() if publish_events else None)

    def get_ohlcv_data(
        self,
        ticker: str,
        interval: str = "day",
        count: int | None = None,
    ) -> pd.DataFrame | None:
        """
        Get OHLCV data for signal generation.

        Args:
            ticker: Trading pair symbol
            interval: Data interval (default: "day")
            count: Number of candles to fetch (uses min_data_points if None)

        Returns:
            DataFrame with OHLCV data or None on error
        """
        if count is None:
            count = max(self.min_data_points * 2, 20)

        try:
            df = self.exchange.get_ohlcv(ticker, interval=interval, count=count)
            if df is None or len(df) < self.min_data_points:
                logger.warning(
                    f"Insufficient data for {ticker}: "
                    f"got {len(df) if df is not None else 0} rows, need {self.min_data_points}"
                )
                return None
            return df
        except Exception as e:
            logger.error(f"Error getting OHLCV for {ticker}: {e}", exc_info=True)
            return None

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
            target_price: Target price for breakout (optional, checked if provided)

        Returns:
            True if entry conditions are met
        """
        try:
            df = self.get_ohlcv_data(ticker)
            if df is None:
                return False

            # Calculate indicators and signals
            df = self.strategy.calculate_indicators(df)
            df = self.strategy.generate_signals(df)

            # Check yesterday's confirmed signal (excluding today's incomplete candle)
            if len(df) < 2:
                return False

            yesterday_signal = df.iloc[-2]["entry_signal"]

            # If target_price provided, check if price has reached target (breakout)
            entry_signal = bool(yesterday_signal)
            if target_price is not None:
                entry_signal = entry_signal and current_price >= target_price

            # Publish event if signal is present
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
            df = self.get_ohlcv_data(ticker)
            if df is None:
                return False

            # Calculate indicators and signals
            df = self.strategy.calculate_indicators(df)
            df = self.strategy.generate_signals(df)

            # Check yesterday's confirmed signal (excluding today's incomplete candle)
            if len(df) < 2:
                return False

            yesterday_signal = df.iloc[-2]["exit_signal"]
            exit_signal = bool(yesterday_signal)

            # Publish event if signal is present
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
        """
        Calculate strategy metrics for a ticker.

        Args:
            ticker: Trading pair symbol
            required_period: Minimum period required (uses strategy default if None)

        Returns:
            Dictionary with metrics (target, k, long_noise, sma, sma_trend) or None
        """
        try:
            if required_period is None:
                # Use a reasonable default based on typical strategy needs
                required_period = 20

            count = max(required_period + 5, self.min_data_points * 2)
            df = self.get_ohlcv_data(ticker, count=count)

            if df is None or len(df) < required_period:
                logger.warning(
                    f"Insufficient data for {ticker}: "
                    f"need {required_period}, got {len(df) if df is not None else 0}"
                )
                return None

            # Use strategy to calculate indicators
            df = self.strategy.calculate_indicators(df)

            # Get latest values (excluding today's incomplete candle)
            # iloc[-2] is yesterday's confirmed candle
            if len(df) < 2:
                return None

            latest = df.iloc[-2]

            return {
                "target": float(latest["target"]),
                "k": float(latest["short_noise"]),
                "long_noise": float(latest["long_noise"]),
                "sma": float(latest["sma"]),
                "sma_trend": float(latest["sma_trend"]),
            }
        except Exception as e:
            logger.error(f"Error calculating metrics for {ticker}: {e}", exc_info=True)
            return None
