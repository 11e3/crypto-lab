"""
Strategy metrics calculation for signal generation.

Separates metrics calculation responsibility from signal detection (SRP).
"""

from src.execution.signal_data import SignalDataLoader
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalMetricsCalculator:
    """
    Calculates strategy metrics from market data.

    Handles metrics calculation, separate from signal detection logic.
    """

    def __init__(
        self,
        strategy: Strategy,
        data_loader: SignalDataLoader,
    ) -> None:
        """
        Initialize metrics calculator.

        Args:
            strategy: Trading strategy instance
            data_loader: Data loader for OHLCV data
        """
        self.strategy = strategy
        self.data_loader = data_loader

    def calculate(
        self,
        ticker: str,
        required_period: int | None = None,
    ) -> dict[str, float] | None:
        """
        Calculate strategy metrics for a ticker.

        Args:
            ticker: Trading pair symbol
            required_period: Minimum period required (uses default 20 if None)

        Returns:
            Dictionary with metrics (target, k, long_noise, sma, sma_trend) or None
        """
        try:
            if required_period is None:
                required_period = 20

            count = max(required_period + 5, self.data_loader.min_data_points * 2)
            df = self.data_loader.get_ohlcv(ticker, count=count)

            if df is None or len(df) < required_period:
                logger.warning(
                    f"Insufficient data for {ticker}: "
                    f"need {required_period}, got {len(df) if df is not None else 0}"
                )
                return None

            df = self.strategy.calculate_indicators(df)

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
