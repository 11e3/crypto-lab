"""
Alpha signal generators.

Provides various signal generation strategies:
- Technical signals (indicators-based)
- Momentum signals
- Mean reversion signals
- Breakout signals
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SignalOutput:
    """Output from a signal generator."""

    # Signal values: ticker -> signal strength (-1 to 1)
    signals: dict[str, float]

    # Confidence: ticker -> confidence (0 to 1)
    confidence: dict[str, float] = field(default_factory=dict)

    # Metadata
    signal_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.signals)


class AlphaSignal(ABC):
    """
    Abstract base class for alpha signals.

    Signals generate trading signals based on various inputs.
    Signal values range from -1 (strong sell) to 1 (strong buy).

    Example:
        >>> signal = MomentumSignal(lookback=20)
        >>> output = signal.generate(prices)
        >>> # output.signals = {"BTC": 0.8, "ETH": -0.3, ...}
    """

    def __init__(self, name: str) -> None:
        """
        Initialize signal.

        Args:
            name: Signal identifier
        """
        self.name = name

    @abstractmethod
    def generate(
        self,
        data: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> SignalOutput:
        """
        Generate signals.

        Args:
            data: Price/OHLCV data
            timestamp: Current timestamp

        Returns:
            SignalOutput with signal values
        """
        pass

    def _normalize_signal(self, value: float) -> float:
        """Normalize signal to [-1, 1] range."""
        return max(-1, min(1, value))


class TechnicalSignal(AlphaSignal):
    """
    Technical indicator-based signals.

    Combines multiple technical indicators.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ) -> None:
        super().__init__("technical")
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate(
        self,
        data: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> SignalOutput:
        """Generate technical signals."""
        signals = {}
        confidence = {}

        for ticker in data.columns:
            prices = data[ticker].dropna()
            if len(prices) < max(self.rsi_period, self.macd_slow, self.bb_period) + 10:
                continue

            # Calculate indicators
            rsi_signal = self._calculate_rsi_signal(prices)
            macd_signal = self._calculate_macd_signal(prices)
            bb_signal = self._calculate_bb_signal(prices)

            # Combine signals
            combined = (rsi_signal + macd_signal + bb_signal) / 3
            signals[ticker] = self._normalize_signal(combined)

            # Confidence based on indicator agreement
            indicator_signals = [rsi_signal, macd_signal, bb_signal]
            same_direction = sum(1 for s in indicator_signals if s * combined > 0)
            confidence[ticker] = same_direction / len(indicator_signals)

        return SignalOutput(
            signals=signals,
            confidence=confidence,
            signal_name=self.name,
            timestamp=timestamp or datetime.now(),
        )

    def _calculate_rsi_signal(self, prices: pd.Series) -> float:
        """Calculate RSI-based signal."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # Signal: oversold (< 30) = buy, overbought (> 70) = sell
        if current_rsi < 30:
            return (30 - current_rsi) / 30  # 0 to 1
        elif current_rsi > 70:
            return -(current_rsi - 70) / 30  # 0 to -1
        else:
            return 0

    def _calculate_macd_signal(self, prices: pd.Series) -> float:
        """Calculate MACD-based signal."""
        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()

        histogram = macd_line - signal_line

        # Normalize by price level
        current_hist = histogram.iloc[-1] / prices.iloc[-1]

        # Signal: positive histogram = bullish, negative = bearish
        return self._normalize_signal(current_hist * 100)

    def _calculate_bb_signal(self, prices: pd.Series) -> float:
        """Calculate Bollinger Bands signal."""
        sma = prices.rolling(self.bb_period).mean()
        std = prices.rolling(self.bb_period).std()

        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std

        current_price = prices.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_sma = sma.iloc[-1]

        band_width = current_upper - current_lower
        if band_width == 0:
            return 0

        # Position within bands
        position = (current_price - current_lower) / band_width

        # Signal: below lower band = buy, above upper = sell
        if position < 0:
            return min(1, -position)  # Buy signal
        elif position > 1:
            return max(-1, 1 - position)  # Sell signal
        else:
            return 0


class MomentumSignal(AlphaSignal):
    """
    Price momentum signal.

    Generates signals based on price momentum.
    """

    def __init__(
        self,
        lookback: int = 20,
        normalize_by_volatility: bool = True,
    ) -> None:
        super().__init__("momentum")
        self.lookback = lookback
        self.normalize_by_volatility = normalize_by_volatility

    def generate(
        self,
        data: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> SignalOutput:
        """Generate momentum signals."""
        signals = {}
        confidence = {}

        for ticker in data.columns:
            prices = data[ticker].dropna()
            if len(prices) < self.lookback + 5:
                continue

            # Calculate momentum
            returns = prices.pct_change()
            momentum = prices.iloc[-1] / prices.iloc[-self.lookback] - 1

            if self.normalize_by_volatility:
                volatility = returns.iloc[-self.lookback:].std()
                if volatility > 0:
                    momentum = momentum / (volatility * np.sqrt(self.lookback))

            signals[ticker] = self._normalize_signal(momentum)

            # Confidence based on consistency
            period_returns = returns.iloc[-self.lookback:]
            if momentum > 0:
                confidence[ticker] = (period_returns > 0).mean()
            else:
                confidence[ticker] = (period_returns < 0).mean()

        return SignalOutput(
            signals=signals,
            confidence=confidence,
            signal_name=self.name,
            timestamp=timestamp or datetime.now(),
        )


class MeanReversionSignal(AlphaSignal):
    """
    Mean reversion signal.

    Generates signals based on deviation from moving average.
    """

    def __init__(
        self,
        lookback: int = 20,
        z_score_threshold: float = 2.0,
    ) -> None:
        super().__init__("mean_reversion")
        self.lookback = lookback
        self.z_score_threshold = z_score_threshold

    def generate(
        self,
        data: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> SignalOutput:
        """Generate mean reversion signals."""
        signals = {}
        confidence = {}

        for ticker in data.columns:
            prices = data[ticker].dropna()
            if len(prices) < self.lookback + 5:
                continue

            # Calculate z-score
            lookback_prices = prices.iloc[-self.lookback:]
            mean = lookback_prices.mean()
            std = lookback_prices.std()

            if std == 0:
                continue

            z_score = (prices.iloc[-1] - mean) / std

            # Signal: negative z-score = buy (below mean), positive = sell
            # Invert because mean reversion expects return to mean
            signal = -z_score / self.z_score_threshold
            signals[ticker] = self._normalize_signal(signal)

            # Confidence based on how extreme the z-score is
            confidence[ticker] = min(1, abs(z_score) / self.z_score_threshold)

        return SignalOutput(
            signals=signals,
            confidence=confidence,
            signal_name=self.name,
            timestamp=timestamp or datetime.now(),
        )


class BreakoutSignal(AlphaSignal):
    """
    Breakout signal.

    Generates signals based on price breakouts from ranges.
    """

    def __init__(
        self,
        lookback: int = 20,
        breakout_threshold: float = 0.02,
    ) -> None:
        super().__init__("breakout")
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold

    def generate(
        self,
        data: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> SignalOutput:
        """Generate breakout signals."""
        signals = {}
        confidence = {}

        for ticker in data.columns:
            prices = data[ticker].dropna()
            if len(prices) < self.lookback + 5:
                continue

            # Calculate range
            lookback_prices = prices.iloc[-self.lookback:-1]  # Exclude current
            high = lookback_prices.max()
            low = lookback_prices.min()
            range_width = high - low

            current = prices.iloc[-1]

            if range_width == 0:
                continue

            # Check for breakout
            if current > high * (1 + self.breakout_threshold):
                # Bullish breakout
                breakout_pct = (current - high) / high
                signals[ticker] = min(1, breakout_pct / self.breakout_threshold)
            elif current < low * (1 - self.breakout_threshold):
                # Bearish breakout
                breakout_pct = (low - current) / low
                signals[ticker] = max(-1, -breakout_pct / self.breakout_threshold)
            else:
                signals[ticker] = 0

            # Confidence based on volume (if available) or range position
            confidence[ticker] = abs(signals[ticker])

        return SignalOutput(
            signals=signals,
            confidence=confidence,
            signal_name=self.name,
            timestamp=timestamp or datetime.now(),
        )


class VolumeSignal(AlphaSignal):
    """
    Volume-based signal.

    Generates signals based on unusual volume patterns.
    """

    def __init__(
        self,
        lookback: int = 20,
        volume_threshold: float = 2.0,
    ) -> None:
        super().__init__("volume")
        self.lookback = lookback
        self.volume_threshold = volume_threshold

    def generate(
        self,
        data: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> SignalOutput:
        """Generate volume signals."""
        # This requires volume data - placeholder for now
        return SignalOutput(
            signals={},
            confidence={},
            signal_name=self.name,
            timestamp=timestamp or datetime.now(),
        )


__all__ = [
    "AlphaSignal",
    "SignalOutput",
    "TechnicalSignal",
    "MomentumSignal",
    "MeanReversionSignal",
    "BreakoutSignal",
    "VolumeSignal",
]
