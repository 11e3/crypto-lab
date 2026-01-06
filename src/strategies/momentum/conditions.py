"""
Entry and exit conditions for Momentum strategy.

These conditions use momentum indicators like RSI, MACD, and moving averages
to identify trend-following opportunities.
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition


class PriceAboveSMACondition(Condition):
    """
    Condition: Current price must be above SMA.

    Used to confirm uptrend before entry.
    """

    def __init__(self, sma_key: str = "sma", name: str = "PriceAboveSMA") -> None:
        """
        Initialize price above SMA condition.

        Args:
            sma_key: Key for SMA value in indicators dict
            name: Condition name
        """
        super().__init__(name)
        self.sma_key = sma_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if close is above SMA."""
        sma = indicators.get(self.sma_key)

        if sma is None:
            return False

        return current.close > sma


class PriceBelowSMACondition(Condition):
    """
    Exit condition: Close price falls below SMA.

    Standard exit signal for momentum strategy.
    """

    def __init__(self, sma_key: str = "sma", name: str = "PriceBelowSMA") -> None:
        """
        Initialize price below SMA condition.

        Args:
            sma_key: Key for SMA value in indicators dict
            name: Condition name
        """
        super().__init__(name)
        self.sma_key = sma_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if close is below SMA."""
        sma = indicators.get(self.sma_key)

        if sma is None:
            return False

        return current.close < sma


class RSIOversoldCondition(Condition):
    """
    Entry condition: RSI is oversold (below threshold).

    Indicates potential buying opportunity when momentum is recovering.
    """

    def __init__(
        self,
        rsi_key: str = "rsi",
        oversold_threshold: float = 30.0,
        name: str = "RSIOversold",
    ) -> None:
        """
        Initialize RSI oversold condition.

        Args:
            rsi_key: Key for RSI value in indicators dict
            oversold_threshold: RSI threshold for oversold (default 30)
            name: Condition name
        """
        super().__init__(name)
        self.rsi_key = rsi_key
        self.oversold_threshold = oversold_threshold

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if RSI is oversold."""
        rsi = indicators.get(self.rsi_key)

        if rsi is None:
            return False

        return rsi < self.oversold_threshold


class RSIOverboughtCondition(Condition):
    """
    Exit condition: RSI is overbought (above threshold).

    Indicates potential selling opportunity when momentum is weakening.
    """

    def __init__(
        self,
        rsi_key: str = "rsi",
        overbought_threshold: float = 70.0,
        name: str = "RSIOverbought",
    ) -> None:
        """
        Initialize RSI overbought condition.

        Args:
            rsi_key: Key for RSI value in indicators dict
            overbought_threshold: RSI threshold for overbought (default 70)
            name: Condition name
        """
        super().__init__(name)
        self.rsi_key = rsi_key
        self.overbought_threshold = overbought_threshold

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if RSI is overbought."""
        rsi = indicators.get(self.rsi_key)

        if rsi is None:
            return False

        return rsi > self.overbought_threshold


class MACDBullishCondition(Condition):
    """
    Entry condition: MACD line crosses above signal line (bullish crossover).

    Indicates strengthening upward momentum.
    """

    def __init__(
        self,
        macd_key: str = "macd",
        signal_key: str = "macd_signal",
        name: str = "MACDBullish",
    ) -> None:
        """
        Initialize MACD bullish condition.

        Args:
            macd_key: Key for MACD line value in indicators dict
            signal_key: Key for MACD signal line value in indicators dict
            name: Condition name
        """
        super().__init__(name)
        self.macd_key = macd_key
        self.signal_key = signal_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if MACD is above signal line."""
        macd = indicators.get(self.macd_key)
        signal = indicators.get(self.signal_key)

        if macd is None or signal is None:
            return False

        # Check current crossover
        if len(history) < 2:
            return macd > signal

        # Check if MACD just crossed above signal (bullish crossover)
        prev_macd = history.iloc[-2].get("macd")
        prev_signal = history.iloc[-2].get("macd_signal")

        if prev_macd is None or prev_signal is None:
            return macd > signal

        # Bullish crossover: MACD was below signal, now above
        crossover = (prev_macd <= prev_signal) and (macd > signal)
        # Or already above signal line
        above_signal = macd > signal

        return crossover or above_signal


class MACDBearishCondition(Condition):
    """
    Exit condition: MACD line crosses below signal line (bearish crossover).

    Indicates weakening upward momentum.
    """

    def __init__(
        self,
        macd_key: str = "macd",
        signal_key: str = "macd_signal",
        name: str = "MACDBearish",
    ) -> None:
        """
        Initialize MACD bearish condition.

        Args:
            macd_key: Key for MACD line value in indicators dict
            signal_key: Key for MACD signal line value in indicators dict
            name: Condition name
        """
        super().__init__(name)
        self.macd_key = macd_key
        self.signal_key = signal_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if MACD is below signal line."""
        macd = indicators.get(self.macd_key)
        signal = indicators.get(self.signal_key)

        if macd is None or signal is None:
            return False

        # Check current crossover
        if len(history) < 2:
            return macd < signal

        # Check if MACD just crossed below signal (bearish crossover)
        prev_macd = history.iloc[-2].get("macd")
        prev_signal = history.iloc[-2].get("macd_signal")

        if prev_macd is None or prev_signal is None:
            return macd < signal

        # Bearish crossover: MACD was above signal, now below
        crossover = (prev_macd >= prev_signal) and (macd < signal)
        # Or already below signal line
        below_signal = macd < signal

        return crossover or below_signal


class MomentumStrengthCondition(Condition):
    """
    Condition: Momentum strength based on price change over period.

    Filters for strong momentum moves.
    """

    def __init__(
        self,
        lookback: int = 10,
        min_change_pct: float = 0.02,
        name: str = "MomentumStrength",
    ) -> None:
        """
        Initialize momentum strength condition.

        Args:
            lookback: Lookback period for momentum calculation
            min_change_pct: Minimum price change percentage (default 2%)
            name: Condition name
        """
        super().__init__(name)
        self.lookback = lookback
        self.min_change_pct = min_change_pct

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if momentum is strong enough."""
        if len(history) < self.lookback:
            return False

        # Calculate price change over lookback period
        past_close = history.iloc[-self.lookback]["close"]
        if past_close <= 0:
            return False

        change_pct = (current.close - past_close) / past_close
        return change_pct >= self.min_change_pct
