"""
Momentum Trading Strategy.

Implements trend-following momentum strategy using RSI, MACD, and moving averages.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.strategies.base import Condition, Strategy
from src.strategies.momentum.conditions import (
    MACDBearishCondition,
    MACDBullishCondition,
    PriceAboveSMACondition,
    PriceBelowSMACondition,
    RSIOverboughtCondition,
)
from src.utils.indicators import macd, rsi, sma


class MomentumStrategy(Strategy):
    """
    Momentum Trading Strategy.

    Default configuration:
    - Entry: Price above SMA + RSI recovering from oversold + MACD bullish
    - Exit: Price below SMA OR RSI overbought OR MACD bearish

    The strategy follows trends by entering when momentum indicators
    align in a bullish direction.
    """

    def __init__(
        self,
        name: str = "MomentumStrategy",
        sma_period: int = 20,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        entry_conditions: Sequence[Condition] | None = None,
        exit_conditions: Sequence[Condition] | None = None,
        use_default_conditions: bool = True,
    ) -> None:
        """
        Initialize Momentum strategy.

        Args:
            name: Strategy name
            sma_period: Period for trend SMA
            rsi_period: Period for RSI calculation
            macd_fast: Fast EMA period for MACD
            macd_slow: Slow EMA period for MACD
            macd_signal: Signal line period for MACD
            rsi_oversold: RSI threshold for oversold (entry signal)
            rsi_overbought: RSI threshold for overbought (exit signal)
            entry_conditions: Custom entry conditions (optional)
            exit_conditions: Custom exit conditions (optional)
            use_default_conditions: Whether to add default conditions
        """
        # Store indicator parameters
        self.sma_period = sma_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Build default conditions
        default_entry: list[Condition] = []
        default_exit: list[Condition] = []

        if use_default_conditions:
            default_entry = [
                PriceAboveSMACondition(sma_key="sma"),
                MACDBullishCondition(),
            ]
            default_exit = [
                PriceBelowSMACondition(sma_key="sma"),
                RSIOverboughtCondition(rsi_key="rsi", overbought_threshold=rsi_overbought),
                MACDBearishCondition(),
            ]

        # Merge with custom conditions
        all_entry = list(entry_conditions or []) + default_entry
        all_exit = list(exit_conditions or []) + default_exit

        super().__init__(
            name=name,
            entry_conditions=all_entry,
            exit_conditions=all_exit,
        )

    def required_indicators(self) -> list[str]:
        """Return list of required indicators."""
        return [
            "sma",
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all momentum indicators."""
        df = df.copy()

        # Simple Moving Average for trend
        df["sma"] = sma(df["close"], self.sma_period)

        # RSI for momentum strength
        df["rsi"] = rsi(df["close"], self.rsi_period)

        # MACD for momentum direction
        macd_line, signal_line, histogram = macd(
            df["close"],
            fast_period=self.macd_fast,
            slow_period=self.macd_slow,
            signal_period=self.macd_signal,
        )
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals using vectorized operations.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            DataFrame with 'entry_signal' and 'exit_signal' columns
        """
        df = df.copy()

        # Build entry signal based on configured conditions
        entry_signal = pd.Series(True, index=df.index)

        # Check each entry condition
        for condition in self.entry_conditions.conditions:
            if condition.name == "PriceAboveSMA":
                entry_signal = entry_signal & (df["close"] > df["sma"])
            elif condition.name == "MACDBullish":
                # MACD above signal line
                entry_signal = entry_signal & (df["macd"] > df["macd_signal"])

        # Build exit signal based on configured conditions
        exit_signal = pd.Series(False, index=df.index)

        for condition in self.exit_conditions.conditions:
            if condition.name == "PriceBelowSMA":
                exit_signal = exit_signal | (df["close"] < df["sma"])
            elif condition.name == "RSIOverbought":
                overbought_threshold = getattr(condition, "overbought_threshold", 70.0)
                exit_signal = exit_signal | (df["rsi"] > overbought_threshold)
            elif condition.name == "MACDBearish":
                # MACD below signal line
                exit_signal = exit_signal | (df["macd"] < df["macd_signal"])

        df["entry_signal"] = entry_signal
        df["exit_signal"] = exit_signal

        return df


class SimpleMomentumStrategy(MomentumStrategy):
    """
    Simplified momentum strategy with only price above SMA condition.

    Useful as a baseline for comparing condition effectiveness.
    """

    def __init__(
        self,
        name: str = "SimpleMomentum",
        **kwargs: Any,
    ) -> None:
        # Only use price above SMA condition
        super().__init__(
            name=name,
            entry_conditions=[PriceAboveSMACondition(sma_key="sma")],
            exit_conditions=[PriceBelowSMACondition(sma_key="sma")],
            use_default_conditions=False,
            **kwargs,
        )
