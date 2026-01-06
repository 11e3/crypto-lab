"""
Mean Reversion Trading Strategy.

Implements mean reversion strategy using Bollinger Bands, RSI, and moving averages.
Assumes prices will revert to their mean after deviating.
"""

import pandas as pd

from src.strategies.base import Condition, Strategy
from src.strategies.mean_reversion.conditions import (
    BollingerLowerBandCondition,
    BollingerUpperBandCondition,
    PriceAboveSMACondition,
    PriceBelowSMACondition,
    RSIOverboughtCondition,
    RSIOversoldCondition,
)
from src.utils.indicators import bollinger_bands, rsi, sma


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Trading Strategy.

    Default configuration:
    - Entry: Price below SMA + touches lower Bollinger Band + RSI oversold
    - Exit: Price above SMA OR touches upper Bollinger Band OR RSI overbought

    The strategy assumes prices will revert to their mean after deviating.
    Best suited for ranging/consolidating markets.
    """

    def __init__(
        self,
        name: str = "MeanReversionStrategy",
        sma_period: int = 20,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        entry_conditions: list[Condition] | None = None,
        exit_conditions: list[Condition] | None = None,
        use_default_conditions: bool = True,
    ) -> None:
        """
        Initialize Mean Reversion strategy.

        Args:
            name: Strategy name
            sma_period: Period for SMA calculation
            bb_period: Period for Bollinger Bands
            bb_std: Standard deviation multiplier for Bollinger Bands
            rsi_period: Period for RSI calculation
            rsi_oversold: RSI threshold for oversold (entry signal)
            rsi_overbought: RSI threshold for overbought (exit signal)
            entry_conditions: Custom entry conditions (optional)
            exit_conditions: Custom exit conditions (optional)
            use_default_conditions: Whether to add default conditions
        """
        # Store indicator parameters
        self.sma_period = sma_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Build default conditions
        default_entry: list[Condition] = []
        default_exit: list[Condition] = []

        if use_default_conditions:
            default_entry = [
                PriceBelowSMACondition(sma_key="sma"),
                BollingerLowerBandCondition(),
                RSIOversoldCondition(
                    rsi_key="rsi", oversold_threshold=rsi_oversold
                ),
            ]
            default_exit = [
                PriceAboveSMACondition(sma_key="sma"),
                BollingerUpperBandCondition(),
                RSIOverboughtCondition(
                    rsi_key="rsi", overbought_threshold=rsi_overbought
                ),
            ]

        # Merge with custom conditions
        all_entry = (entry_conditions or []) + default_entry
        all_exit = (exit_conditions or []) + default_exit

        super().__init__(
            name=name,
            entry_conditions=all_entry,
            exit_conditions=all_exit,
        )

    def required_indicators(self) -> list[str]:
        """Return list of required indicators."""
        return [
            "sma",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "rsi",
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all mean reversion indicators."""
        df = df.copy()

        # Simple Moving Average for mean
        df["sma"] = sma(df["close"], self.sma_period)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger_bands(
            df["close"],
            period=self.bb_period,
            std_dev=self.bb_std,
        )
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower

        # RSI for momentum confirmation
        df["rsi"] = rsi(df["close"], self.rsi_period)

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
            if condition.name == "PriceBelowSMA":
                entry_signal = entry_signal & (df["close"] < df["sma"])
            elif condition.name == "BollingerLowerBand":
                entry_signal = entry_signal & (df["low"] <= df["bb_lower"])
            elif condition.name == "RSIOversold":
                oversold_threshold = getattr(condition, "oversold_threshold", 30.0)
                entry_signal = entry_signal & (df["rsi"] < oversold_threshold)

        # Build exit signal based on configured conditions
        exit_signal = pd.Series(False, index=df.index)

        for condition in self.exit_conditions.conditions:
            if condition.name == "PriceAboveSMA":
                exit_signal = exit_signal | (df["close"] > df["sma"])
            elif condition.name == "BollingerUpperBand":
                exit_signal = exit_signal | (df["high"] >= df["bb_upper"])
            elif condition.name == "RSIOverbought":
                overbought_threshold = getattr(condition, "overbought_threshold", 70.0)
                exit_signal = exit_signal | (df["rsi"] > overbought_threshold)

        df["entry_signal"] = entry_signal
        df["exit_signal"] = exit_signal

        return df


class SimpleMeanReversionStrategy(MeanReversionStrategy):
    """
    Simplified mean reversion strategy with only Bollinger Bands.

    Useful as a baseline for comparing condition effectiveness.
    """

    def __init__(
        self,
        name: str = "SimpleMeanReversion",
        **kwargs,
    ) -> None:
        # Only use Bollinger Bands
        super().__init__(
            name=name,
            entry_conditions=[BollingerLowerBandCondition()],
            exit_conditions=[BollingerUpperBandCondition()],
            use_default_conditions=False,
            **kwargs,
        )
