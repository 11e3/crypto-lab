"""
Base classes for strategy abstraction.

Provides modular interfaces for building trading strategies with
composable conditions and filters.

This module re-exports all base components for backward compatibility:
- SignalType, Signal, Position, OHLCV from base_models
- Condition, CompositeCondition from base_conditions
- Strategy ABC (defined here)
"""

from abc import ABC, abstractmethod

import pandas as pd

# Re-export models for backward compatibility
from src.strategies.base_conditions import CompositeCondition, Condition
from src.strategies.base_models import OHLCV, Position, Signal, SignalType

__all__ = [
    # Models
    "SignalType",
    "Signal",
    "Position",
    "OHLCV",
    # Conditions
    "Condition",
    "CompositeCondition",
    # Strategy
    "Strategy",
]


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies combine conditions to generate trading signals based on:
    - entry_conditions: AND-combined technical conditions for buy signals.
    - exit_conditions: AND-combined conditions for sell signals.

    Default VBO (Volatility Breakout) logic:
    1. Breakout: buy when price rises k * prev_range above open.
    2. Trend filter: only enter above SMA (confirms uptrend).
    3. Next-day exit: sell at next close to limit overnight risk.

    Entry price: target = open + prev_range * k
    Exit price: next-day close

    Subclasses must implement:
    - name property: unique strategy identifier (e.g. "VBO_K0.5").
    - required_indicators(): list of required indicator column names.
    - calculate_indicators(): builds those columns on the DataFrame.

    The default generate_signals() provides standard VBO signal generation,
    which can be overridden for custom signal logic.
    """

    def __init__(
        self,
        name: str | None = None,
        entry_conditions: list[Condition] | None = None,
        exit_conditions: list[Condition] | None = None,
    ) -> None:
        """
        Initialize strategy with conditions.

        Args:
            name: Strategy name (optional, uses class name if not provided)
            entry_conditions: List of entry conditions (default empty)
            exit_conditions: List of exit conditions (default empty)
        """
        self._name = name
        self.entry_conditions: CompositeCondition = CompositeCondition(
            entry_conditions or [], "AND"
        )
        self.exit_conditions: CompositeCondition = CompositeCondition(exit_conditions or [], "AND")

    @property
    def name(self) -> str:
        """
        Strategy name identifier.

        Returns:
            Unique name for the strategy (e.g., "VBO_K0.5", "MomentumSMA20")
        """
        return self._name or self.__class__.__name__

    @property
    def is_pair_trading(self) -> bool:
        """Return True if this is a pair trading strategy.

        Subclasses that implement pair trading should override this property.

        Returns:
            False by default; True for pair trading strategies.
        """
        return False

    @property
    def exit_price_column(self) -> str:
        """Column name to use as exit price in backtesting.

        Override in subclasses that use a non-close exit price.
        For example, VBOV1 exits at the next day's open price:

            @property
            def exit_price_column(self) -> str:
                return "exit_price_base"

        The data loader will use df[exit_price_column] as 'exit_price'.
        The column must be present in the DataFrame returned by generate_signals().

        Returns:
            Column name (default: "close")
        """
        return "close"

    @classmethod
    def parameter_schema(cls) -> dict[str, object]:
        """Return the parameter schema for optimization.

        Override in subclasses to expose tunable parameters.
        Used by the web optimization layer to build the parameter sweep UI.

        Returns:
            Dict mapping parameter names to their range/type spec.
            Example::

                {
                    "k": {"type": "float", "min": 0.1, "max": 0.9, "step": 0.1},
                    "sma_period": {"type": "int", "min": 5, "max": 60, "step": 5},
                }

            Empty dict means no tunable parameters (default).
        """
        return {}

    @abstractmethod
    def required_indicators(self) -> list[str]:
        """
        List of indicator names required by this strategy.

        Every name returned here must be produced by calculate_indicators().
        Example: ["sma", "target", "noise"] → df must have those columns.

        Returns:
            List of indicator column names required for signal generation.
        """
        pass

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required indicators for the strategy.

        Accuracy here directly affects return calculations:
        - Correct indicators → reliable signals → better performance.
        - Wrong indicators → distorted signals → losses.

        Each indicator must satisfy:
        1. Correct formula (e.g. SMA(n) as a proper rolling mean).
        2. Adequate lookback (SMA(20) needs at least 20 bars).
        3. NaN for insufficient data — caller handles dropna/forward-fill.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume).

        Returns:
            Copy of df with added indicator columns (sma, target, noise, etc.).
        """
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals using vectorized operations.

        Default VBO logic (override for custom strategies):
        1. Entry: high >= target AND target > sma (breakout + trend filter).
        2. Optional filters: sma_trend (long-term trend), noise (volatility gate).
        3. Exit: close < sma (trend reversal triggers immediate sell).

        Args:
            df: DataFrame with OHLCV and indicator columns.

        Returns:
            Copy of df with 'entry_signal' and 'exit_signal' boolean columns.
        """
        df = df.copy()
        entry_signal = self._build_entry_signal(df)
        exit_signal = df["close"] < df["sma"]

        if "entry_signal" not in df.columns:
            df["entry_signal"] = entry_signal
        if "exit_signal" not in df.columns:
            df["exit_signal"] = exit_signal

        return df

    def _build_entry_signal(self, df: pd.DataFrame) -> pd.Series:
        """Build the vectorized entry signal for default VBO logic.

        Combines breakout, trend filter, and optional noise/trend filters.
        """
        # Breakout: high crosses the target price (previous range * noise)
        entry_signal = (df["high"] >= df["target"]) & (df["target"] > df["sma"])

        # Long-term trend gate: target must also be above the slower SMA
        if "sma_trend" in df.columns:
            entry_signal = entry_signal & (df["target"] > df["sma_trend"])

        # Noise filter: only trade when short-term noise < long-term noise
        # (low short_noise relative to long_noise means a cleaner trend)
        if "short_noise" in df.columns and "long_noise" in df.columns:
            entry_signal = entry_signal & (df["short_noise"] < df["long_noise"])

        return entry_signal

    def check_entry(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """
        Check if entry conditions are met.

        Args:
            current: Current bar data
            history: Historical data
            indicators: Current indicator values

        Returns:
            True if should enter position
        """
        return self.entry_conditions.evaluate(current, history, indicators)

    def check_exit(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
        position: Position,
    ) -> bool:
        """
        Check if exit conditions are met.

        Args:
            current: Current bar data
            history: Historical data
            indicators: Current indicator values
            position: Current position

        Returns:
            True if should exit position
        """
        return self.exit_conditions.evaluate(current, history, indicators)

    def add_entry_condition(self, condition: Condition) -> "Strategy":
        """Add entry condition and return self for chaining."""
        self.entry_conditions.add(condition)
        return self

    def add_exit_condition(self, condition: Condition) -> "Strategy":
        """Add exit condition and return self for chaining."""
        self.exit_conditions.add(condition)
        return self

    def remove_entry_condition(self, condition: Condition) -> "Strategy":
        """Remove entry condition and return self for chaining."""
        self.entry_conditions.remove(condition)
        return self

    def remove_exit_condition(self, condition: Condition) -> "Strategy":
        """Remove exit condition and return self for chaining."""
        self.exit_conditions.remove(condition)
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"entry_conditions={len(self.entry_conditions.conditions)}, "
            f"exit_conditions={len(self.exit_conditions.conditions)})"
        )
