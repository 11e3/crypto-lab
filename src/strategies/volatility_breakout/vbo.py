"""
Vanilla Volatility Breakout (VBO) Strategy.

Implements the classic volatility breakout strategy with modular
conditions and filters that can be added or removed.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.strategies.base import Condition, Strategy
from src.strategies.volatility_breakout.conditions import (  # Backward compatibility aliases
    BreakoutCondition,
    NoiseCondition,
    PriceBelowSMACondition,
    SMABreakoutCondition,
    TrendCondition,
)
from src.utils.indicators import add_vbo_indicators


class VanillaVBO(Strategy):
    """
    Vanilla Volatility Breakout Strategy.

    Default configuration:
    - Entry: Price breaks above target (Open + Range * K)
    - Exit: Close falls below SMA
    - Market conditions: Trend alignment + Noise condition

    The strategy is highly customizable through:
    - Adding/removing conditions
    - Adjusting indicator parameters
    """

    def __init__(
        self,
        name: str = "VanillaVBO",
        sma_period: int = 4,
        trend_sma_period: int = 8,
        short_noise_period: int = 4,
        long_noise_period: int = 8,
        entry_conditions: Sequence[Condition] | None = None,
        exit_conditions: Sequence[Condition] | None = None,
        use_default_conditions: bool = True,
        exclude_current: bool = False,
    ) -> None:
        """
        Initialize Vanilla VBO strategy.

        Args:
            name: Strategy name
            sma_period: Period for exit SMA
            trend_sma_period: Period for trend SMA
            short_noise_period: Period for K value calculation
            long_noise_period: Period for noise baseline
            entry_conditions: Custom entry conditions (optional)
            exit_conditions: Custom exit conditions (optional)
            use_default_conditions: Whether to add default conditions (includes market conditions)
            exclude_current: If True, exclude current bar from calculations (matching legacy/bt.py)
        """
        # Store indicator parameters
        self.sma_period = sma_period
        self.trend_sma_period = trend_sma_period
        self.short_noise_period = short_noise_period
        self.long_noise_period = long_noise_period
        self.exclude_current = exclude_current

        # Build default conditions
        default_entry: list[Condition] = []
        default_exit: list[Condition] = []

        if use_default_conditions:
            default_entry = [
                BreakoutCondition(),
                SMABreakoutCondition(),
                TrendCondition(),  # Formerly TrendFilter
                NoiseCondition(),  # Formerly NoiseFilter
            ]
            default_exit = [
                PriceBelowSMACondition(),
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
            "noise",
            "short_noise",
            "long_noise",
            "sma",
            "sma_trend",
            "target",
            "prev_high",
            "prev_low",
            "prev_range",
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all VBO indicators."""
        return add_vbo_indicators(
            df,
            sma_period=self.sma_period,
            trend_sma_period=self.trend_sma_period,
            short_noise_period=self.short_noise_period,
            long_noise_period=self.long_noise_period,
            exclude_current=self.exclude_current,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals using vectorized operations.

        Respects configured conditions and filters for flexible signal generation.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            DataFrame with 'entry_signal' and 'exit_signal' columns
        """
        df = df.copy()

        # Build entry signal based on configured conditions
        entry_signal = pd.Series(True, index=df.index)

        # Check each entry condition (including market conditions)
        for condition in self.entry_conditions.conditions:
            if condition.name == "Breakout":
                entry_signal = entry_signal & (df["high"] >= df["target"])
            elif condition.name == "SMABreakout":
                entry_signal = entry_signal & (df["target"] > df["sma"])
            elif condition.name == "TrendCondition" or condition.name == "TrendFilter":
                entry_signal = entry_signal & (df["target"] > df["sma_trend"])
            elif condition.name == "NoiseCondition" or condition.name == "NoiseFilter":
                entry_signal = entry_signal & (df["short_noise"] < df["long_noise"])
            elif (
                condition.name == "NoiseThresholdCondition"
                or condition.name == "NoiseThresholdFilter"
            ):
                max_noise = getattr(condition, "max_noise", 0.7)
                entry_signal = entry_signal & (df["short_noise"] <= max_noise)
            elif (
                condition.name == "VolatilityRangeCondition" or condition.name == "VolatilityFilter"
            ):
                min_vol = getattr(condition, "min_volatility_pct", 0.005)
                max_vol = getattr(condition, "max_volatility_pct", 0.15)
                range_pct = df["prev_range"] / df["open"]
                entry_signal = entry_signal & (range_pct >= min_vol) & (range_pct <= max_vol)
            elif condition.name == "VolatilityThreshold":
                min_range_pct = getattr(condition, "min_range_pct", 0.01)
                range_pct = df["prev_range"] / df["open"]
                entry_signal = entry_signal & (range_pct >= min_range_pct)

        # Build exit signal based on configured conditions
        exit_signal = pd.Series(False, index=df.index)

        for condition in self.exit_conditions.conditions:
            if condition.name == "PriceBelowSMA":
                exit_signal = exit_signal | (df["close"] < df["sma"])

        df["entry_signal"] = entry_signal
        df["exit_signal"] = exit_signal

        return df


class MinimalVBO(VanillaVBO):
    """
    Minimal VBO with only breakout condition (no market conditions).

    Useful as a baseline for comparing condition effectiveness.
    """

    def __init__(
        self,
        name: str = "MinimalVBO",
        **kwargs: Any,
    ) -> None:
        # Only use breakout condition, no market conditions
        from src.strategies.volatility_breakout.conditions import BreakoutCondition

        super().__init__(
            name=name,
            entry_conditions=[BreakoutCondition()],
            use_default_conditions=False,
            **kwargs,
        )


class StrictVBO(VanillaVBO):
    """
    Strict VBO with additional conditions for higher quality signals.

    Includes noise threshold and volatility range conditions.
    """

    def __init__(
        self,
        name: str = "StrictVBO",
        max_noise: float = 0.6,
        min_volatility_pct: float = 0.01,
        **kwargs: Any,
    ) -> None:
        from src.strategies.volatility_breakout.conditions import (
            NoiseThresholdCondition,
            VolatilityRangeCondition,
        )

        extra_conditions = [
            NoiseThresholdCondition(max_noise=max_noise),
            VolatilityRangeCondition(min_volatility_pct=min_volatility_pct),
        ]

        # Add to existing entry conditions
        existing_entry = kwargs.get("entry_conditions", [])
        entry_conditions = list(existing_entry) + extra_conditions
        kwargs["entry_conditions"] = entry_conditions

        super().__init__(
            name=name,
            **kwargs,
        )


def create_vbo_strategy(
    name: str = "CustomVBO",
    sma_period: int = 4,
    trend_sma_period: int = 8,
    short_noise_period: int = 4,
    long_noise_period: int = 8,
    use_breakout: bool = True,
    use_sma_breakout: bool = True,
    use_sma_exit: bool = True,
    use_trend_filter: bool = True,
    use_noise_filter: bool = True,
    extra_entry_conditions: Sequence[Condition] | None = None,
    extra_exit_conditions: Sequence[Condition] | None = None,
    exclude_current: bool = False,
) -> VanillaVBO:
    """
    Factory function to create customized VBO strategy.

    Provides fine-grained control over which conditions
    are included in the strategy.

    Args:
        name: Strategy name
        sma_period: Period for exit SMA
        trend_sma_period: Period for trend SMA
        short_noise_period: Period for K value
        long_noise_period: Period for noise baseline
        use_breakout: Include breakout condition
        use_sma_breakout: Include SMA breakout condition
        use_sma_exit: Include SMA exit condition
        use_trend_filter: Include trend condition (formerly filter)
        use_noise_filter: Include noise condition (formerly filter)
        extra_entry_conditions: Additional entry conditions
        extra_exit_conditions: Additional exit conditions
        exclude_current: If True, exclude current bar from calculations (matching legacy/bt.py)

    Returns:
        Configured VanillaVBO instance

    Example:
        Create VBO without noise condition::

            strategy = create_vbo_strategy(
                name="VBO_NoNoise",
                use_noise_filter=False,
            )

        Add custom momentum condition::

            from src.strategies.volatility_breakout.conditions import ConsecutiveUpCondition
            strategy = create_vbo_strategy(
                name="VBO_Momentum",
                extra_entry_conditions=[ConsecutiveUpCondition(days=2)],
            )
    """
    # Build entry conditions
    entry_conditions: list[Condition] = []
    if use_breakout:
        entry_conditions.append(BreakoutCondition())
    if use_sma_breakout:
        entry_conditions.append(SMABreakoutCondition())
    if use_trend_filter:
        entry_conditions.append(TrendCondition())
    if use_noise_filter:
        entry_conditions.append(NoiseCondition())
    if extra_entry_conditions:
        entry_conditions.extend(list(extra_entry_conditions))

    # Build exit conditions
    exit_conditions: list[Condition] = []
    if use_sma_exit:
        exit_conditions.append(PriceBelowSMACondition())
    if extra_exit_conditions:
        exit_conditions.extend(list(extra_exit_conditions))

    # Create strategy with custom components
    return VanillaVBO(
        name=name,
        sma_period=sma_period,
        trend_sma_period=trend_sma_period,
        short_noise_period=short_noise_period,
        long_noise_period=long_noise_period,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        use_default_conditions=False,
        exclude_current=exclude_current,
    )


# Convenience function for quick testing
def quick_vbo(
    sma: int = 4,
    n: int = 2,
) -> VanillaVBO:
    """
    Create VBO with simplified parameters matching original bt.py.

    Args:
        sma: Base SMA period
        n: Multiplier for trend/long-term periods

    Returns:
        VanillaVBO instance
    """
    return VanillaVBO(
        name=f"VBO_SMA{sma}_N{n}",
        sma_period=sma,
        trend_sma_period=sma * n,
        short_noise_period=sma,
        long_noise_period=sma * n,
    )
