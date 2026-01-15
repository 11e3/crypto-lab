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
from src.strategies.volatility_breakout.vbo_indicators import calculate_vbo_indicators
from src.strategies.volatility_breakout.vbo_signals import (
    build_entry_signal,
    build_exit_signal,
)


class VanillaVBO(Strategy):
    """
    Vanilla Volatility Breakout Strategy.

    Profit Mechanism:
    1. Volatility-based Target: target = Open + Range * K (Higher K = conservative)
    2. Breakout Entry: High >= target (volatility expansion signal)
    3. SMA Filter: target > SMA (confirms uptrend)
    4. Trend Filter: target > SMA_Trend (long-term uptrend confirmation)
    5. Noise Filter: short_noise < long_noise (volatility stability)
    6. SMA Exit: Close < SMA (trend reversal = exit)

    Default config: Entry on breakout, Exit below SMA, with trend+noise filters.
    Highly customizable through conditions and indicator parameters.
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
        # Phase 2 Improvements (v2 features)
        use_improved_noise: bool = False,
        use_adaptive_k: bool = False,
        atr_period: int = 14,
        base_k: float = 0.5,
    ) -> None:
        """
        Initialize Vanilla VBO strategy.

        Args:
            name: Strategy name
            sma_period: SMA period (Exit signal generation)
                       -> Shorter: Frequent exits. Longer: Trend following.
                       Default 4: Very short-term, high frequency.
            trend_sma_period: Long-term trend SMA period
                             -> Trend filter role, confirms long-term uptrend.
                             Default 8: Mid-term trend confirmation.
            short_noise_period: K-value calculation period (Short-term volatility)
                               -> Reflects recent volatility (High sensitivity).
                               Default 4: Applies very recent volatility.
            long_noise_period: Noise baseline period (Long-term volatility)
                              -> Determines stability by comparing with short_noise.
                              Default 8: Mid-term average volatility.
            entry_conditions: Custom entry conditions (optional)
            exit_conditions: Custom exit conditions (optional)
            use_default_conditions: Whether to add default conditions (includes market conditions)
                                   Recommended to use default.
            exclude_current: If True, exclude current bar from calculations (matching legacy/bt.py)
                            Use only past close data (No look-ahead bias).
            use_improved_noise: Enable Phase 2 ATR-normalized noise (v2 feature)
            use_adaptive_k: Enable Phase 2 dynamic K-value adjustment (v2 feature)
            atr_period: ATR period for improved indicators (default: 14)
            base_k: Base K value for adaptive calculation (default: 0.5)
        """
        # Store indicator parameters
        self.sma_period = sma_period
        self.trend_sma_period = trend_sma_period
        self.short_noise_period = short_noise_period
        self.long_noise_period = long_noise_period
        self.exclude_current = exclude_current

        # Store Phase 2 feature flags
        self.use_improved_noise = use_improved_noise
        self.use_adaptive_k = use_adaptive_k
        self.atr_period = atr_period
        self.base_k = base_k

        # Build default conditions
        default_entry: list[Condition] = []
        default_exit: list[Condition] = []

        if use_default_conditions:
            default_entry = [
                BreakoutCondition(),  # 고가 돌파: high >= target
                SMABreakoutCondition(),  # SMA 필터: target > SMA
                TrendCondition(),  # 장기 추세: target > sma_trend
                NoiseCondition(),  # 노이즈 필터: short_noise < long_noise
            ]
            default_exit = [
                PriceBelowSMACondition(),  # SMA 퇴출: close < SMA
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
        """Return list of required indicators.

        All indicators required by VBO strategy:
        - noise: Volatility-based K value
        - short_noise: Short-term average volatility (High sensitivity)
        - long_noise: Long-term average volatility (High stability)
        - sma: Short-term Simple Moving Average (Exit signal)
        - sma_trend: Long-term Simple Moving Average (Trend filter)
        - target: Volatility-based breakout target price
        - prev_high: Previous day's high
        - prev_low: Previous day's low
        - prev_range: Previous day's range (High - Low)
        """
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
        """Calculate all VBO indicators. See vbo_indicators module for details."""
        return calculate_vbo_indicators(
            df,
            sma_period=self.sma_period,
            trend_sma_period=self.trend_sma_period,
            short_noise_period=self.short_noise_period,
            long_noise_period=self.long_noise_period,
            exclude_current=self.exclude_current,
            use_improved_noise=self.use_improved_noise,
            use_adaptive_k=self.use_adaptive_k,
            atr_period=self.atr_period,
            base_k=self.base_k,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals using vectorized operations.

        Uses modular signal builders from vbo_signals module.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            DataFrame with 'entry_signal' and 'exit_signal' columns
        """
        df = df.copy()

        df["entry_signal"] = build_entry_signal(df, list(self.entry_conditions.conditions))
        df["exit_signal"] = build_exit_signal(df, list(self.exit_conditions.conditions))

        return df


class MinimalVBO(VanillaVBO):
    """Minimal VBO with only breakout condition (no market conditions)."""

    def __init__(self, name: str = "MinimalVBO", **kwargs: Any) -> None:
        super().__init__(
            name=name,
            entry_conditions=[BreakoutCondition()],
            use_default_conditions=False,
            **kwargs,
        )


class StrictVBO(VanillaVBO):
    """Strict VBO with additional conditions for higher quality signals."""

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

        extra: list[Condition] = [
            NoiseThresholdCondition(max_noise=max_noise),
            VolatilityRangeCondition(min_volatility_pct=min_volatility_pct),
        ]
        existing: Sequence[Condition] = kwargs.get("entry_conditions", [])
        kwargs["entry_conditions"] = list(existing) + extra
        super().__init__(name=name, **kwargs)


# Re-export factory functions for backward compatibility
from src.strategies.volatility_breakout.vbo_factory import (  # noqa: E402
    create_vbo_strategy,
    quick_vbo,
)

__all__ = [
    "VanillaVBO",
    "MinimalVBO",
    "StrictVBO",
    "create_vbo_strategy",
    "quick_vbo",
]
