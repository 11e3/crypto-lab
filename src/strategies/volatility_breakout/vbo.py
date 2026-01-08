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

    Profit Mechanism:
    1. Volatility-based Target Setting
       target = Previous Open + (Previous High - Previous Low) * K
       K = (Previous High - Previous Low) / (Larger Range)
       -> Higher volatility results in a higher target = more conservative entry.

    2. Breakout Entry
       Entry Condition: Current High >= target
       Meaning: Signal of volatility expansion + Trend reversal.

    3. SMA Filter (Trend Filter)
       target > SMA -> Confirms uptrend.
       -> Removes cases with volatility but no trend (False signal filtering).

    4. Trend Filter (Long-term Trend)
       target > SMA_Trend -> Confirms long-term uptrend.
       -> Removes short-term noise, trades only in strong trends.

    5. Noise Filter (Volatility Stability)
       short_noise < long_noise -> Short-term volatility < Long-term average.
       -> Volatility is within normal range = High signal reliability.

    6. SMA Exit (Stop Loss)
       Close < SMA -> Trend reversal signal.
       -> Immediate exit to minimize losses.

    Profit Maximization Points:
    - K Value Adjustment: Higher K = Harder entry (Win rate up, Trade count down). Lower K = Easier entry (Trade count up, Loss up).
    - SMA Period: Longer = Higher trend signal accuracy, Shorter = Higher trade frequency.
    - Filter Combination: More filters = Trade count down, Win rate up. Fewer filters = Trade count up, Loss up.

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
                TrendCondition(),  # Formerly TrendFilter, 장기 추세: target > sma_trend
                NoiseCondition(),  # Formerly NoiseFilter, 노이즈 필터: short_noise < long_noise
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
        """Calculate all VBO indicators.

        VBO 전략의 핵심 지표 계산:

        1. 변동성 K값 (noise)
           K = (고가 - 저가) / max(고가, 저가)
           범위: 0 < K < 1
           의미: 일일 변동성의 비율

        2. 목표가(target)
           target = 시가_오늘 + (고가 - 저가)_어제 × K
           = 어제의 범위만큼 위로 올린 오늘의 시가
           의미: 변동성이 어제와 비슷하면 이 정도 올라올 것이라는 예상

        3. SMA(short_noise_period) = 단기 이동평균
           단기 추세 파악 및 퇴출 신호

        4. SMA_Trend(trend_sma_period) = 장기 이동평균
           장기 추세 확인 및 추세필터

        5. Short_Noise = 최근 K값들의 평균
           recent volatility 측정

        6. Long_Noise = 전체 K값들의 평균
           historical volatility baseline
        """
        # Calculate base indicators
        df = add_vbo_indicators(
            df,
            sma_period=self.sma_period,
            trend_sma_period=self.trend_sma_period,
            short_noise_period=self.short_noise_period,
            long_noise_period=self.long_noise_period,
            exclude_current=self.exclude_current,
        )

        # Apply Phase 2 improvements if enabled
        if self.use_improved_noise or self.use_adaptive_k:
            from src.utils.indicators import add_improved_indicators

            df = add_improved_indicators(
                df,
                short_period=self.short_noise_period,
                long_period=self.long_noise_period,
                atr_period=self.atr_period,
                base_k=self.base_k,
            )

            # Override noise and k-value with improved versions if enabled
            if self.use_improved_noise:
                df["short_noise"] = df["short_noise_adaptive"]
                df["long_noise"] = df["long_noise_adaptive"]

            if self.use_adaptive_k:
                # Recalculate target with adaptive K
                df["target"] = df["open"] + df["prev_range"] * df["k_value_adaptive"]

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals using vectorized operations.

        신호 생성의 핵심 로직:

        [진입 신호(entry_signal)] 생성
        조건1: 고가 >= 목표가 (변동성 돌파)
        조건2: 목표가 > SMA (상승추세)
        조건3: 목표가 > SMA_Trend (장기 상승추세)
        조건4: short_noise < long_noise (안정적 변동성)
        → 모든 조건 AND 결합 (엄격한 필터링)

        예시:
        - 고가=$100, 목표가=$98, SMA=$97, SMA_Trend=$96
        - short_noise=0.02, long_noise=0.025
        → 모든 조건 만족 → 매수 신호 발생

        [퇴출 신호(exit_signal)] 생성
        조건: 종가 < SMA (추세 반전)
        → 빠른 손실 차단

        신호의 품질이 수익률 직결:
        - False positives 많으면: 손실 거래 증가
        - False negatives 많으면: 수익 거래 놓침
        - 조건 추가 = 신호 감소 but 품질 향상
        - 조건 제거 = 신호 증가 but 거짓신호 증가

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
                # 고가 >= 목표가: 변동성 돌파 신호
                entry_signal = entry_signal & (df["high"] >= df["target"])
            elif condition.name == "SMABreakout":
                # 목표가 > SMA: 단기 추세 필터 (상승추세 확인)
                entry_signal = entry_signal & (df["target"] > df["sma"])
            elif condition.name == "TrendCondition" or condition.name == "TrendFilter":
                # 목표가 > SMA_Trend: 장기 추세 필터 (강한 상승추세만)
                entry_signal = entry_signal & (df["target"] > df["sma_trend"])
            elif condition.name == "NoiseCondition" or condition.name == "NoiseFilter":
                # short_noise < long_noise: 변동성 안정성 필터
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
