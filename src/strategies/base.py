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
    - 진입 조건(entry_conditions): 매수 신호 생성 위한 기술적 조건들의 AND 조합
    - 퇴출 조건(exit_conditions): 매도 신호 생성 위한 조건들의 AND 조합

    VBO(Volatility Breakout) 전략의 핵심 아이디어:
    1. 변동성 돌파: 전일 변동폭(고가-저가)의 k배만큼 상승시 매수
    2. 추세 필터: SMA 위에서만 매수 (상승추세 확인)
    3. 익일 매도: 다음날 종가에 자동 매도 (오버나이트 리스크 관리)

    수익률 계산의 정확성이 전략 성능의 핵심:
    - 슬리피지/수수료 포함 시 실제 수익률 반영
    - 진입가: 목표가(target) = 시가 + (전일 변동폭 * k)
    - 청산가: 다음날 종가

    Subclasses must implement:
    - name property: 전략 식별자 (예: "VBO_K0.5", "MomentumSMA20")
    - required_indicators(): 전략에 필요한 지표 목록
    - calculate_indicators(): 지표 계산 로직

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

        이 메서드가 반환하는 지표 목록은 calculate_indicators()에서 생성해야 함.
        예: ["sma", "target", "noise"] → df["sma"], df["target"], df["noise"] 필요

        Returns:
            List of indicator names required for signal generation
        """
        pass

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required indicators for the strategy.

        이 메서드에서 계산된 지표들이 수익률 계산에 직결됨.
        - 정확한 지표 계산 = 수익률 최대화
        - 잘못된 지표 = 신호 왜곡 = 손실 발생

        각 지표는 다음을 만족해야 함:
        1. 정확한 계산식: SMA(n)은 정확한 이동평균, ATR(n)은 정확한 ATR
        2. 적절한 lookback: 지표 필요 기간(예: SMA(20)은 최소 20일)
        3. NaN 처리: 부족한 데이터는 NaN으로, 나중에 forward fill이나 dropna 처리됨

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with added indicator columns (sma, target, momentum, rsi, etc.)

        Note:
            - DataFrame은 반드시 복사본을 반환해야 원본 데이터 보존
            - 지표 열 이름은 required_indicators()의 반환값과 정확히 일치해야 함
            - 지표 계산이 수익률 계산의 기반이므로 정확성이 가장 중요
        """
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals using vectorized operations.

        신호 생성은 수익률의 핵심:
        - entry_signal: 매수 신호 (정확한 타이밍 = 더 좋은 진입가)
        - exit_signal: 매도 신호 (리스크 관리 = 손실 최소화)

        기본 VBO 신호 로직:
        1. 진입(entry_signal): 고가>=목표가 AND 목표가>SMA AND 필터조건
           → 하한선(목표가)을 돌파할 때만 진입 (노이즈 제거)
        2. 퇴출(exit_signal): 종가<SMA
           → 추세 전환 시점에 자동 퇴출 (손실 제한)

        Override this method for custom vectorized signal logic.
        Default implementation uses standard VBO signals.

        Args:
            df: DataFrame with OHLCV and indicators (sma, target, noise 등)

        Returns:
            DataFrame with added 'entry_signal' and 'exit_signal' boolean columns

        Note:
            신호의 정확도가 전체 전략 수익성을 결정함:
            - 거래 빈도: 신호 수 = 거래 횟수 = 승률 × 거래당평균수익
            - 신호 품질: False Signal 많으면 손실, 정확한 신호면 수익 증가
        """
        df = df.copy()

        # Default: use indicator-based signals
        # 진입 신호: 변동성 돌파(고가 >= 목표가)와 추세필터(목표가 > SMA) 결합
        # → 상한선(저항) 돌파 시 거래량 급증 시점에 진입
        entry_breakout = df["high"] >= df["target"]
        # 추세 필터: 목표가가 SMA를 위에 있어야 상승추세
        entry_sma = df["target"] > df["sma"]

        # 마켓 필터: 노이즈 레벨이 낮을 때만 거래 (변동성 필터)
        has_trend_filter = "sma_trend" in df.columns
        has_noise_filter = "short_noise" in df.columns and "long_noise" in df.columns

        # 초기 진입 신호 = 고가돌파 AND 추세필터
        entry_signal = entry_breakout & entry_sma

        # 추가 필터 적용: sma_trend는 장기 추세 확인
        if has_trend_filter:
            entry_signal = entry_signal & (df["target"] > df["sma_trend"])

        # 노이즈 필터: short_noise < long_noise이면 변동성이 낮음 (안정적)
        # → 거짓신호 많은 고변동성 구간 제거 (수익성 개선)
        if has_noise_filter:
            entry_signal = entry_signal & (df["short_noise"] < df["long_noise"])

        # 퇴출 신호: 종가가 SMA 아래로 내려오면 추세 반전으로 판단하고 즉시 매도
        # → 손실 방지 (리스크 관리), 추세 전환 시 빠른 탈출
        exit_signal = df["close"] < df["sma"]

        if "entry_signal" not in df.columns:
            df["entry_signal"] = entry_signal
        if "exit_signal" not in df.columns:
            df["exit_signal"] = exit_signal

        return df

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
