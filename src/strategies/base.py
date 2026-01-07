"""
Base classes for strategy abstraction.

Provides modular interfaces for building trading strategies with
composable conditions and filters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from typing import Any

import pandas as pd


class SignalType(Enum):
    """Trading signal types."""

    BUY = auto()
    SELL = auto()
    HOLD = auto()


@dataclass
class Signal:
    """Trading signal with metadata."""

    signal_type: SignalType
    ticker: str
    price: float
    date: date
    strength: float = 1.0  # Signal strength/confidence (0-1)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Represents a trading position."""

    ticker: str
    amount: float
    entry_price: float
    entry_date: date


@dataclass
class OHLCV:
    """Single OHLCV bar data."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range(self) -> float:
        """Price range (high - low)."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Candle body size."""
        return abs(self.close - self.open)


class Condition(ABC):
    """
    Abstract base class for entry/exit conditions.

    Conditions are atomic logic units that evaluate market data
    and return True/False for signal generation.
    """

    def __init__(self, name: str | None = None) -> None:
        """
        Initialize condition.

        Args:
            name: Human-readable condition name
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """
        Evaluate condition against market data.

        Args:
            current: Current bar OHLCV data
            history: Historical DataFrame with OHLCV + indicators
            indicators: Pre-calculated indicator values for current bar

        Returns:
            True if condition is met
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# Alias for backward compatibility
# Filters are now treated as Conditions in the unified architecture
Filter = Condition


class CompositeCondition(Condition):
    """Combines multiple conditions with AND/OR logic."""

    def __init__(
        self,
        conditions: list[Condition],
        operator: str = "AND",
        name: str | None = None,
    ) -> None:
        """
        Initialize composite condition.

        Args:
            conditions: List of conditions to combine
            operator: "AND" or "OR"
            name: Human-readable name
        """
        super().__init__(name)
        self.conditions = conditions
        self.operator = operator.upper()

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Evaluate all conditions with specified operator."""
        if not self.conditions:
            return True

        results = [c.evaluate(current, history, indicators) for c in self.conditions]

        if self.operator == "AND":
            return all(results)
        elif self.operator == "OR":
            return any(results)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def add(self, condition: Condition) -> "CompositeCondition":
        """Add a condition and return self for chaining."""
        self.conditions.append(condition)
        return self

    def remove(self, condition: Condition) -> "CompositeCondition":
        """Remove a condition and return self for chaining."""
        self.conditions.remove(condition)
        return self


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies combine conditions to generate trading signals based on:
    - 진입 조건(entry_conditions): 매수 신호 생성 위한 기술적 조건들의 AND 조합
    - 퇴출 조건(exit_conditions): 매도 신호 생성 위한 조건들의 AND 조합
    
    모든 조건(마켓 필터 포함)은 entry/exit conditions로 처리됨.
    
    수익 메커니즘:
    1. entry_conditions 모두 만족 → BUY 신호 발생 → 진입가격 = target(VBO) or close
    2. exit_conditions 만족 → SELL 신호 발생 → 퇴출가격 = close price
    3. 수익률(PnL%) = ((exit_price - entry_price) / entry_price - fee) * 100
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
            name: Strategy name (defaults to class name)
            entry_conditions: 진입 신호 생성 조건들 (기술적 지표 기반, AND 결합)
                            예: [고가>목표가, 목표가>SMA(20), 노이즈필터]
            exit_conditions: 퇴출 신호 생성 조건들 (AND 결합)
                            예: [종가<SMA(20)]
                            
        Note:
            - 조건들은 AND 연산으로 결합됨 (모두 만족해야 신호 발생)
            - entry_conditions의 강도가 수익성에 직결: 엄격할수록 거래수 감소, 승률 상승
            - exit_conditions의 엄격도가 리스크 관리에 영향: 느슨할수록 손실 확대 가능
        """
        self.name = name or self.__class__.__name__
        # entry_conditions: AND로 결합된 진입 조건들
        # 모든 조건이 동시에 만족되어야만 BUY 신호 발생
        self.entry_conditions = CompositeCondition(
            entry_conditions or [], operator="AND", name="EntryConditions"
        )
        # exit_conditions: AND로 결합된 퇴출 조건들
        # 모든 조건이 동시에 만족되어야만 SELL 신호 발생
        self.exit_conditions = CompositeCondition(
            exit_conditions or [], operator="AND", name="ExitConditions"
        )

    @abstractmethod
    def required_indicators(self) -> list[str]:
        """
        Return list of required indicator names for this strategy.
        
        수익 계산에 필수적인 지표들의 이름 반환.
        이 지표들이 calculate_indicators()에서 계산되어야 함.
        
        예시:
        - ['sma', 'target', 'short_noise', 'long_noise'] (VBO 전략)
        - ['momentum', 'rsi'] (모멘텀 전략)
        - ['atr', 'high_20', 'low_20'] (변동성 돌파 전략)

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
