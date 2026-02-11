# 전략 커스터마이징 가이드

## 전략 구조 이해

모든 전략은 `Strategy` 기본 클래스를 상속받아 구현됩니다:

```python
from src.strategies.base import Strategy

class MyCustomStrategy(Strategy):
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 계산"""
        pass
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """신호 생성"""
        pass
    
    def check_entry(self, df: pd.DataFrame, current_idx: int) -> bool:
        """진입 조건 확인"""
        pass
    
    def check_exit(self, df: pd.DataFrame, current_idx: int) -> bool:
        """청산 조건 확인"""
        pass
```

## VBOV1 전략 분석

### 지표 계산

```python
def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    # SMA 계산
    df["sma"] = sma(df["close"], period=self.sma_period)
    
    # 트렌드 SMA 계산
    df["sma_trend"] = sma(df["close"], period=self.trend_sma_period)
    
    # 노이즈 비율 계산
    df["noise_ratio"] = noise_ratio_sma(
        df, 
        short_period=self.short_noise_period,
        long_period=self.long_noise_period
    )
    
    # 타겟 가격 계산
    df["target"] = df["close"] * (1 + df["noise_ratio"] * self.k)
    
    return df
```

### 진입 조건

```python
def check_entry(self, df: pd.DataFrame, current_idx: int) -> bool:
    current = df.iloc[current_idx]
    prev = df.iloc[current_idx - 1]
    
    # 조건 1: 현재가가 타겟 가격 이상
    if current["close"] < current["target"]:
        return False
    
    # 조건 2: 트렌드 SMA 위에 있어야 함
    if current["close"] < current["sma_trend"]:
        return False
    
    # 조건 3: 노이즈 비율 조건
    if current["noise_ratio"] >= current["long_noise"]:
        return False
    
    # 조건 4: 이전에 진입 신호가 없었어야 함
    if prev.get("entry_signal", False):
        return False
    
    return True
```

### 청산 조건

```python
def check_exit(self, df: pd.DataFrame, current_idx: int) -> bool:
    current = df.iloc[current_idx]
    
    # 현재가가 SMA 아래로 떨어지면 청산
    return current["close"] < current["sma"]
```

## 커스텀 전략 만들기

### 예제: 단순 이동평균 크로스오버 전략

```python
from src.strategies.base import Strategy
import pandas as pd
from src.utils.indicators import sma

class SMACrossStrategy(Strategy):
    """단순 이동평균 크로스오버 전략"""
    
    def __init__(
        self,
        name: str = "SMA Cross",
        fast_period: int = 5,
        slow_period: int = 20,
    ) -> None:
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 계산"""
        df = df.copy()
        df["sma_fast"] = sma(df["close"], period=self.fast_period)
        df["sma_slow"] = sma(df["close"], period=self.slow_period)
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """신호 생성"""
        df = df.copy()
        
        # 골든 크로스: 빠른 SMA가 느린 SMA를 상향 돌파
        df["entry_signal"] = (
            (df["sma_fast"] > df["sma_slow"]) &
            (df["sma_fast"].shift(1) <= df["sma_slow"].shift(1))
        )
        
        # 데드 크로스: 빠른 SMA가 느린 SMA를 하향 돌파
        df["exit_signal"] = (
            (df["sma_fast"] < df["sma_slow"]) &
            (df["sma_fast"].shift(1) >= df["sma_slow"].shift(1))
        )
        
        return df
    
    def check_entry(self, df: pd.DataFrame, current_idx: int) -> bool:
        """진입 조건 확인"""
        return bool(df.iloc[current_idx]["entry_signal"])
    
    def check_exit(self, df: pd.DataFrame, current_idx: int) -> bool:
        """청산 조건 확인"""
        return bool(df.iloc[current_idx]["exit_signal"])
```

### 사용 예제

```python
from src.backtester import run_backtest, BacktestConfig

strategy = SMACrossStrategy(fast_period=5, slow_period=20)
config = BacktestConfig(initial_capital=1_000_000.0)

results = run_backtest(
    tickers=["KRW-BTC"],
    strategy=strategy,
    config=config,
)
```

## Conditions와 Filters 활용

기존 Conditions와 Filters를 조합하여 새로운 전략을 만들 수 있습니다:

```python
from src.strategies.volatility_breakout.conditions import (
    PriceAboveTargetCondition,
    TrendFilterCondition,
)
from src.strategies.volatility_breakout.filters import (
    VolumeFilter,
    NoiseRatioFilter,
)

class CustomVBOStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__("CustomVBO")
        self.conditions = [
            PriceAboveTargetCondition(),
            TrendFilterCondition(),
        ]
        self.filters = [
            VolumeFilter(min_volume_ratio=1.2),
            NoiseRatioFilter(max_noise_ratio=0.5),
        ]
    
    def check_entry(self, df: pd.DataFrame, current_idx: int) -> bool:
        """조건 및 필터 조합"""
        # 모든 조건 만족 확인
        for condition in self.conditions:
            if not condition.check(df, current_idx):
                return False
        
        # 모든 필터 통과 확인
        for filter_obj in self.filters:
            if not filter_obj.filter(df, current_idx):
                return False
        
        return True
```

## 베스트 프랙티스

### 1. 지표 계산 최적화

- 필요한 지표만 계산
- 캐싱 활용 (`use_cache=True`)
- 벡터화된 연산 사용

### 2. 신호 생성

- 명확한 진입/청산 조건 정의
- 백테스트와 실시간 거래에서 동일한 로직 사용
- 엣지 케이스 처리

### 3. 테스트

- 단위 테스트 작성
- 다양한 시장 조건에서 백테스트
- 실시간 거래 전에 페이퍼 트레이딩

### 4. 문서화

- 전략 로직 설명
- 파라미터 의미 명시
- 사용 예제 제공

## 고급 주제

### 동적 파라미터 조정

```python
class AdaptiveStrategy(Strategy):
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # 시장 변동성에 따라 파라미터 조정
        volatility = df["close"].pct_change().std()
        
        if volatility > 0.05:  # 고변동성
            self.sma_period = 10
        else:  # 저변동성
            self.sma_period = 5
        
        return super().calculate_indicators(df)
```

### 멀티 타임프레임 분석

```python
class MultiTimeframeStrategy(Strategy):
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # 일봉 데이터
        df["sma_daily"] = sma(df["close"], period=20)
        
        # 4시간봉 데이터 (별도 로드 필요)
        # df_4h = load_4h_data()
        # df["sma_4h"] = sma(df_4h["close"], period=10)
        
        return df
```

## 참고 자료

- [Strategy Base Class API](../api/strategies.md)
- [Indicators Reference](../api/indicators.md)
- [Backtest Engine](../api/backtester.md)
