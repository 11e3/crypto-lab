# Strategy Layer API

전략 레이어는 거래 전략을 정의하고 신호를 생성하는 핵심 모듈입니다.

## 모듈 개요

- `src.strategies.base`: 전략 기본 클래스 및 조건 인터페이스
- `src.strategies.volatility_breakout`: 변동성 돌파 전략 구현

## 핵심 클래스

### `Strategy` (추상 클래스)

모든 거래 전략의 기본 클래스입니다.

#### 메서드

##### `required_indicators() -> list[str]`

전략에 필요한 지표 이름 목록을 반환합니다.

**반환값:**
- `list[str]`: 지표 이름 목록 (예: `["sma", "target", "noise"]`)

**예시:**
```python
class MyStrategy(Strategy):
    def required_indicators(self) -> list[str]:
        return ["sma", "rsi", "macd"]
```

##### `calculate_indicators(df: pd.DataFrame) -> pd.DataFrame`

전략에 필요한 모든 지표를 계산합니다.

**매개변수:**
- `df`: OHLCV 데이터가 포함된 DataFrame

**반환값:**
- `pd.DataFrame`: 지표 컬럼이 추가된 DataFrame

**예시:**
```python
def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    df["sma"] = df["close"].rolling(window=20).mean()
    df["rsi"] = calculate_rsi(df["close"], period=14)
    return df
```

##### `generate_signals(df: pd.DataFrame) -> pd.DataFrame`

벡터화된 연산을 사용하여 진입/청산 신호를 생성합니다.

**매개변수:**
- `df`: OHLCV 및 지표가 포함된 DataFrame

**반환값:**
- `pd.DataFrame`: `entry_signal` 및 `exit_signal` 컬럼이 추가된 DataFrame

**예시:**
```python
def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["entry_signal"] = (df["close"] > df["sma"]) & (df["rsi"] < 70)
    df["exit_signal"] = df["close"] < df["sma"]
    return df
```

##### `check_entry(current: OHLCV, history: pd.DataFrame, indicators: dict[str, float]) -> bool`

진입 조건이 충족되는지 확인합니다.

**매개변수:**
- `current`: 현재 봉의 OHLCV 데이터
- `history`: 과거 데이터 DataFrame
- `indicators`: 현재 봉의 사전 계산된 지표 값

**반환값:**
- `bool`: 포지션 진입 여부

##### `check_exit(current: OHLCV, history: pd.DataFrame, indicators: dict[str, float], position: Position) -> bool`

청산 조건이 충족되는지 확인합니다.

**매개변수:**
- `current`: 현재 봉의 OHLCV 데이터
- `history`: 과거 데이터 DataFrame
- `indicators`: 현재 봉의 사전 계산된 지표 값
- `position`: 현재 포지션

**반환값:**
- `bool`: 포지션 청산 여부

##### `add_entry_condition(condition: Condition) -> Strategy`

진입 조건을 추가합니다 (체이닝 지원).

**매개변수:**
- `condition`: 추가할 조건

**반환값:**
- `Strategy`: 자기 자신 (체이닝용)

**예시:**
```python
strategy = MyStrategy()
strategy.add_entry_condition(BreakoutCondition())
    .add_entry_condition(TrendCondition())
```

##### `add_exit_condition(condition: Condition) -> Strategy`

청산 조건을 추가합니다 (체이닝 지원).

##### `remove_entry_condition(condition: Condition) -> Strategy`

진입 조건을 제거합니다.

##### `remove_exit_condition(condition: Condition) -> Strategy`

청산 조건을 제거합니다.

### `Condition` (추상 클래스)

진입/청산 조건의 기본 클래스입니다.

#### 메서드

##### `evaluate(current: OHLCV, history: pd.DataFrame, indicators: dict[str, float]) -> bool`

시장 데이터에 대해 조건을 평가합니다.

**매개변수:**
- `current`: 현재 봉의 OHLCV 데이터
- `history`: OHLCV 및 지표가 포함된 과거 DataFrame
- `indicators`: 현재 봉의 사전 계산된 지표 값

**반환값:**
- `bool`: 조건 충족 여부

**예시:**
```python
class PriceAboveSMACondition(Condition):
    def evaluate(self, current, history, indicators):
        return current.close > indicators.get("sma", 0)
```

### `CompositeCondition`

여러 조건을 AND/OR 논리로 결합합니다.

#### 생성자

```python
CompositeCondition(
    conditions: list[Condition],
    operator: str = "AND",
    name: str | None = None
)
```

**매개변수:**
- `conditions`: 결합할 조건 목록
- `operator`: "AND" 또는 "OR"
- `name`: 사람이 읽을 수 있는 이름

#### 메서드

##### `add(condition: Condition) -> CompositeCondition`

조건을 추가하고 체이닝을 위해 자기 자신을 반환합니다.

##### `remove(condition: Condition) -> CompositeCondition`

조건을 제거하고 체이닝을 위해 자기 자신을 반환합니다.

### `VBOV1`

변동성 돌파 전략의 기본 구현입니다.

#### 생성자

```python
VBOV1(
    name: str = "VBOV1",
    sma_period: int = 4,
    trend_sma_period: int = 8,
    short_noise_period: int = 4,
    long_noise_period: int = 8,
    entry_conditions: list[Condition] | None = None,
    exit_conditions: list[Condition] | None = None,
    use_default_conditions: bool = True,
    exclude_current: bool = False
)
```

**매개변수:**
- `name`: 전략 이름
- `sma_period`: 청산 SMA 기간
- `trend_sma_period`: 트렌드 SMA 기간
- `short_noise_period`: K 값 계산 기간
- `long_noise_period`: 노이즈 기준선 기간
- `entry_conditions`: 커스텀 진입 조건 (선택)
- `exit_conditions`: 커스텀 청산 조건 (선택)
- `use_default_conditions`: 기본 조건 사용 여부
- `exclude_current`: 현재 봉을 계산에서 제외할지 여부

#### 기본 조건

**진입 조건:**
- `BreakoutCondition`: 가격이 타겟 가격을 돌파
- `SMABreakoutCondition`: 타겟이 SMA 위에 있음
- `TrendCondition`: 타겟이 트렌드 SMA 위에 있음
- `NoiseCondition`: 단기 노이즈 < 장기 노이즈

**청산 조건:**
- `PriceBelowSMACondition`: 종가가 SMA 아래로 떨어짐

## 데이터 클래스

### `Signal`

거래 신호를 나타냅니다.

```python
@dataclass
class Signal:
    signal_type: SignalType  # BUY, SELL, HOLD
    ticker: str
    price: float
    date: date
    strength: float = 1.0  # 신호 강도 (0-1)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### `Position`

거래 포지션을 나타냅니다.

```python
@dataclass
class Position:
    ticker: str
    amount: float
    entry_price: float
    entry_date: date
```

### `OHLCV`

단일 봉의 OHLCV 데이터를 나타냅니다.

```python
@dataclass
class OHLCV:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
```

**속성:**
- `range`: 가격 범위 (high - low)
- `body`: 캔들 몸통 크기 (abs(close - open))

## 사용 예제

### 커스텀 전략 생성

```python
from src.strategies.base import Strategy, Condition, OHLCV
from src.strategies.volatility_breakout.conditions import BreakoutCondition
import pandas as pd

class MyCustomStrategy(Strategy):
    def required_indicators(self) -> list[str]:
        return ["sma", "target"]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sma"] = df["close"].rolling(window=20).mean()
        df["target"] = df["open"] + (df["high"] - df["low"]) * 0.5
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["entry_signal"] = df["high"] >= df["target"]
        df["exit_signal"] = df["close"] < df["sma"]
        return df

# 전략 사용
strategy = MyCustomStrategy()
strategy.add_entry_condition(BreakoutCondition())
```

### 조건 조합

```python
from src.strategies.base import CompositeCondition
from src.strategies.volatility_breakout.conditions import (
    BreakoutCondition,
    TrendCondition,
    NoiseCondition
)

# AND 조건
entry_conditions = CompositeCondition(
    conditions=[
        BreakoutCondition(),
        TrendCondition(),
        NoiseCondition()
    ],
    operator="AND"
)

# OR 조건
exit_conditions = CompositeCondition(
    conditions=[...],
    operator="OR"
)
```

## 관련 문서

- [전략 커스터마이징 가이드](../guides/strategy_customization.md)
- [아키텍처 문서](../architecture.md)
