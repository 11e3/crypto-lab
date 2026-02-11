# 전략 가이드

이 문서는 crypto-lab에서 제공하는 거래 전략의 상세한 설명을 제공합니다.

---

## 변동성 돌파 전략 (VBOV1)

### 개요

**VBOV1** 전략은 변동성을 활용한 추세 추종 전략입니다. 전일의 변동 범위를 기반으로 목표 가격을 설정하고, 가격이 목표를 돌파할 때 진입합니다.

### 작동 원리

1. **목표 가격 계산**
   - 전일 고가와 저가의 범위를 계산
   - 고정 K=0.5를 곱하여 변동성 조정
   - 목표 가격 = 당일 시가 + (전일 범위 × K)

2. **진입 조건**
   - 가격이 목표 가격을 돌파 (High ≥ Target)
   - 목표 가격이 SMA 위에 있음 (단기 이동평균 필터)
   - BTC MA20 필터: BTC 종가가 20일 이동평균 위에 있을 때만 진입

3. **청산 조건**
   - 전일 종가가 전일 SMA 아래로 떨어짐 (shift(1)로 look-ahead bias 방지)
   - 시가(open)에서 매도 (exit_price_base 컨벤션)

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `ma_short` (lookback) | 5 | 청산용 SMA 기간 |
| `btc_ma` | 10 (lookback × multiplier) | BTC 시장 필터 MA 기간 |
| `noise_ratio` | 0.5 | 고정 K 값 |

### 사용 예제

```python
from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.backtester.models import BacktestConfig
from src.strategies.volatility_breakout.vbo_v1 import VBOV1

# 전략 생성
strategy = VBOV1(
    name="VBOV1",
    ma_short=5,
    btc_ma=10,
    data_dir=Path("data/raw"),
    interval="day",
)

# 백테스트 설정
config = BacktestConfig(
    initial_capital=10_000_000,
    fee_rate=0.0005,
    slippage_rate=0.0005,
    max_slots=3,
)

# 백테스트 실행
engine = VectorizedBacktestEngine(config)
result = engine.run(
    strategy=strategy,
    data_files=data_files,
)
```

### 대시보드 사용

Streamlit 대시보드에서 VBO 전략을 선택하면 lookback/multiplier 파라미터를 슬라이더로 조절할 수 있습니다.

```bash
streamlit run src/web/app.py
```

### 특징

**장점:**
- 변동성이 큰 시장에서 효과적
- 명확한 진입/청산 신호
- BTC MA 필터로 하락장 회피
- 시가 매도로 실행 가능성 높음

**단점:**
- 횡보장에서 손실 가능
- 노이즈가 많은 시장에서 성과 저하
- 파라미터 튜닝 필요

---

## 커스텀 전략 추가

`Strategy` 서브클래스를 작성하면 대시보드에 자동으로 노출됩니다:

```python
from src.strategies.base import Strategy, Signal, SignalType
import pandas as pd

class MyStrategy(Strategy):
    """Custom strategy implementation."""

    def __init__(self, my_param: int = 10):
        super().__init__(name="MyStrategy")
        self.my_param = my_param

    def generate_signals(self, ohlcv_data: pd.DataFrame, ticker: str) -> list[Signal]:
        signals = []
        # Your signal generation logic here
        return signals
```

`StrategyRegistry`가 `__init__` 시그니처에서 파라미터를 자동 추출하여 대시보드 UI를 생성합니다.

---

## 관련 문서

- [시스템 아키텍처](../architecture.md)
- [백테스터 모듈 구조](backtester_modules.md)
