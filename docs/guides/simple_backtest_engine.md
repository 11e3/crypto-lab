# EventDrivenBacktestEngine

간단하고 명확한 Event-driven 백테스트 엔진입니다.

## 특징

### 장점
- ✅ **디버깅 용이**: 날짜별로 순회하며 명확한 로직
- ✅ **범용성**: 모든 전략과 호환
- ✅ **명확한 거래 실행**: Entry/Exit 신호를 직접 처리
- ✅ **Trade 기록 정확**: 모든 거래가 기록됨

### 단점
- ❌ **느린 속도**: 벡터화가 아닌 루프 기반 처리
- ❌ **기본 포지션 사이징**: 단순 균등 배분

## 사용법

### 기본 사용

```python
from datetime import date
from pathlib import Path

from src.backtester.models import BacktestConfig
from src.backtester.engine.event_driven import EventDrivenBacktestEngine
from src.strategies.volatility_breakout.vbo_v1 import VBOV1

# 전략 생성
strategy = VBOV1(
    name="VBOV1",
    ma_short=5,
    btc_ma=10,
    data_dir=Path("data/raw"),
    interval="day",
)

# 데이터 파일
data_files = {
    "KRW-BTC": Path("data/raw/KRW-BTC_day.parquet"),
    "KRW-ETH": Path("data/raw/KRW-ETH_day.parquet"),
}

# 백테스트 설정
config = BacktestConfig(
    initial_capital=10_000_000,
    fee_rate=0.0005,
    slippage_rate=0.0005,
    max_slots=2,
)

# 엔진 생성 및 실행
engine = EventDrivenBacktestEngine(config)
result = engine.run(
    strategy=strategy,
    data_files=data_files,
    start_date=date(2023, 3, 1),
    end_date=None,
)

# 결과 출력
print(f"CAGR: {result.cagr:.2f}%")
print(f"MDD: {result.mdd:.2f}%")
print(f"Trades: {result.total_trades}")
```

## 동작 원리

### 처리 흐름

1. **데이터 로드**
   - 각 자산별 OHLCV 데이터 로드
   - 전략의 `calculate_indicators()` 호출
   - 전략의 `generate_signals()` 호출

2. **날짜별 순회** (Event-driven)
   ```
   for each date:
       ├─ EXIT LOGIC (먼저 처리)
       │  ├─ 청산 시그널 확인
       │  ├─ Trailing stop 확인
       │  ├─ Stop loss 확인
       │  └─ Take profit 확인
       │
       ├─ ENTRY LOGIC
       │  ├─ 진입 시그널 확인
       │  ├─ 슬롯 가용성 확인
       │  └─ 포지션 오픈
       │
       └─ EQUITY 계산
          └─ cash + portfolio_value
   ```

3. **메트릭 계산**
   - CAGR, MDD, Sharpe, Calmar 등
   - 거래 통계 (승률, Profit Factor)

## VectorizedBacktestEngine과 비교

| 항목 | EventDrivenEngine | VectorizedEngine |
|------|-------------------|------------------|
| **속도** | 느림 (루프 기반) | 빠름 (벡터화) |
| **호환성** | 모든 전략 ✅ | 모든 전략 ✅ |
| **디버깅** | 쉬움 ✅ | 어려움 |
| **거래 실행** | 명확함 ✅ | 복잡함 |
| **Trade 기록** | 정확함 ✅ | 불완전할 수 있음 |
| **권장 용도** | 전략 개발/테스트 | 대량 데이터 분석 |

## 예제

### 1. 단일 자산 백테스트

```python
data_files = {"KRW-BTC": Path("data/raw/KRW-BTC_day.parquet")}
config = BacktestConfig(initial_capital=10_000_000, max_slots=1)

engine = EventDrivenBacktestEngine(config)
result = engine.run(strategy, data_files)

print(f"CAGR: {result.cagr:.2f}%")
print(f"MDD: {result.mdd:.2f}%")
```

### 2. 다중 자산 포트폴리오

```python
data_files = {
    "KRW-BTC": Path("data/raw/KRW-BTC_day.parquet"),
    "KRW-ETH": Path("data/raw/KRW-ETH_day.parquet"),
    "KRW-XRP": Path("data/raw/KRW-XRP_day.parquet"),
}

config = BacktestConfig(
    initial_capital=10_000_000,
    max_slots=3,
)

engine = EventDrivenBacktestEngine(config)
result = engine.run(strategy, data_files)

for trade in result.trades[:10]:
    print(f"{trade.ticker} {trade.entry_date} -> {trade.exit_date}: {trade.pnl_pct:+.2f}%")
```

## 팁

### 디버깅

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# EventDrivenBacktestEngine은 상세 로그 출력
```

### 성능 개선

- 날짜 범위 제한: `start_date`, `end_date` 활용
- 적은 자산으로 먼저 테스트
- 대량 분석은 VectorizedBacktestEngine 권장

## 관련 문서

- [Backtester 모듈 구조](backtester_modules.md)
- [전략 가이드](strategies.md)
