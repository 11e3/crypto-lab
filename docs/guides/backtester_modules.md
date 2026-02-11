# Backtester 모듈 구조

백테스팅 엔진의 모듈화된 구조입니다.

## 파일 구조

```
src/backtester/
├── __init__.py           # 패키지 exports
├── models.py             # 데이터 모델 (BacktestConfig, Trade, BacktestResult)
├── metrics.py            # 성능 메트릭 계산
├── engine/               # 백테스트 엔진
│   ├── vectorized.py     # VectorizedBacktestEngine (고속)
│   ├── event_driven.py   # EventDrivenBacktestEngine (범용)
│   ├── signal_processor.py # 진입/퇴출 가격 (exit_price_base 지원)
│   ├── trade_simulator.py  # 벡터화 거래 시뮬레이션
│   ├── event_exec.py     # 이벤트 기반 거래 실행
│   └── trade_costs.py    # 수수료/슬리피지 계산
├── analysis/             # CPCV, Bootstrap, Monte Carlo
├── wfa/                  # Walk-Forward Analysis
├── report.py             # 리포트 생성
└── optimization.py       # 파라미터 최적화
```

## 모듈 설명

### models.py
**공통 데이터 구조**

- `BacktestConfig`: 백테스트 설정
  - initial_capital, fee_rate, slippage_rate
  - max_slots, position_sizing
  - stop_loss_pct, take_profit_pct, trailing_stop_pct

- `Trade`: 거래 기록
  - ticker, entry_date, entry_price
  - exit_date, exit_price, amount
  - pnl, pnl_pct, exit_reason

- `BacktestResult`: 백테스트 결과
  - 성능 지표 (CAGR, MDD, Sharpe, Calmar)
  - 거래 통계 (total_trades, win_rate, profit_factor)
  - 시계열 데이터 (equity_curve, dates)

### metrics.py
**성능 메트릭 계산**

- `calculate_metrics()`: 백테스트 결과에서 모든 메트릭 계산
  - CAGR, MDD, Sharpe Ratio, Calmar Ratio
  - 거래 통계 (승률, Profit Factor)
  - 포트폴리오 리스크 메트릭

### engine/vectorized.py
**VectorizedBacktestEngine**

- 고속 벡터화 처리 (pandas/numpy)
- 모든 Strategy 서브클래스와 호환
- 대량 데이터 분석 및 파라미터 최적화용

### engine/event_driven.py
**EventDrivenBacktestEngine**

- Event-driven 방식 (날짜별 순회)
- 모든 전략과 호환
- 명확한 로직, 디버깅 용이
- 전략 개발/테스트용

## 사용 예제

### 기본 사용

```python
from src.backtester.models import BacktestConfig
from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.strategies.volatility_breakout.vbo_v1 import VBOV1

# 설정
config = BacktestConfig(
    initial_capital=10_000_000,
    fee_rate=0.0005,
    max_slots=3,
)

# 전략
strategy = VBOV1(
    name="VBOV1",
    ma_short=5,
    btc_ma=10,
    data_dir=DATA_DIR,
    interval="day",
)

# 백테스트
engine = VectorizedBacktestEngine(config)
result = engine.run(strategy, data_files)

# 결과
print(f"CAGR: {result.cagr:.2f}%")
print(f"MDD: {result.mdd:.2f}%")
print(f"Trades: {result.total_trades}")
```

### 메트릭 직접 계산

```python
from src.backtester.metrics import calculate_metrics

result = calculate_metrics(
    equity_curve=equity_curve,
    dates=dates,
    trades=trades_list,
    config=config,
    strategy_name="MyStrategy",
)
```

### 모델 재사용

```python
from src.backtester.models import Trade

trade = Trade(
    ticker="KRW-BTC",
    entry_date=date(2023, 1, 1),
    entry_price=30_000_000,
    exit_date=date(2023, 1, 10),
    exit_price=35_000_000,
    amount=0.1,
    pnl=500_000,
    pnl_pct=16.67,
    exit_reason="trailing_stop",
)
```

## 엔진 선택 가이드

| 상황 | 권장 엔진 |
|------|----------|
| 파라미터 최적화 | VectorizedBacktestEngine |
| 대량 데이터 (수년 × 수십 자산) | VectorizedBacktestEngine |
| 전략 개발/디버깅 | EventDrivenBacktestEngine |
| 명확한 Trade 기록 필요 | EventDrivenBacktestEngine |

## 확장 가이드

### 새로운 전략 추가

1. `Strategy` 서브클래스 생성
2. `generate_signals()` 구현
3. `StrategyRegistry`가 자동 검색 → 대시보드 UI 자동 생성

### 새로운 메트릭 추가

`metrics.py`의 `calculate_metrics()` 함수에 추가:

```python
def calculate_metrics(...) -> BacktestResult:
    result = BacktestResult(...)
    result.sortino_ratio = calculate_sortino(...)
    return result
```

## 관련 문서

- [SimpleBacktestEngine 가이드](simple_backtest_engine.md)
- [전략 가이드](strategies.md)
