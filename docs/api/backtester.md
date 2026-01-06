# Backtester API

백테스팅 엔진은 과거 데이터를 사용하여 거래 전략의 성능을 시뮬레이션합니다.

## 모듈 개요

- `src.backtester.engine`: 벡터화된 백테스팅 엔진
- `src.backtester.report`: 성능 리포트 생성

## 핵심 클래스

### `VectorizedBacktestEngine`

pandas/numpy를 사용한 고성능 벡터화 백테스팅 엔진입니다.

#### 생성자

```python
VectorizedBacktestEngine(config: BacktestConfig | None = None)
```

**매개변수:**
- `config`: 백테스팅 설정 (기본값: `BacktestConfig()`)

#### 메서드

##### `load_data(filepath: Path) -> pd.DataFrame`

Parquet 파일에서 OHLCV 데이터를 로드합니다.

**매개변수:**
- `filepath`: Parquet 파일 경로

**반환값:**
- `pd.DataFrame`: OHLCV 데이터가 포함된 DataFrame

**예외:**
- `FileNotFoundError`: 파일이 존재하지 않는 경우
- `ValueError`: 파일이 손상되었거나 유효하지 않은 경우

##### `run_backtest(tickers: list[str], strategy: Strategy, start_date: date | None = None, end_date: date | None = None) -> BacktestResult`

백테스트를 실행합니다.

**매개변수:**
- `tickers`: 거래할 티커 목록
- `strategy`: 사용할 거래 전략
- `start_date`: 시작 날짜 (선택)
- `end_date`: 종료 날짜 (선택)

**반환값:**
- `BacktestResult`: 백테스트 결과

**예시:**
```python
from src.backtester import VectorizedBacktestEngine, BacktestConfig
from src.strategies.volatility_breakout import VanillaVBO

engine = VectorizedBacktestEngine(BacktestConfig())
strategy = VanillaVBO()
result = engine.run_backtest(
    tickers=["KRW-BTC", "KRW-ETH"],
    strategy=strategy
)
print(result.summary())
```

### `BacktestConfig`

백테스팅 설정을 나타내는 데이터 클래스입니다.

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0  # 초기 자본
    fee_rate: float = 0.0005  # 수수료율 (0.05%)
    slippage_rate: float = 0.0005  # 슬리피지율 (0.05%)
    max_slots: int = 4  # 최대 보유 종목 수
    position_sizing: str = "equal"  # 포지션 크기 조정 방식
    use_cache: bool = True  # 지표 계산 캐싱 사용 여부
```

**속성:**
- `initial_capital`: 초기 자본금
- `fee_rate`: 거래 수수료율
- `slippage_rate`: 슬리피지율
- `max_slots`: 동시에 보유할 수 있는 최대 종목 수
- `position_sizing`: 포지션 크기 조정 방식 ("equal" 또는 "custom")
- `use_cache`: 지표 계산 캐싱 사용 여부

### `BacktestResult`

백테스트 결과를 나타내는 데이터 클래스입니다.

```python
@dataclass
class BacktestResult:
    # 성능 지표
    total_return: float  # 총 수익률 (%)
    cagr: float  # 연평균 복리 수익률 (%)
    mdd: float  # 최대 낙폭 (%)
    calmar_ratio: float  # 칼마 비율
    sharpe_ratio: float  # 샤프 비율
    win_rate: float  # 승률 (%)
    profit_factor: float  # 수익 팩터
    
    # 거래 통계
    total_trades: int  # 총 거래 횟수
    winning_trades: int  # 승리 거래 횟수
    losing_trades: int  # 손실 거래 횟수
    avg_trade_return: float  # 평균 거래 수익률
    
    # 시계열 데이터
    equity_curve: np.ndarray  # 자산 곡선
    dates: np.ndarray  # 날짜 배열
    trades: list[Trade]  # 거래 목록
    
    # 추가 정보
    config: BacktestConfig | None  # 사용된 설정
    strategy_name: str  # 전략 이름
```

#### 메서드

##### `summary() -> str`

결과 요약 문자열을 생성합니다.

**반환값:**
- `str`: 포맷된 요약 문자열

**예시:**
```python
result = engine.run_backtest(...)
print(result.summary())
# 출력:
# ==================================================
# Strategy: VanillaVBO
# ==================================================
# CAGR: 105.40%
# MDD: 24.97%
# Calmar Ratio: 4.22
# Sharpe Ratio: 1.97
# Win Rate: 36.03%
# Total Trades: 705
# Final Equity: 383314.00
# ==================================================
```

### `Trade`

단일 거래 기록을 나타내는 데이터 클래스입니다.

```python
@dataclass
class Trade:
    ticker: str  # 티커
    entry_date: date  # 진입 날짜
    entry_price: float  # 진입 가격
    exit_date: date | None  # 청산 날짜
    exit_price: float | None  # 청산 가격
    amount: float  # 거래 수량
    pnl: float  # 손익
    pnl_pct: float  # 손익률 (%)
    is_whipsaw: bool  # 휩소 여부
```

**속성:**
- `is_closed`: 거래가 종료되었는지 여부

## 편의 함수

### `run_backtest()`

백테스트를 간편하게 실행하는 함수입니다.

```python
from src.backtester import run_backtest, BacktestConfig
from src.strategies.volatility_breakout import VanillaVBO

strategy = VanillaVBO()
config = BacktestConfig(
    initial_capital=1_000_000.0,
    fee_rate=0.0005,
    max_slots=4
)

result = run_backtest(
    tickers=["KRW-BTC", "KRW-ETH"],
    strategy=strategy,
    config=config
)
```

**매개변수:**
- `tickers`: 거래할 티커 목록
- `strategy`: 사용할 거래 전략
- `config`: 백테스팅 설정
- `start_date`: 시작 날짜 (선택)
- `end_date`: 종료 날짜 (선택)

**반환값:**
- `BacktestResult`: 백테스트 결과

## 리포트 생성

### `generate_report()`

백테스트 결과를 시각적 리포트로 생성합니다.

```python
from src.backtester.report import generate_report
from pathlib import Path

generate_report(
    result=backtest_result,
    save_path=Path("reports/my_backtest.html"),
    show=True  # 브라우저에서 자동 열기
)
```

**매개변수:**
- `result`: `BacktestResult` 객체
- `save_path`: 저장할 파일 경로
- `show`: 브라우저에서 자동으로 열지 여부

**생성되는 리포트 내용:**
- 자산 곡선 차트
- 낙폭 차트
- 월별 수익률 히트맵
- 거래 통계
- 성능 지표 요약

## 사용 예제

### 기본 백테스트

```python
from src.backtester import run_backtest, BacktestConfig
from src.strategies.volatility_breakout import VanillaVBO

# 전략 생성
strategy = VanillaVBO(
    sma_period=4,
    trend_sma_period=8,
    short_noise_period=4,
    long_noise_period=8
)

# 설정
config = BacktestConfig(
    initial_capital=1_000_000.0,
    fee_rate=0.0005,
    slippage_rate=0.0005,
    max_slots=4
)

# 백테스트 실행
result = run_backtest(
    tickers=["KRW-BTC", "KRW-ETH"],
    strategy=strategy,
    config=config
)

# 결과 출력
print(result.summary())

# 리포트 생성
from src.backtester.report import generate_report
from pathlib import Path

generate_report(
    result=result,
    save_path=Path("reports/backtest_report.html"),
    show=True
)
```

### 커스텀 설정으로 백테스트

```python
from datetime import date
from src.backtester import run_backtest, BacktestConfig

config = BacktestConfig(
    initial_capital=10_000_000.0,  # 1천만원
    fee_rate=0.0005,  # 0.05%
    slippage_rate=0.001,  # 0.1%
    max_slots=8,  # 최대 8개 종목
    use_cache=True
)

result = run_backtest(
    tickers=["KRW-BTC", "KRW-ETH", "KRW-XRP"],
    strategy=strategy,
    config=config,
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31)
)
```

### 거래 내역 분석

```python
result = run_backtest(...)

# 승리 거래만 필터링
winning_trades = [t for t in result.trades if t.pnl > 0]

# 손실 거래만 필터링
losing_trades = [t for t in result.trades if t.pnl < 0]

# 평균 수익률 계산
avg_win = sum(t.pnl_pct for t in winning_trades) / len(winning_trades)
avg_loss = sum(t.pnl_pct for t in losing_trades) / len(losing_trades)

print(f"평균 승리: {avg_win:.2f}%")
print(f"평균 손실: {avg_loss:.2f}%")
```

## 성능 최적화

### 캐싱 사용

지표 계산 결과를 캐싱하여 성능을 향상시킬 수 있습니다:

```python
config = BacktestConfig(use_cache=True)
```

### 벡터화 연산

엔진은 pandas/numpy의 벡터화 연산을 사용하여 빠른 성능을 제공합니다. 전략의 `generate_signals()` 메서드에서 벡터화된 연산을 사용하는 것이 좋습니다.

## 관련 문서

- [시작 가이드](../guides/getting_started.md)
- [전략 커스터마이징](../guides/strategy_customization.md)
- [아키텍처 문서](../architecture.md)
