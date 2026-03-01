# crypto-lab

Upbit/Binance를 위한 퀀트 백테스팅 및 전략 연구 플랫폼.

- **벡터화 백테스터** — 현실적인 비용 모델(수수료 + 슬리피지)을 갖춘 이벤트 기반 시뮬레이션
- **파라미터 최적화** — 병렬 실행을 지원하는 그리드 및 랜덤 서치
- **워크포워드 분석** — 과최적화 감지를 위한 아웃오브샘플 검증
- **데이터 수집** — Upbit 및 Binance에서 OHLCV 증분 수집
- **리스크 분석** — VaR, CVaR, 포트폴리오 최적화(MPT, 리스크 패리티)
- **CLI** — 전체 워크플로우를 단일 `crypto-lab` 명령으로 실행

---

## 설치

Python 3.12 이상 필요.

```bash
# 전체 의존성 설치
pip install -e ".[analysis,dev]"

# 또는 uv 사용
uv sync --all-extras
```

---

## 빠른 시작

### 등록된 전략 목록 확인

```bash
crypto-lab list
# VBO
# VBO_DAY
```

### 백테스트 실행

```bash
crypto-lab backtest \
  --tickers KRW-BTC KRW-ETH \
  --strategy VBO \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --capital 10000000 \
  --slots 2

# === Backtest: VBO ===
#   Total Return : 142.35%
#   CAGR         : 43.21%
#   MDD          : -17.8%
#   Sharpe       : 2.54
#   Win Rate     : 61.2%
#   Total Trades : 552
```

### 파라미터 최적화

```bash
crypto-lab optimize \
  --tickers KRW-BTC KRW-ETH \
  --strategy VBO \
  --metric sharpe_ratio \
  --method grid \
  --workers 4

# === Optimize: VBO ===
#   Best Score (sharpe_ratio): 2.5400
#   Best Params:
#     noise_ratio: 0.6
#     btc_ma: 30
#     ma_short: 3
```

### 워크포워드 분석

```bash
crypto-lab wfa \
  --tickers KRW-BTC KRW-ETH \
  --strategy VBO \
  --opt-days 365 \
  --test-days 90 \
  --step-days 90 \
  --metric sharpe_ratio

# === Walk-Forward Analysis ===
#   Periods       : 8
#   Positive      : 7/8
#   Consistency   : 87.5%
#   Avg CAGR      : 38.50%
#   Avg Sharpe    : 2.21
#   Avg MDD       : -19.2%
```

### 데이터 수집

```bash
crypto-lab collect \
  --tickers KRW-BTC KRW-ETH KRW-XRP \
  --interval day \
  --source upbit
```

---

## CLI 레퍼런스

```
crypto-lab [--log-level LEVEL] COMMAND

Commands:
  backtest   과거 데이터로 전략 백테스트 실행
  optimize   그리드/랜덤 파라미터 서치
  collect    거래소에서 OHLCV 데이터 수집
  wfa        워크포워드 분석
  list       등록된 전략 목록 출력
```

`backtest`, `optimize`, `wfa` 공통 플래그:

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--tickers` | (필수) | 공백으로 구분된 Upbit 티커. 예: `KRW-BTC KRW-ETH` |
| `--strategy` | (필수) | 등록된 전략 이름 |
| `--start` | 전체 데이터 | 시작일 `YYYY-MM-DD` |
| `--end` | 전체 데이터 | 종료일 `YYYY-MM-DD` |
| `--capital` | 1,000,000 | 초기 자본금 (원) |
| `--slots` | 5 | 최대 동시 포지션 수 |
| `--fee` | 0.0005 | 거래당 수수료율 (0.05%) |
| `--interval` | `day` | 캔들 인터벌 (`day`, `minute240`, …) |

---

## 프로젝트 구조

```
src/
├── main.py                  # CLI 진입점
├── cli/                     # CLI 서브커맨드 모듈
│   ├── cmd_backtest.py      #   backtest, optimize
│   ├── cmd_data.py          #   collect
│   └── cmd_wfa.py           #   wfa
├── strategies/
│   ├── base.py              # Strategy 추상 기반 클래스
│   ├── registry.py          # StrategyFactory 싱글턴
│   └── volatility_breakout/
│       ├── vbo_v1.py        # VBOV1 — 돌파 + BTC MA 필터 + MA 청산
│       └── vbo_day_exit.py  # VBODayExit — 고정 1일 보유
├── backtester/
│   ├── engine/              # 벡터화 + 이벤트 기반 엔진, run_backtest()
│   ├── optimization.py      # 그리드/랜덤 파라미터 최적화기
│   ├── wfa/                 # 워크포워드 분석
│   ├── analysis/            # 순열 테스트, 강건성, 부트스트랩
│   └── models.py            # BacktestConfig, BacktestResult, Trade
├── data/
│   ├── collector.py         # UpbitDataCollector (증분 parquet 업데이트)
│   ├── collector_factory.py # DataCollectorFactory
│   ├── upbit_source.py      # Upbit OHLCV 데이터 소스
│   ├── binance_source.py    # Binance OHLCV 데이터 소스
│   └── cache/               # LRU 데이터 캐시
├── risk/
│   ├── position_sizing.py   # 균등, 변동성 조정, 켈리 포지션 사이징
│   ├── portfolio_methods.py # MPT + 리스크 패리티 최적화
│   └── metrics*.py          # VaR, CVaR, Sharpe, Sortino, …
├── config/                  # Pydantic 설정 + YAML 로더
└── utils/
    ├── indicators.py         # SMA, EMA, ATR, RSI, 볼린저 밴드
    └── indicators_vbo.py     # VBO 전용: 목표가, 노이즈 범위
```

---

## 전략 작성

`Strategy`를 서브클래싱하고 `@registry.register`로 등록:

```python
from src.strategies.base import Strategy
from src.strategies.registry import registry
import pandas as pd


@registry.register("MyStrategy")
class MyStrategy(Strategy):
    def __init__(self, period: int = 20) -> None:
        super().__init__(name="MyStrategy")
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sma"] = df["close"].rolling(self.period).mean()
        df["entry_signal"] = df["close"] > df["sma"]
        df["exit_signal"] = df["close"] < df["sma"]
        return df

    @classmethod
    def parameter_schema(cls) -> dict[str, object]:
        return {
            "period": {"type": "int", "min": 5, "max": 60, "step": 5},
        }
```

`parameter_schema()`가 `optimize`와 `wfa`를 구동하므로 별도 연결이 필요 없다.

---

## 변동성 돌파 전략 (VBO)

진입 조건 (두 조건 모두 충족 시):

1. **돌파**: `high ≥ open + 전일_range × noise_ratio`
2. **BTC 필터**: `전일 BTC 종가 > 전일 BTC MA(btc_ma)`

청산 조건 (VBOV1): 전일 `close < SMA(ma_short)` → 다음 시가에 청산

청산 조건 (VBODayExit): 항상 다음 날 시가에 청산

### 최적 파라미터 (BTC+ETH, 2020–2024)

| 목표 | noise_ratio | btc_ma | ma_short | Sharpe | CAGR | MDD |
|------|-------------|--------|----------|--------|------|-----|
| 균형 | 0.6 | 30 | 3 | 2.54 | +121% | −17.9% |
| 수익 극대화 | 0.3 | 10 | 3 | 2.20 | +128% | −23.9% |
| 낙폭 최소화 | 0.8 | 30 | 3 | 2.05 | +101% | −16.7% |

전체 스윕 연구 결과: [`src/research/results/vb_upbit/README.md`](src/research/results/vb_upbit/README.md)

---

## 개발

```bash
# 테스트 실행
pytest tests/ -x -q

# 린트 + 포맷
ruff check src/ && ruff format src/

# 타입 검사
mypy src/ --strict

# 세 가지 모두 (품질 게이트)
pytest tests/ -x -q && ruff check src/ && mypy src/ --strict
```

커버리지 기준: **80%** (현재 ~84%).

### 코드 컨벤션

- 파일 200줄 이하, 함수 50줄 이하
- 모든 public 함수에 타입 어노테이션 필수 (mypy strict)
- 주석은 영어로만 작성
- 무거운 의존성(pyupbit, ccxt, matplotlib)은 함수 내부에서 지연 임포트

---

## 데이터 레이아웃

```
data/
├── upbit/          # Parquet 파일: KRW-BTC_day.parquet, KRW-ETH_minute240.parquet, …
└── binance/        # Parquet 파일: BTC_USDT_1d.parquet, …
```

파일은 `crypto-lab collect` 실행 시 자동으로 생성/업데이트된다.

---

## 라이선스

MIT
