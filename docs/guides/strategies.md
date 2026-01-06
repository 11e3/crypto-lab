# 전략 가이드

이 문서는 Upbit Quant System에서 제공하는 거래 전략들의 상세한 설명을 제공합니다.

## 📚 목차

- [변동성 돌파 전략 (VBO)](#변동성-돌파-전략-vbo)
- [모멘텀 전략](#모멘텀-전략)
- [평균 회귀 전략](#평균-회귀-전략)
- [전략 비교](#전략-비교)

---

## 변동성 돌파 전략 (VBO)

### 개요

**Volatility Breakout (VBO)** 전략은 변동성을 활용한 추세 추종 전략입니다. 전일의 변동 범위를 기반으로 목표 가격을 설정하고, 가격이 목표를 돌파할 때 진입합니다.

### 작동 원리

1. **목표 가격 계산**
   - 전일 고가와 저가의 범위를 계산
   - 노이즈 비율(K 값)을 곱하여 변동성 조정
   - 목표 가격 = 당일 시가 + (전일 범위 × K)

2. **진입 조건**
   - 가격이 목표 가격을 돌파 (High ≥ Target)
   - 목표 가격이 SMA 위에 있음
   - 트렌드 정렬: 목표가 트렌드 SMA 위에 있음
   - 노이즈 조건: 단기 노이즈 < 장기 노이즈

3. **청산 조건**
   - 종가가 SMA 아래로 떨어짐

### 전략 변형

#### VanillaVBO
기본 VBO 전략으로 모든 필터를 포함합니다.

```python
from src.strategies.volatility_breakout import VanillaVBO

strategy = VanillaVBO(
    sma_period=4,
    trend_sma_period=8,
    short_noise_period=4,
    long_noise_period=8,
)
```

**파라미터:**
- `sma_period`: 청산용 SMA 기간 (기본값: 4)
- `trend_sma_period`: 트렌드 확인용 SMA 기간 (기본값: 8)
- `short_noise_period`: K 값 계산 기간 (기본값: 4)
- `long_noise_period`: 노이즈 기준선 기간 (기본값: 8)

#### MinimalVBO
최소한의 조건만 사용하는 단순 버전입니다.

```python
from src.strategies.volatility_breakout import MinimalVBO

strategy = MinimalVBO()
```

**특징:**
- 돌파 조건만 사용
- 필터 없음
- 빠른 진입/청산

#### StrictVBO
추가 필터를 사용하는 엄격한 버전입니다.

```python
from src.strategies.volatility_breakout import StrictVBO

strategy = StrictVBO(
    max_noise=0.6,
    min_volatility_pct=0.01,
)
```

**추가 조건:**
- 노이즈 임계값: 최대 노이즈 비율 제한
- 변동성 범위: 최소/최대 변동성 범위 내에서만 거래

### 사용 예제

```python
from src.backtester import run_backtest, BacktestConfig
from src.strategies.volatility_breakout import VanillaVBO

# 전략 생성
strategy = VanillaVBO(
    sma_period=5,
    trend_sma_period=10,
    short_noise_period=5,
    long_noise_period=10,
)

# 백테스트 설정
config = BacktestConfig(
    initial_capital=1_000_000.0,
    fee_rate=0.0005,
    max_slots=4,
)

# 백테스트 실행
results = run_backtest(
    tickers=["KRW-BTC", "KRW-ETH"],
    strategy=strategy,
    interval="day",
    config=config,
)
```

### CLI 사용

```bash
# Vanilla VBO
uv run upbit-quant backtest --strategy vanilla --tickers KRW-BTC

# Minimal VBO
uv run upbit-quant backtest --strategy minimal --tickers KRW-BTC

# Legacy VBO (기존 bt.py와 동일한 설정)
uv run upbit-quant backtest --strategy legacy --tickers KRW-BTC
```

### 특징

**장점:**
- 변동성이 큰 시장에서 효과적
- 명확한 진입/청산 신호
- 다양한 필터로 신호 품질 향상 가능

**단점:**
- 횡보장에서 손실 가능
- 노이즈가 많은 시장에서 성과 저하
- 파라미터 튜닝 필요

---

## 모멘텀 전략

### 개요

**Momentum Strategy**는 추세를 따라가는 모멘텀 기반 전략입니다. RSI, MACD, 이동평균선을 조합하여 강한 추세를 포착합니다.

### 작동 원리

1. **지표 계산**
   - **SMA**: 추세 확인용 이동평균선
   - **RSI**: 모멘텀 강도 측정 (과매수/과매도)
   - **MACD**: 추세 방향 및 모멘텀 변화 감지

2. **진입 조건**
   - 가격이 SMA 위에 있음 (상승 추세 확인)
   - MACD가 시그널선 위에 있음 (강세 모멘텀)

3. **청산 조건**
   - 가격이 SMA 아래로 떨어짐 (추세 전환)
   - RSI가 과매수 구간 (기본값: 70 이상)
   - MACD가 시그널선 아래로 교차 (약세 전환)

### 전략 변형

#### MomentumStrategy
기본 모멘텀 전략으로 모든 조건을 포함합니다.

```python
from src.strategies.momentum import MomentumStrategy

strategy = MomentumStrategy(
    sma_period=20,
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    rsi_oversold=30.0,
    rsi_overbought=70.0,
)
```

**파라미터:**
- `sma_period`: 추세 확인용 SMA 기간 (기본값: 20)
- `rsi_period`: RSI 계산 기간 (기본값: 14)
- `macd_fast`: MACD 빠른 EMA 기간 (기본값: 12)
- `macd_slow`: MACD 느린 EMA 기간 (기본값: 26)
- `macd_signal`: MACD 시그널선 기간 (기본값: 9)
- `rsi_oversold`: RSI 과매도 임계값 (기본값: 30)
- `rsi_overbought`: RSI 과매수 임계값 (기본값: 70)

#### SimpleMomentumStrategy
SMA만 사용하는 단순 버전입니다.

```python
from src.strategies.momentum import SimpleMomentumStrategy

strategy = SimpleMomentumStrategy()
```

**특징:**
- 가격이 SMA 위에 있으면 매수
- 가격이 SMA 아래로 떨어지면 매도
- 가장 단순한 추세 추종 전략

### 사용 예제

```python
from src.backtester import run_backtest, BacktestConfig
from src.strategies.momentum import MomentumStrategy

# 전략 생성
strategy = MomentumStrategy(
    sma_period=20,
    rsi_period=14,
    rsi_overbought=75.0,  # 더 보수적인 청산
)

# 백테스트 실행
results = run_backtest(
    tickers=["KRW-BTC", "KRW-ETH"],
    strategy=strategy,
    interval="day",
    config=BacktestConfig(initial_capital=1_000_000.0),
)
```

### CLI 사용

```bash
# Momentum 전략
uv run upbit-quant backtest --strategy momentum --tickers KRW-BTC

# Simple Momentum 전략
uv run upbit-quant backtest --strategy simple-momentum --tickers KRW-BTC
```

### 특징

**장점:**
- 강한 추세에서 높은 수익
- 명확한 추세 방향 확인
- 다양한 모멘텀 지표 조합 가능

**단점:**
- 횡보장에서 손실 가능
- 추세 전환 시 늦은 반응
- RSI 과매수 구간에서 조기 청산 가능

---

## 페어 트레이딩 전략 (Pair Trading Strategy)

### 개요

페어 트레이딩은 두 종목 간의 가격 차이(스프레드)를 이용하는 통계적 차익거래 전략입니다. 두 종목이 상관관계가 높을 때, 스프레드가 평균에서 벗어나면 평균 회귀를 기대하여 거래합니다.

### 원리

1. **스프레드 계산**: 두 종목의 가격 비율 또는 차이를 계산
2. **Z-score 계산**: 스프레드가 평균에서 얼마나 벗어났는지 표준화
3. **진입**: Z-score가 임계값을 넘으면 스프레드가 줄어들 것으로 예상하여 진입
4. **청산**: Z-score가 평균(0)에 가까워지면 청산

### 주요 파라미터

- `lookback_period` (기본값: 60): 스프레드 평균/표준편차 계산 기간
- `entry_z_score` (기본값: 2.0): 진입 Z-score 임계값
- `exit_z_score` (기본값: 0.5): 청산 Z-score 임계값
- `spread_type` (기본값: "ratio"): 스프레드 계산 방식 ("ratio" 또는 "difference")

### 사용 예시

#### Python 코드

```python
from src.strategies.pair_trading import PairTradingStrategy
from src.backtester import run_backtest, BacktestConfig

# 전략 생성
strategy = PairTradingStrategy(
    name="PairTradingStrategy",
    lookback_period=60,
    entry_z_score=2.0,
    exit_z_score=0.5,
    spread_type="ratio",  # 또는 "difference"
)

# 백테스트 설정
config = BacktestConfig(
    initial_capital=1_000_000.0,
    fee_rate=0.0005,
    slippage_rate=0.0005,
    max_slots=4,
)

# 백테스트 실행 (정확히 2개 티커 필요)
results = run_backtest(
    tickers=["KRW-BTC", "KRW-ETH"],  # 정확히 2개 필요
    strategy=strategy,
    config=config,
)
```

#### CLI 사용

```bash
# 페어 트레이딩 백테스트 (정확히 2개 티커 필요)
uv run upbit-quant backtest \
  --strategy pair-trading \
  --tickers KRW-BTC --tickers KRW-ETH \
  --initial-capital 1000000
```

### 주의사항

- **정확히 2개 티커 필요**: 페어 트레이딩은 반드시 2개의 티커가 필요합니다
- **상관관계**: 두 종목이 높은 상관관계를 가져야 효과적입니다
- **충분한 데이터**: `lookback_period` 이상의 데이터가 필요합니다

### 장단점

**장점:**
- 시장 방향성에 덜 의존 (상대적 가격 차이 이용)
- 변동성이 낮은 시장에서도 수익 가능
- 리스크 분산 효과

**단점:**
- 두 종목의 상관관계가 깨지면 손실 가능
- 거래 기회가 상대적으로 적을 수 있음
- 두 종목 모두에 대한 포지션 관리 필요

---

## 전략 비교

### 전략 선택 가이드

| 전략 | 추천 시장 환경 | 주요 특징 |
|------|---------------|----------|
| **VBO** | 변동성이 큰 시장 | 돌파 기반, 빠른 진입 |
| **Momentum** | 강한 추세 시장 | 추세 추종, 안정적 수익 |
| **Mean Reversion** | 횡보/변동성 큰 시장 | 평균 회귀, 과매수/과매도 포착 |
| **Pair Trading** | 상관관계 높은 종목 쌍 | 통계적 차익거래, 시장 중립적 |

### 성능 비교 예시

백테스트 결과 (참고용, 실제 결과는 시장 조건에 따라 다름):

```
VBO (Legacy):
- CAGR: ~15-25%
- MDD: ~15-20%
- Win Rate: ~50-55%

Momentum:
- CAGR: ~13-33%
- MDD: ~14-16%
- Win Rate: ~53-54%

Mean Reversion:
- CAGR: 시장 환경에 따라 다름
- MDD: 변동성에 따라 다름
- Win Rate: ~50-60% (횡보장에서 높음)
```

### 전략 조합

여러 전략을 조합하여 포트폴리오를 구성할 수 있습니다:

```python
from src.strategies.volatility_breakout import VanillaVBO
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy

# 전략별로 다른 티커 할당
vbo_strategy = VanillaVBO()
momentum_strategy = MomentumStrategy()
mean_reversion_strategy = MeanReversionStrategy()

# 각 전략으로 별도 백테스트 실행
vbo_results = run_backtest(tickers=["KRW-BTC"], strategy=vbo_strategy, ...)
momentum_results = run_backtest(tickers=["KRW-ETH"], strategy=momentum_strategy, ...)
mean_reversion_results = run_backtest(
    tickers=["KRW-XRP"], strategy=mean_reversion_strategy, ...
)
```

---

## 관련 문서

- [전략 커스터마이징 가이드](strategy_customization.md) - 커스텀 전략 작성 방법
- [API 참조](../api/strategies.md) - 전략 API 상세 문서
- [시작 가이드](getting_started.md) - 백테스트 실행 방법
