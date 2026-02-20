# crypto-lab

암호화폐 전략 백테스팅 및 연구 툴킷.

## Setup

```bash
pip install -e .
```

## CLI 워크플로우

```
collect → backtest → optimize → wfa → analyze
```

### 1. 데이터 수집

```bash
crypto-lab collect --tickers KRW-BTC KRW-ETH --interval day
```

Upbit API에서 OHLCV 데이터를 수집해 `data/upbit/` 에 parquet으로 저장한다.

### 2. 백테스트 (빠른 확인)

```bash
crypto-lab backtest --tickers KRW-BTC KRW-ETH --strategy VBO_DAY --start 2022-01-01
```

### 3. 파라미터 최적화

```bash
crypto-lab optimize --tickers KRW-BTC KRW-ETH --strategy VBO_DAY --start 2022-01-01
```

### 4. WFA — 과최적화 검증

```bash
crypto-lab wfa --tickers KRW-BTC KRW-ETH --strategy VBO_DAY --start 2022-01-01 --slots 2
```

기간별로 CAGR / MDD / Sharpe / Sortino / WinRate / Trades / 파라미터를 출력한다.
`--opt-days`(기본 365) 구간에서 최적화 후 `--test-days`(기본 90) 구간으로 검증한다.

### 5. 통계적 유의성 + 강건성 분석

```bash
crypto-lab analyze --tickers KRW-BTC KRW-ETH --strategy VBO_DAY --start 2022-01-01
crypto-lab analyze ... --skip-perm --skip-robust   # 빠른 실행 (backtest만)
```

permutation test(알파 유의성), bootstrap CI, 파라미터 강건성, 벤치마크 비교, go-live 체크리스트.

### 전략 목록

```bash
crypto-lab list
```

---

## 전략 개발

1. `src/strategies/<name>/<file>.py` 에 `Strategy` 서브클래스 작성
2. 필수 구현:
   - `calculate_indicators(df)` — 시그널에 필요한 컬럼 계산
   - `generate_signals(df)` — `entry_signal`, `exit_signal` bool 컬럼 생성
   - `parameter_schema()` classmethod — 튜닝 가능한 파라미터 정의
3. `src/strategies/registry.py` 에 등록
4. `crypto-lab list` 로 확인

`parameter_schema()` 예시:
```python
@classmethod
def parameter_schema(cls) -> dict[str, object]:
    return {
        "noise_ratio": {"type": "float", "min": 0.1, "max": 0.9, "step": 0.1},
        "ma_short":    {"type": "int",   "min": 3,   "max": 20,  "step": 1},
    }
```

---

## 테스트

```bash
pytest tests/ -x -q
pytest tests/integration/ -v
```

---

## 알려진 버그 패턴 (주의)

### 캐시 키 충돌

`src/backtester/engine/data_loader.py:get_cache_params()`는 반드시
`strategy.parameter_schema()`를 기반으로 동적으로 키를 생성해야 한다.
하드코딩 속성명을 사용하면 모든 파라미터 조합이 동일한 캐시를 공유하게 되어
WFA/optimize 결과가 파라미터와 무관하게 동일해진다.

```python
# 올바른 방식 (현재 구현)
schema = strategy.parameter_schema()
for name in schema:
    if hasattr(strategy, name):
        params[name] = getattr(strategy, name)

# 잘못된 방식 (하지 말 것)
for name in ["sma_period", "trend_sma_period", ...]:  # 하드코딩
    ...
```

### WFA 데이터 누수

`generate_periods()`는 `gap_days=1`로 최적화 구간 끝과 테스트 구간 시작 사이에
1일 갭을 둔다. 이를 제거하면 데이터 누수가 발생한다.

### sortino_ratio 누락

`BacktestResult.sortino_ratio`는 두 경로 모두에서 계산해야 한다:
- `src/backtester/engine/metrics_calculator.py` (풀 엔진)
- `src/backtester/wfa/walk_forward_backtest.py` (WFA 경량 경로)

한 쪽에서만 계산하면 WFA 최적화 메트릭으로 `sortino_ratio`를 사용할 때
모든 값이 0.0으로 나온다.
