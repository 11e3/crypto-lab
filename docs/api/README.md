# API 문서

이 디렉토리에는 API 참조 문서가 포함됩니다.

## 📚 API 문서 목록

### 핵심 레이어

- **[Strategy Layer](strategies.md)** - 전략 및 조건 인터페이스
  - `Strategy` 기본 클래스
  - `Condition` 인터페이스
  - `VBOV1` 전략 구현

- **[Backtester API](backtester.md)** - 백테스팅 엔진
  - `VectorizedBacktestEngine`
  - `BacktestConfig` 및 `BacktestResult`
  - 리포트 생성

- **[Execution Layer](execution.md)** - 실시간 거래 봇
  - `TradingBotFacade`
  - `OrderManager`, `PositionManager`
  - `SignalHandler` 및 이벤트 버스

- **[Exchange Layer](exchange.md)** - 거래소 추상화
  - `Exchange` 인터페이스
  - `UpbitExchange` 구현
  - 주문 및 데이터 조회

- **[Data Layer](data.md)** - 데이터 수집 및 캐싱
  - `DataSource` 인터페이스
  - `UpbitDataCollector`
  - `IndicatorCache`

## 🚀 빠른 시작

### 전략 사용

```python
from src.strategies.volatility_breakout import VBOV1

strategy = VBOV1(
    sma_period=4,
    trend_sma_period=8
)
```

### 백테스트 실행

```python
from src.backtester import run_backtest, BacktestConfig

config = BacktestConfig(initial_capital=1_000_000.0)
result = run_backtest(
    tickers=["KRW-BTC"],
    strategy=strategy,
    config=config
)
```

### 실시간 거래

```python
from src.execution.bot_facade import TradingBotFacade

bot = TradingBotFacade()
bot.run()
```

## 📖 상세 문서

각 레이어의 상세 API 문서는 위 링크를 참조하세요.

## 🔗 관련 문서

- [시작 가이드](../guides/getting_started.md)
- [전략 커스터마이징](../guides/strategy_customization.md)
- [아키텍처 문서](../architecture.md)

## 📝 자동 생성 문서

Sphinx를 사용한 자동 API 문서 생성이 구현되었습니다.

### 문서 빌드

**Windows (PowerShell):**
```powershell
# 의존성 설치
uv sync --extra docs

# 문서 빌드
cd docs
.\build.ps1

# 또는 직접 실행
cd docs
uv run sphinx-build -b html . _build/html
```

**Linux/Mac:**
```bash
# 의존성 설치
uv sync --extra docs

# 문서 빌드
make docs

# 또는 직접 실행
cd docs
uv run sphinx-build -b html . _build/html
```

### 로컬에서 문서 확인

**Windows (PowerShell):**
```powershell
# 문서 빌드 및 로컬 서버 실행
cd docs
.\serve.ps1

# 브라우저에서 http://localhost:8000 열기
```

**Linux/Mac:**
```bash
# 문서 빌드 및 로컬 서버 실행
make docs-serve

# 브라우저에서 http://localhost:8000 열기
```

### 문서 정리

```bash
# 빌드 아티팩트 삭제
make docs-clean
```

### 생성되는 문서

빌드 후 `docs/_build/html/` 디렉토리에 다음이 생성됩니다:

- **자동 생성 API 문서**: 소스 코드의 docstring에서 자동 생성
- **모듈 인덱스**: 모든 모듈의 인덱스
- **검색 기능**: 전체 문서 검색
- **크로스 레퍼런스**: 모듈 간 자동 링크

### 문서 구조

- `index.html`: 메인 문서 페이지
- `api/index.html`: API 참조 인덱스
- `api/strategies.html`: Strategy Layer API
- `api/backtester.html`: Backtester API
- `api/execution.html`: Execution Layer API
- `api/exchange.html`: Exchange Layer API
- `api/data.html`: Data Layer API
