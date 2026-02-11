# 시작 가이드

## 설치

### 사전 요구사항

이 프로젝트는 **`uv`**를 사용하여 Python 버전, 가상환경, 의존성을 관리합니다.

**`uv` 설치** (아직 설치하지 않은 경우):

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

자세한 설치 방법: https://docs.astral.sh/uv/getting-started/installation/

### 1. 저장소 클론 및 가상 환경 설정

```bash
git clone <repository-url>
cd upbit-quant-system

# 가상환경 생성
uv venv

# 가상환경 활성화 (선택사항 - uv run을 사용하면 자동으로 활성화됨)
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. 의존성 설치

**권장 방법: `uv sync` 사용**

```bash
# 모든 의존성 설치 (프로덕션 + 개발)
uv sync --all-groups

# 또는 개발 의존성 포함
uv sync --dev
```

**대안: pip 사용 (기존 방식)**

```bash
# 가상환경 활성화 후
pip install -e ".[dev]"

# 또는 개발 도구 없이
pip install -e .
```

**참고**: `uv sync`는 `uv.lock` 파일을 기반으로 정확한 버전의 의존성을 설치합니다.

### 3. 설정 파일 생성

#### 방법 1: .env 파일 사용 (권장)

로컬 개발 시 `.env` 파일 사용:

```bash
# 1. 템플릿 복사
cp .env.example .env

# 2. .env 파일 편집하여 필수 값 입력
# UPBIT_ACCESS_KEY=your_access_key
# UPBIT_SECRET_KEY=your_secret_key
```

#### 방법 2: 환경 변수 직접 설정 (프로덕션 권장)

프로덕션 환경에서는 환경 변수 직접 설정:

```bash
# Linux/Mac
export UPBIT_ACCESS_KEY="your_access_key"
export UPBIT_SECRET_KEY="your_secret_key"
export TELEGRAM_TOKEN="your_telegram_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Windows PowerShell
$env:UPBIT_ACCESS_KEY="your_access_key"
$env:UPBIT_SECRET_KEY="your_secret_key"
```

#### 방법 3: YAML 파일 사용 (선택적, 기본값만)

YAML 파일은 선택사항이며 기본값만 포함:

```bash
# 템플릿 복사 (선택적)
cp config/settings.yaml.example config/settings.yaml
# settings.yaml 편집 (기본값만, 민감 정보는 .env 사용)
```

**중요**: 
- ✅ **민감한 정보(API 키)는 환경 변수나 .env 파일만 사용**
- ✅ **설정 우선순위**: 환경 변수 > .env > YAML > 기본값
- ✅ **자세한 내용**: [설정 관리 가이드](configuration.md) 참고

## 데이터 수집

### CLI 사용

**`uv run` 사용 (권장 - 가상환경 자동 활성화)**

```bash
# 기본 티커 및 인터벌로 데이터 수집
uv run upbit-quant collect

# 특정 티커 및 인터벌 지정
uv run upbit-quant collect -t KRW-BTC KRW-ETH -i day minute240

# 전체 새로고침 (기존 데이터 무시)
uv run upbit-quant collect --full-refresh
```

**가상환경 활성화 후 직접 실행**

```bash
# 가상환경 활성화 후
upbit-quant collect
upbit-quant collect -t KRW-BTC KRW-ETH -i day minute240
upbit-quant collect --full-refresh
```

### Python 코드 사용

```python
from src.data.collector import UpbitDataCollector, Interval

collector = UpbitDataCollector()
results = collector.collect_multiple(
    tickers=["KRW-BTC", "KRW-ETH"],
    intervals=[Interval.DAY, Interval.MINUTE240],
    force_full_refresh=False
)
```

## 백테스팅

### CLI 사용

**`uv run` 사용 (권장)**

```bash
# 기본 설정으로 백테스트 실행
uv run upbit-quant backtest

# 커스텀 설정으로 백테스트 실행
uv run upbit-quant backtest \
  --tickers KRW-BTC --tickers KRW-ETH \
  --initial-capital 1000000 \
  --fee-rate 0.0005 \
  --max-slots 4 \
  --strategy vanilla
```

**가상환경 활성화 후 직접 실행**

```bash
upbit-quant backtest \
  --tickers KRW-BTC --tickers KRW-ETH \
  --initial-capital 1000000 \
  --fee-rate 0.0005 \
  --max-slots 4 \
  --strategy vanilla
```

**참고**: `--tickers` 옵션은 여러 번 지정할 수 있습니다 (`multiple=True`).

### Python 코드 사용

```python
from src.backtester.models import BacktestConfig
from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.strategies.volatility_breakout.vbo_v1 import VBOV1

# 전략 생성
strategy = VBOV1(
    name="VBOV1",
    ma_short=5,
    btc_ma=10,
)

# 백테스트 설정
config = BacktestConfig(
    initial_capital=1_000_000.0,
    fee_rate=0.0005,
    slippage_rate=0.0005,
    max_slots=4,
    use_cache=True,
)

# 백테스트 실행
results = run_backtest(
    tickers=["KRW-BTC", "KRW-ETH"],
    strategy=strategy,
    config=config,
)

# 리포트 생성
from src.backtester.report import generate_report
from pathlib import Path

generate_report(
    results,
    save_path=Path("reports/my_backtest.html"),
    show=False,
)
```

## 실시간 거래 봇 실행

### CLI 사용

**`uv run` 사용 (권장)**

```bash
# 기본 설정 파일 사용
uv run upbit-quant run-bot

# 커스텀 설정 파일 사용
uv run upbit-quant run-bot --config /path/to/custom/settings.yaml
```

**가상환경 활성화 후 직접 실행**

```bash
upbit-quant run-bot
upbit-quant run-bot --config /path/to/custom/settings.yaml
```

### Python 코드 사용

```python
from src.execution.bot_facade import TradingBotFacade
from pathlib import Path

bot = TradingBotFacade(config_path=Path("config/settings.yaml"))
bot.run()
```

## Jupyter Notebook 사용

### 노트북 의존성 설치

```bash
# Jupyter notebook 관련 의존성 설치
uv sync --extra notebooks

# Jupyter kernel 등록 (선택사항)
uv run python -m ipykernel install --user --name=upbit-quant-system --display-name "Python (upbit-quant-system)"
```

### 노트북 실행

```bash
# Jupyter notebook 실행
uv run jupyter notebook

# 또는 JupyterLab 사용
uv run jupyter lab
```

### 노트북 예제

```python
# notebooks/01_vbo.ipynb 또는 notebooks/02_strategy_experiments.ipynb 참고
import pandas as pd
from src.backtester import run_backtest, BacktestConfig
from src.strategies.volatility_breakout.vbo_v1 import VBOV1

# 전략 및 설정
strategy = VBOV1(name="VBOV1", ma_short=5, btc_ma=10)
config = BacktestConfig(...)

# 백테스트 실행
results = run_backtest(...)

# 결과 분석
print(results.summary())
```

## 개발 도구 사용

### 테스트 실행

```bash
# 모든 단위 테스트 실행
uv run pytest tests/unit -v

# 커버리지 포함
uv run pytest tests/unit --cov=src --cov-report=term-missing

# 특정 테스트만 실행
uv run pytest tests/unit/test_backtester/test_engine.py -v
```

### 코드 포맷팅 및 린팅

```bash
# Ruff로 포맷팅 및 린팅
uv run ruff check .
uv run ruff format .

# MyPy 타입 체크
uv run mypy src
```

### 가상환경 재설치

```bash
# 기존 가상환경 삭제 후 재생성
rm -rf .venv  # Linux/Mac
# 또는
Remove-Item -Recurse -Force .venv  # Windows PowerShell

# 새로 생성 및 의존성 설치
uv venv
uv sync --all-groups
```

## 다음 단계

- [전략 커스터마이징](strategy_customization.md)
- [API 참조](../api/)
- [아키텍처 문서](../architecture.md)
