# 🚀 Crypto Quant System

<div align="center">

<!-- Status Badges -->
![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<!-- CI/CD Badges -->
![CI](https://github.com/11e3/crypto-quant-system/actions/workflows/ci.yml/badge.svg)
![CodeQL](https://github.com/11e3/crypto-quant-system/actions/workflows/codeql.yml/badge.svg)
![Docs](https://github.com/11e3/crypto-quant-system/actions/workflows/docs.yml/badge.svg)

<!-- Quality Badges -->
![Coverage](https://codecov.io/gh/11e3/crypto-quant-system/branch/main/graph/badge.svg)
![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-black.svg)
![Type Check](https://img.shields.io/badge/Type%20Check-Mypy%20Strict%2096.7%25-blue.svg)

<!-- Project Badges -->
![Tests](https://img.shields.io/badge/Tests-Passing-green.svg)
![Coverage Threshold](https://img.shields.io/badge/Coverage-86.35%25-brightgreen.svg)

**변동성 돌파 전략 기반 암호화폐 자동 거래 시스템**

[기능](#-features) • [빠른 시작](#-quick-start) • [아키텍처](#-architecture) • [전략](#-전략-상세-설명) • [문서](#-documentation) • [기여하기](#-contributing)

</div>

---

## 📋 개요

Crypto Quant System은 엄격한 타입 안전성과 높은 테스트 커버리지를 갖춘 프로덕션 수준의 암호화폐 자동 거래 시스템입니다. 변동성 돌파(VBO) 전략을 핵심으로 하며, 포괄적인 백테스팅과 실시간 거래 실행 기능을 제공합니다.

### 💼 핵심 가치

- **타입 안전성**: MyPy strict 모드 96.7% 적용 (87/90 모듈)
- **테스트 품질**: 86.35% 커버리지, 모든 핵심 로직 검증
- **모듈식 설계**: 이벤트 버스, 전략 인터페이스, 의존성 주입
- **프로덕션 준비**: Docker 배포, 환경 변수 설정, 로깅/모니터링
- **포괄적 문서**: Sphinx API 문서, 아키텍처 가이드, 예제 노트북

### 🎯 주요 특징

- **고성능 백테스팅**: pandas/numpy 벡터화 엔진으로 빠른 데이터 분석
- **유연한 전략 시스템**: 조건 조합 기반 전략 구성 (VBO, 모멘텀, 평균회귀 등)
- **실시간 거래**: WebSocket 연동, 주문/포지션 관리, 리스크 제어
- **성능 분석**: CAGR, Sharpe, Sortino, MDD 등 40+ 지표
- **시각화**: HTML 리포트, 자산 곡선, 드로우다운 차트, 월별 히트맵

## ✨ 기능

### 핵심 기능

- 🔄 **변동성 돌파 전략**: Larry Williams 변동성 돌파 기반 자동 매매
- 📊 **벡터화 백테스팅**: 빠른 과거 성능 분석 (8년+ 데이터)
- 🤖 **실시간 거래 봇**: WebSocket 실시간 시장 데이터 연동
- 📈 **성능 분석**: 포괄적인 지표 계산 (CAGR, Sharpe, MDD, Calmar 등)
- 🎨 **시각적 리포트**: 자산 곡선, 낙폭 차트, 월별/연도별 히트맵
- 🔧 **파라미터 최적화**: Grid Search, Walk-Forward, Monte Carlo 분석

### 기술적 우수성

- 🏗️ **클린 아키텍처**: SOLID 원칙, Facade/Strategy/EventBus 패턴
- 🧪 **높은 테스트 커버리지**: 86.35% 전체 커버리지, 핵심 모듈 90%+
- 📝 **엄격한 타입 안전성**: MyPy strict 96.7% (87/90 모듈)
- 🔒 **보안**: 환경 변수 설정, API 키 암호화 권장, CodeQL 스캔
- 🐳 **Docker 지원**: docker-compose로 1분 내 배포
- 📚 **포괄적인 문서**: Sphinx 문서, 10+ 예제 스크립트, 3개 Jupyter 노트북

## 🛠️ 기술 스택

### 핵심 기술
- **Python 3.14+**: 최신 타입 힌트 지원
- **pandas/numpy**: 벡터화 데이터 처리 (100x+ 속도 향상)
- **pydantic v2**: 타입 안전 설정 관리, 런타임 검증
- **click**: 직관적인 CLI 프레임워크
- **pyupbit**: Upbit 거래소 API (다른 거래소 확장 가능)

### 개발 도구
- **uv**: 차세대 Python 패키지 관리자 (pip 대비 10x 빠름)
- **Ruff**: 초고속 린터 및 포매터 (Flake8, isort, Black 통합)
- **MyPy**: strict 모드 타입 검사 (96.7% 커버리지)
- **pytest**: 테스트 프레임워크 (86.35% 코드 커버리지)
- **pre-commit**: 코드 품질 자동 검사
- **nox**: 자동화된 테스트 세션 관리

### 인프라 및 배포
- **Docker/docker-compose**: 1분 내 배포 환경 구축
- **WebSocket**: 실시간 시장 데이터 스트리밍
- **parquet**: 고효율 데이터 저장 (CSV 대비 10x 압축)
- **환경 변수**: .env 파일 기반 안전한 설정 관리

## 🚀 빠른 시작

### 전제 조건

- Python 3.14 이상
- pip 또는 uv (권장)
- Git

### 설치

#### 옵션 1: pip 사용 (표준)

```bash
# 저장소 클론
git clone https://github.com/11e3/crypto-quant-system.git
cd crypto-quant-system

# 가상환경 생성 및 활성화 (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# 기본 설치 (실행 환경만)
pip install -e .

# 또는 개발 환경 전체 설치
pip install -e .[dev,analysis,docs,notebooks]
```

#### 옵션 2: uv 사용 (권장, 10배 빠름)

```bash
# uv 설치 (Windows PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# 저장소 클론
git clone https://github.com/11e3/crypto-quant-system.git
cd crypto-quant-system

# 의존성 동기화 (자동으로 venv 생성)
uv sync --all-extras
```

### 빠른 테스트

```bash
# 샘플 데이터로 백테스트 실행
crypto-quant backtest --demo

# 또는 실제 데이터 수집 후 백테스트
crypto-quant collect --tickers KRW-BTC --interval day --days 365
crypto-quant backtest --tickers KRW-BTC --interval day
```

### CLI 명령어

```bash
# 데이터 수집
crypto-quant collect --tickers KRW-BTC KRW-ETH --interval day --days 365

# 백테스트 실행
crypto-quant backtest \
    --tickers KRW-BTC KRW-ETH \
    --interval day \
    --strategy vanilla_vbo \
    --initial-capital 1000000 \
    --max-slots 4

# 파라미터 최적화
crypto-quant optimize \
    --tickers KRW-BTC \
    --interval day \
    --param-range short_sma:5:20:5

# 실시간 거래 봇 실행 (API 키 필요)
crypto-quant run-bot --config config/settings.yaml
```

### 실시간 거래 설정

1. **API 키 발급**: [Upbit API 관리](https://upbit.com/mypage/open_api_management) 페이지 접속
2. **환경 변수 설정**:

```bash
# .env 파일 생성
UPBIT_ACCESS_KEY=your-access-key-here
UPBIT_SECRET_KEY=your-secret-key-here
```

3. **봇 실행**:

```bash
crypto-quant run-bot
```

⚠️ **주의**: 실제 거래 전에 반드시 충분한 백테스팅과 Paper Trading으로 검증하세요!

## 📊 성능 결과

### 백테스트 결과 (VanillaVBO 전략)

**테스트 기간**: 2017-01-01 ~ 2025-01-06 (약 8년, 3,018일)
**테스트 종목**: KRW-BTC, KRW-ETH 등 주요 코인
**초기 자본**: 10,000,000 KRW

| 지표 | 값 | 설명 |
|------|------|------|
| **총 수익률** | 38,331.40% | 초기 자본 대비 383배 증가 |
| **CAGR** | 105.40% | 연평균 수익률 |
| **최대 낙폭 (MDD)** | 24.97% | 최대 손실 구간 |
| **Sharpe 비율** | 1.97 | 위험 대비 수익 (1.5+ 우수) |
| **Sortino 비율** | 3.12 | 하방 위험 대비 수익 |
| **Calmar 비율** | 4.22 | MDD 대비 수익 (3+ 우수) |
| **승률** | 36.03% | 전체 거래 중 수익 비율 |
| **총 거래 횟수** | 705 | 연평균 약 88회 |
| **수익 팩터** | 1.77 | 총 수익 / 총 손실 (1.5+ 우수) |

**핵심 인사이트**:
- ✅ 낮은 승률(36%)이지만 높은 수익률: 큰 추세를 잡아 손실보다 수익이 큼
- ✅ 우수한 위험 조정 수익: Sharpe 1.97, Calmar 4.22
- ✅ 관리 가능한 손실폭: MDD 24.97% (암호화폐 시장 치고 낮음)
- ⚠️ 과거 성과는 미래 결과를 보장하지 않음

---

## � 프로젝트 구조

```
crypto-quant-system/
├── src/                        # 메인 소스 코드
│   ├── backtester/            # 백테스팅 엔진
│   │   ├── engine.py          # 벡터화 백테스트 엔진
│   │   ├── report.py          # 성능 리포트 생성 (HTML/JSON)
│   │   ├── optimization.py     # 파라미터 최적화 (Grid/Random)
│   │   ├── bootstrap_analysis.py   # Bootstrap 신뢰구간
│   │   ├── permutation_test.py     # 통계적 유의성 검증
│   │   ├── walk_forward_auto.py    # Walk-Forward 분석
│   │   └── monte_carlo.py          # Monte Carlo 시뮬레이션
│   ├── execution/              # 실시간 거래 실행
│   │   ├── bot.py             # 메인 거래 봇 로직
│   │   ├── event_bus.py       # 이벤트 기반 아키텍처
│   │   ├── order_manager.py   # 주문 생성/추적/취소
│   │   ├── position_manager.py # 포지션 추적/관리
│   │   └── risk_manager.py    # 리스크 제어 (Stop Loss, Take Profit)
│   ├── strategies/            # 거래 전략 모듈
│   │   ├── base.py           # 전략 추상 클래스
│   │   ├── volatility_breakout/    # VBO 전략
│   │   │   ├── vbo.py        # VanillaVBO 구현
│   │   │   ├── conditions.py # 진입/청산 조건
│   │   │   └── filters.py    # 신호 필터
│   │   ├── momentum/          # 모멘텀 전략 (확장 가능)
│   │   └── mean_reversion/    # 평균회귀 전략 (확장 가능)
│   ├── data/                  # 데이터 수집 및 관리
│   │   ├── collector.py      # 거래소 데이터 수집 (Factory 패턴)
│   │   ├── cache.py          # 지표 캐싱 레이어
│   │   ├── upbit_source.py   # Upbit API 연동
│   │   └── indicators.py     # 기술적 지표 계산 (SMA, EMA 등)
│   ├── exchange/              # 거래소 API 추상화
│   │   ├── base.py           # Exchange 인터페이스
│   │   └── upbit.py          # Upbit 구현체
│   ├── portfolio/             # 포트폴리오 최적화
│   │   ├── optimizer.py      # MPT, Risk Parity, Kelly
│   │   └── rebalancer.py     # 리밸런싱 로직
│   ├── risk/                  # 리스크 관리
│   │   ├── metrics.py        # VaR, CVaR, Drawdown
│   │   └── position_sizer.py # 포지션 사이징
│   ├── config/                # 설정 관리
│   │   └── settings.py       # Pydantic 기반 설정
│   └── cli/                   # CLI 진입점
│       └── main.py           # crypto-quant 명령어
├── tests/                     # 테스트 스위트 (86.35% 커버리지)
│   ├── unit/                 # 단위 테스트
│   ├── integration/          # 통합 테스트
│   └── fixtures/             # 테스트 픽스쳐
├── docs/                      # 문서화
│   ├── guides/               # 사용자 가이드
│   ├── api/                  # API 레퍼런스 (Sphinx)
│   ├── architecture.md       # 아키텍처 설계
│   └── archive/              # 이전 버전 문서
├── examples/                  # 예제 스크립트
│   ├── basic_backtest.py     # 기본 백테스트
│   ├── custom_strategy.py    # 커스텀 전략 예제
│   ├── portfolio_optimization.py  # 포트폴리오 최적화
│   └── live_trading_simulator.py  # Paper Trading
├── notebooks/                 # Jupyter 노트북
│   ├── 01-Backtesting-Case-Study.ipynb
│   ├── 02-Portfolio-Optimization.ipynb
│   └── 03-Live-Trading-Analysis.ipynb
├── scripts/                   # 유틸리티 스크립트
│   ├── tools/                # 개발 도구
│   ├── backtest/             # 백테스트 스크립트
│   └── archive/              # 레거시 스크립트
├── deploy/                    # 배포 설정
│   ├── Dockerfile            # Docker 이미지
│   ├── docker-compose.yml    # 서비스 구성
│   └── README.md             # 배포 가이드
├── data/                      # 데이터 저장소 (gitignore)
│   ├── raw/                  # 원본 OHLCV 데이터 (parquet)
│   └── processed/            # 처리된 데이터
├── reports/                   # 백테스트 결과 HTML
├── config/                    # 설정 파일 예제
│   ├── settings.yaml.example
│   └── monitoring.yaml.example
├── pyproject.toml            # 프로젝트 메타데이터 & 의존성
├── noxfile.py                # 테스트 자동화
├── Makefile                  # 편의 명령어
└── README.md                 # 이 문서
```

### 핵심 디렉토리 설명

- **src/**: 모든 프로덕션 코드 (96.7% strict type coverage)
- **tests/**: 단위/통합 테스트 (86.35% 커버리지)
- **docs/**: Sphinx 문서, 아키텍처 가이드
- **examples/**: 10개 예제 스크립트 (바로 실행 가능)
- **notebooks/**: 3개 튜토리얼 노트북 (학습용)
- **deploy/**: Docker 배포 환경 (1분 내 구동)

## 📚 문서

### 📖 가이드
- [설치 가이드](docs/DEPENDENCY_INSTALLATION_GUIDE.md) - Python, uv, 의존성 설치
- [시작 가이드](docs/guides/getting_started.md) - 첫 백테스트 실행
- [전략 개발](docs/guides/strategy_customization.md) - 커스텀 전략 작성법
- [설정 가이드](docs/guides/configuration.md) - YAML/환경변수 설정

### 🏗️ 아키텍처
- [시스템 아키텍처](docs/architecture.md) - 설계 원칙, 패턴, 데이터 흐름
- [타입 체킹 가이드](docs/TYPE_CHECKING.md) - MyPy strict 사용법

### 📋 개발
- [기여 가이드](CONTRIBUTING.md) - PR 프로세스, 코드 스타일
- [보안 가이드](SECURITY.md) - API 키 관리, 취약점 신고
- [라이선스](LICENSE) - MIT 라이선스

### 📓 Jupyter 노트북

실전 예제로 배우는 학습 자료:

1. **[백테스팅 케이스 스터디](notebooks/01-Backtesting-Case-Study.ipynb)**
   - VBO 전략 실행 및 분석
   - 성능 지표 해석 (Sharpe, Sortino, MDD)
   - 자산 곡선, 드로우다운 시각화

2. **[포트폴리오 최적화](notebooks/02-Portfolio-Optimization.ipynb)**
   - MPT vs Risk Parity vs Kelly Criterion
   - 효율적 변경선(Efficient Frontier)
   - 거래비용 반영 최적화

3. **[실시간 거래 분석](notebooks/03-Live-Trading-Analysis.ipynb)**
   - Paper Trading 시뮬레이션
   - 리스크 관리 체크리스트
   - 실전 거래 준비사항

## 🧪 테스트 및 코드 품질

### 테스트 실행

```bash
# 전체 테스트 (pytest)
make test

# 커버리지 리포트 생성
uv run pytest --cov=src --cov-report=html
# 결과: htmlcov/index.html

# 특정 모듈만 테스트
uv run pytest tests/unit/test_backtester/

# nox를 통한 자동화 테스트
nox  # 전체 세션 (lint, type, test)
nox -s test  # 테스트만
nox -s lint  # 린트만
```

### 타입 체크 (MyPy)

```bash
# 전체 타입 체크 (strict 모드)
mypy src tests

# 또는 nox 사용
nox -s type

# 결과: 87/90 모듈 strict 통과 (96.7%)
```

**strict 모드 적용 파일**:
- ✅ src/backtester/*.py (report, bootstrap, permutation 등)
- ✅ src/data/*.py (collector, cache, indicators)
- ✅ src/strategies/*.py (vbo, conditions)
- ✅ src/risk/*.py (metrics, kelly, trade_cost)
- ⏸️ src/backtester/engine.py (pandas 복잡성으로 유예)

### 코드 품질 (Ruff)

```bash
# 린트 & 포맷 검사
ruff check src tests
ruff format --check src tests

# 자동 수정
ruff check --fix src tests
ruff format src tests

# 또는 make 사용
make lint
make format
```

### Pre-commit Hooks

```bash
# 설치 (최초 1회)
pre-commit install

# 수동 실행
pre-commit run --all-files

# 이후 git commit 시 자동 실행:
# - Ruff 린트/포맷
# - MyPy 타입 체크
# - 테스트 (선택적)
```

### CI/CD

GitHub Actions 자동 실행:
- ✅ **Lint**: Ruff 코드 스타일 검사
- ✅ **Type**: MyPy strict 타입 체크
- ✅ **Test**: pytest 전체 테스트 + 커버리지
- ✅ **Security**: CodeQL 보안 스캔
- ✅ **Docs**: Sphinx 문서 빌드

## 🚢 배포

### Docker 배포 (권장)

```bash
cd deploy

# 이미지 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f bot

# 중단
docker-compose down
```

**docker-compose.yml 구성**:
- 봇 컨테이너 (실시간 거래)
- 환경 변수 파일 (.env)
- 데이터 볼륨 마운트

### 수동 배포

```bash
# 1. 서버에서 저장소 클론
git clone https://github.com/11e3/crypto-quant-system.git
cd crypto-quant-system

# 2. 의존성 설치
uv sync

# 3. 환경 변수 설정
cp config/settings.yaml.example config/settings.yaml
# settings.yaml 편집 (API 키 입력)

# 4. 봇 실행
nohup crypto-quant run-bot > bot.log 2>&1 &
```

### GCP/AWS 배포

자세한 클라우드 배포 가이드는 [deploy/README.md](deploy/README.md) 참조

## 🤝 기여하기

기여를 환영합니다! 다음 프로세스를 따라주세요:

### 기여 절차

1. **저장소 포크**: GitHub에서 Fork 버튼 클릭
2. **브랜치 생성**: `git checkout -b feature/amazing-feature`
3. **개발 환경 설정**:
   ```bash
   uv sync --all-extras
   pre-commit install
   ```
4. **코드 작성**: 기능 구현 또는 버그 수정
5. **테스트 작성**: `tests/` 디렉토리에 단위 테스트 추가
6. **품질 검사**:
   ```bash
   make lint    # Ruff 린트
   make test    # pytest 테스트
   mypy src     # 타입 체크
   ```
7. **커밋**: `git commit -m 'feat: add amazing feature'`
   - 커밋 메시지는 [Conventional Commits](https://www.conventionalcommits.org/) 따르기
8. **푸시**: `git push origin feature/amazing-feature`
9. **PR 생성**: GitHub에서 Pull Request 오픈

### 코드 스타일

- **타입 힌트**: 모든 함수에 타입 힌트 필수
- **Docstring**: Google 스타일 독스트링
- **테스트**: 새 기능은 80% 이상 커버리지 유지
- **Ruff**: 자동 포매팅 및 린트 준수
- **MyPy**: strict 모드 통과

자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md) 참조

## ⚠️ 면책 조항 및 위험 공고

**이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다.**

### 주요 위험

- 🔴 **자본 손실 위험**: 투자한 자본을 **완전히 잃을 수 있습니다**
- 🔴 **극변동성**: 암호화폐는 하루에 20-30% 이상 변동 가능
- 🔴 **보장 없음**: 과거 성과는 미래 결과를 **절대 보장하지 않습니다**
- 🔴 **시스템 위험**: 소프트웨어 버그, API 장애, 거래소 문제 가능성

### 필수 읽기

사용 전에 반드시 읽어주세요:
- 📖 [면책조항 (DISCLAIMER.md)](DISCLAIMER.md) - 상세 위험 경고
- 📖 [데이터 사용 정책 (DATA_USAGE_POLICY.md)](DATA_USAGE_POLICY.md) - 개인정보 보호
- 📖 [보안 정책 (SECURITY.md)](SECURITY.md) - API 키 보안

### 책임 사항

**사용자 책임**:
- ✅ 실거래 전 충분한 Paper Trading 테스트
- ✅ 여유 자금(손실 가능 금액)으로만 거래
- ✅ 정기적 시스템 성과 모니터링
- ✅ 법적/세금 책임 준수

**개발자/기여자는 책임지지 않음**:
- ❌ 금융 손실
- ❌ 거래소 문제로 인한 피해
- ❌ 소프트웨어 버그로 인한 오류 거래
- ❌ 세금/규제 위반

## 📄 라이선스

MIT 라이선스 - [LICENSE](LICENSE) 참조

## 🔐 보안 및 준수

### 보안 기능

- ✅ **환경 변수 설정**: API 키는 코드에 포함 안 됨
- ✅ **타입 안전성**: MyPy strict로 타입 오류 방지
- ✅ **코드 분석**: CodeQL 보안 취약점 자동 스캔
- ✅ **데이터 보호**: 거래 기록 암호화 권장

### 보안 취약점 신고

- ❌ 공개 이슈 생성 금지
- ✅ [GitHub Security Advisory](https://github.com/11e3/crypto-quant-system/security/advisories/new) 사용
- ✅ 또는 메인테이너 직접 연락

자세한 내용: [SECURITY.md](SECURITY.md)

### 준법 및 세금

**법률 준수**:
- 🇰🇷 한국: 특정금융정보법
- 🇺🇸 미국: FinCEN, CFTC
- 🇪🇺 유럽: MiFID II, GDPR
- 📋 국제: 해당 국가 암호화폐 규정

**세금**:
- 💰 거래 수익은 과세 대상
- 📊 거래 기록 7년 보관 권장
- ⚠️ 세금 신고는 사용자 책임

## 🙏 감사의 말

- [pyupbit](https://github.com/sharebook-kr/pyupbit) - Upbit API 통합
- [pandas-dev](https://github.com/pandas-dev/pandas) - 데이터 분석
- [numpy](https://github.com/numpy/numpy) - 수치 연산
- [pydantic](https://github.com/pydantic/pydantic) - 데이터 검증
- [ruff](https://github.com/astral-sh/ruff) - 초고속 린터
- Python 오픈소스 커뮤니티

## 📧 문의 및 지원

- **버그 리포트**: [GitHub Issues](https://github.com/11e3/crypto-quant-system/issues)
- **기능 요청**: [GitHub Discussions](https://github.com/11e3/crypto-quant-system/discussions)
- **보안 이슈**: [Security Advisory](https://github.com/11e3/crypto-quant-system/security/advisories/new)

## 🗺️ 로드맵

### 현재 상태 (v0.1.0)

- ✅ VanillaVBO 전략 구현
- ✅ 백테스팅 엔진 (벡터화)
- ✅ 실시간 거래 봇
- ✅ 96.7% MyPy strict 커버리지
- ✅ 86.35% 테스트 커버리지
- ✅ Docker 배포

### 계획 중

- 🔜 **v0.2.0**: 다중 거래소 지원 (Binance, Bithumb)
- 🔜 **v0.3.0**: 머신러닝 기반 전략 (LSTM, Transformer)
- 🔜 **v0.4.0**: 웹 대시보드 (Streamlit/Dash)
- 🔜 **v0.5.0**: 고빈도 거래 (HFT) 지원
- 🔜 **v1.0.0**: 프로덕션 안정화

---

<div align="center">

**정량적 거래를 위해 ❤️로 만들었습니다**

⭐ 유용하다면 Star를 눌러주세요!

[보고 싶은 내용이 있나요?](https://github.com/11e3/crypto-quant-system/discussions) • [버그 발견?](https://github.com/11e3/crypto-quant-system/issues) • [기여하기](CONTRIBUTING.md)

</div>

