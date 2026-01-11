# Streamlit 백테스팅 웹 UI

이벤트 드리븐 백테스팅 엔진 기반 웹 인터페이스입니다.

## 🚀 실행 방법

### 개발 모드 (Hot Reload)

```bash
# 의존성 설치
uv sync --extra web

# 앱 실행
uv run streamlit run src/web/app.py --server.runOnSave true
```

### 프로덕션 모드

```bash
uv run streamlit run src/web/app.py --server.port 8501 --server.headless true
```

## 📁 디렉토리 구조

```
src/web/
├── app.py                  # 메인 진입점
├── config/                 # 설정 모듈
│   ├── __init__.py
│   └── app_settings.py     # Pydantic Settings
├── pages/                  # 멀티 페이지
│   ├── __init__.py
│   ├── backtest.py         # 백테스트 페이지 (Phase 2)
│   ├── optimization.py     # 최적화 페이지 (Phase 4)
│   └── analysis.py         # 고급 분석 페이지 (Phase 5)
├── components/             # UI 컴포넌트
│   ├── sidebar/            # 사이드바 컴포넌트 (Phase 2)
│   ├── metrics/            # 메트릭 표시 컴포넌트 (Phase 3)
│   └── charts/             # 차트 컴포넌트 (Phase 3)
├── services/               # 비즈니스 로직
│   ├── __init__.py
│   ├── parameter_models.py # 데이터 모델
│   ├── strategy_registry.py # 전략 레지스트리
│   ├── backtest_runner.py  # 백테스트 실행 (Phase 2)
│   ├── data_loader.py      # 데이터 로딩 (Phase 2)
│   └── metrics_calculator.py # 메트릭 계산 (Phase 3)
└── utils/                  # 유틸리티
    ├── __init__.py
    ├── formatters.py       # 숫자/퍼센트 포맷터 (Phase 2)
    └── validators.py       # 입력 검증 (Phase 2)
```

## 🎯 개발 현황

### ✅ Phase 1 완료 (기초 인프라)

- [x] 기본 디렉토리 구조 생성
- [x] Streamlit 앱 진입점 구현
- [x] 멀티 페이지 구조 설정
- [x] Pydantic Settings 기반 앱 설정
- [x] ParameterSpec, StrategyInfo 데이터 모델
- [x] StrategyRegistry 자동 전략 감지 서비스
- [x] pyproject.toml에 web 의존성 추가

### ✅ Phase 2 완료 (사이드바 컴포넌트)

- [x] 날짜 설정 컴포넌트 (시작일/종료일)
- [x] 거래 설정 컴포넌트 (인터벌, 수수료, 슬리피지)
- [x] 전략 선택기 + 동적 파라미터 편집
- [x] 자산 선택기 (멀티 선택)
- [x] 백테스트 실행 서비스 (EventDrivenBacktestEngine)
- [x] 데이터 로더 서비스 (Upbit OHLCV)
- [x] 백테스트 페이지 통합 (사이드바 + 결과 표시)
- [x] 기본 메트릭 표시 (CAGR, MDD, Sharpe 등)

### 🚧 Phase 3 진행 중 (차트 및 고급 메트릭)

- [ ] Plotly 수익률 곡선 (인터랙티브)
- [ ] 언더워터 곡선 (드로다운)
- [ ] 월별 수익률 히트맵
- [ ] 연도별 수익률 막대그래프
- [ ] 확장 메트릭 (Sortino, Calmar, VaR, CVaR 등)

### 📅 Phase 4-5 계획

- [ ] 파라미터 최적화 페이지 (Grid/Random Search)
- [ ] Walk-Forward Analysis
- [ ] 순열 검정 (과적합 검증)
- [ ] Monte Carlo 시뮬레이션

## 🧪 테스트

```bash
# 전략 레지스트리 테스트
uv run python -c "
from src.web.services import StrategyRegistry
registry = StrategyRegistry()
strategies = registry.list_strategies()
for s in strategies:
    print(f'{s.name}: {len(s.parameters)} parameters')
"
```

## 📝 환경 변수

`.env` 파일에서 설정 가능:

```env
# Web 서버 설정
WEB_SERVER_PORT=8501
WEB_SERVER_ADDRESS=localhost
WEB_SERVER_HEADLESS=false

# 캐시 설정
WEB_CACHE_TTL=3600
WEB_ENABLE_CACHING=true

# UI 설정
WEB_DEFAULT_THEME=light
WEB_SHOW_DEBUG_INFO=false

# 백테스트 기본값
WEB_MAX_PARALLEL_WORKERS=4
WEB_DEFAULT_INITIAL_CAPITAL=10000000.0
WEB_DEFAULT_FEE_RATE=0.0005
WEB_DEFAULT_SLIPPAGE_RATE=0.0005
```

## 🎨 주요 기능 (예정)

### 📈 백테스트 (Phase 2-3)
- 동적 파라미터 설정
- 다중 자산 백테스트
- 실시간 메트릭 표시
- 인터랙티브 차트

### 🔧 파라미터 최적화 (Phase 4)
- Grid Search
- Random Search
- 병렬 처리

### 📊 고급 분석 (Phase 5)
- Walk-Forward Analysis
- 순열 검정
- VaR/CVaR

## 📚 참고 문서

- [전체 계획서](../../docs/planning/streamlit-backtest-ui-plan.md)
- [Phase 1 완료 보고](../../docs/planning/streamlit-backtest-ui-plan.md#phase-1-기초-인프라-week-1)
