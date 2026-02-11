# Streamlit 백테스팅 UI 웹페이지 제작 계획

## 📋 프로젝트 개요

**목적**: 이벤트 드리븐 백테스팅 엔진(`EventDrivenBacktestEngine`)을 활용한 풀스택 백테스팅 웹 인터페이스 구축

**기술 스택**:
- Frontend: Streamlit
- Backend: 기존 `src.backtester` 모듈 활용
- Charts: Plotly (인터랙티브), Matplotlib (정적)
- Data: Pandas, NumPy

---

## 🏗️ 아키텍처 설계

### 디렉토리 구조

```
src/
└── web/
    ├── __init__.py
    ├── app.py                      # Streamlit 진입점
    ├── config/
    │   ├── __init__.py
    │   └── app_settings.py         # 앱 설정 (Pydantic Settings)
    ├── pages/
    │   ├── __init__.py
    │   ├── backtest.py             # 메인 백테스트 페이지
    │   ├── optimization.py         # 파라미터 최적화 페이지
    │   └── analysis.py             # 고급 분석 페이지 (WFA, 순열검정 등)
    ├── components/
    │   ├── __init__.py
    │   ├── sidebar/
    │   │   ├── __init__.py
    │   │   ├── date_config.py      # 시작일/종료일 설정
    │   │   ├── strategy_selector.py # 전략 선택 (레지스트리 기반)
    │   │   ├── parameter_editor.py  # 동적 파라미터 편집기
    │   │   ├── asset_selector.py    # 자산군 선택
    │   │   └── trading_config.py    # 수수료/슬리피지/인터벌 설정
    │   ├── metrics/
    │   │   ├── __init__.py
    │   │   ├── summary_cards.py     # 요약 메트릭 카드
    │   │   ├── detailed_metrics.py  # 상세 메트릭 테이블
    │   │   └── risk_metrics.py      # 리스크 메트릭 (VaR, CVaR 등)
    │   └── charts/
    │       ├── __init__.py
    │       ├── equity_curve.py      # 수익률 곡선
    │       ├── underwater.py        # 언더워터 곡선 (드로다운)
    │       ├── monthly_heatmap.py   # 월별 수익률 히트맵
    │       └── yearly_bar.py        # 연도별 수익률 막대그래프
    ├── services/
    │   ├── __init__.py
    │   ├── strategy_registry.py     # 전략 레지스트리 서비스
    │   ├── backtest_runner.py       # 백테스트 실행 서비스
    │   ├── data_loader.py           # 데이터 로딩 서비스
    │   └── metrics_calculator.py    # 메트릭 계산 서비스
    └── utils/
        ├── __init__.py
        ├── formatters.py            # 숫자/퍼센트 포맷터
        └── validators.py            # 입력 검증 유틸
```

---

## 🎨 UI/UX 설계

### 1. 사이드바 (Configuration Panel)

```
┌─────────────────────────────────────┐
│          📊 Backtest Config         │
├─────────────────────────────────────┤
│ 📅 기간 설정                         │
│ ├─ 시작일: [2023-01-01] 📆          │
│ └─ 종료일: [2024-12-31] 📆          │
├─────────────────────────────────────┤
│ ⏱️ 캔들 인터벌                       │
│ └─ [day ▼] (day/minute240/week)    │
├─────────────────────────────────────┤
│ 💰 거래 비용                         │
│ ├─ 수수료: [0.05] %                 │
│ └─ 슬리피지: [0.05] %               │
├─────────────────────────────────────┤
│ 📈 전략 선택                         │
│ └─ [VanillaVBO ▼]                   │
│     • MomentumStrategy              │
│     • MeanReversionStrategy         │
│     • PairTradingStrategy           │
│     • ORBStrategy                   │
├─────────────────────────────────────┤
│ 🎛️ 전략 파라미터 (동적)             │
│ ├─ sma_period: [4] 슬라이더         │
│ ├─ trend_sma_period: [8]            │
│ ├─ use_trend_filter: [✓]            │
│ └─ use_noise_filter: [✓]            │
├─────────────────────────────────────┤
│ 🎯 필터 설정                         │
│ ├─ 추세 필터: [✓]                   │
│ └─ 노이즈 필터: [✓]                 │
├─────────────────────────────────────┤
│ 🪙 자산 선택                         │
│ ├─ [✓] KRW-BTC                      │
│ ├─ [✓] KRW-ETH                      │
│ ├─ [ ] KRW-XRP                      │
│ └─ [ ] KRW-TRX                      │
├─────────────────────────────────────┤
│ ⚙️ 고급 설정                         │
│ ├─ 초기자본: [10,000,000] KRW       │
│ ├─ 최대 슬롯: [4]                   │
│ ├─ 포지션 사이징: [equal ▼]         │
│ ├─ 스탑로스: [5.0] %                │
│ ├─ 테이크프로핏: [10.0] %           │
│ └─ 트레일링 스탑: [5.0] %           │
├─────────────────────────────────────┤
│        [🚀 백테스트 실행]            │
│        [🔧 파라미터 최적화]          │
└─────────────────────────────────────┘
```

### 2. 메인 화면 (Results Dashboard)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        📊 백테스트 결과 대시보드                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │ 총 수익률    │ │    CAGR    │ │    MDD     │ │   Sharpe   │            │
│ │   +45.2%    │ │   +18.5%   │ │   -12.3%   │ │    1.85    │            │
│ │    ▲        │ │     ▲      │ │     ▼      │ │     ▲      │            │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │  Sortino    │ │   Calmar   │ │  Win Rate  │ │ Num Trades │            │
│ │    2.12     │ │    1.50    │ │   58.3%    │ │    156     │            │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                              📈 수익률 곡선                                  │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ [Plotly 인터랙티브 차트]                                                 │ │
│ │ - 포트폴리오 가치 곡선                                                   │ │
│ │ - 벤치마크 (선택적)                                                      │ │
│ │ - 호버 시 상세 정보 표시                                                 │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                              📉 언더워터 곡선                                │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ [드로다운 영역 차트]                                                     │ │
│ │ - 0%에서 시작하여 아래로 표시                                            │ │
│ │ - 최대 낙폭 구간 강조                                                    │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Tab: 월별 히트맵]  [Tab: 연도별 수익률]  [Tab: 상세 메트릭]  [Tab: 거래내역] │
├─────────────────────────────────────────────────────────────────────────────┤
│ 📅 월별 수익률 히트맵                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │      Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   │ │
│ │ 2023 +2.1  -1.5  +4.2  +0.8  -2.3  +3.1  +1.2  -0.5  +2.8  +1.5  +0.9  │ │
│ │ 2024 +3.2  +1.8  -0.3  +2.5  +1.1  -1.8  +4.5  +2.1  -0.7  +1.9  +2.3  │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│ 📊 연도별 수익률                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ [막대 그래프]                                                            │ │
│ │ 2023: ████████████████████ +18.5%                                       │ │
│ │ 2024: ████████████████████████████ +26.7%                               │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. 상세 메트릭 테이블

| 카테고리 | 메트릭 | 설명 | 값 |
|---------|--------|------|-----|
| **기본 정보** | 기간 | 백테스트 기간 | 2023-01-01 ~ 2024-12-31 |
| | 거래일수 | 총 거래 일수 | 730일 |
| **수익률** | 총 수익률 | 전체 기간 수익률 | +45.2% |
| | CAGR | 연환산 복리수익률 | +18.5% |
| **리스크** | MDD | 최대 낙폭 | -12.3% |
| | Sharpe Ratio | 샤프 비율 | 1.85 |
| | Sortino Ratio | 소르티노 비율 | 2.12 |
| | Calmar Ratio | 칼마 비율 | 1.50 |
| **거래 통계** | 총 거래수 | 전체 거래 횟수 | 156 |
| | 승률 | 수익 거래 비율 | 58.3% |
| | 평균 수익 거래 | 수익 거래 평균 수익률 | +3.2% |
| | 평균 손실 거래 | 손실 거래 평균 손실률 | -1.8% |
| **변동성** | 상방 변동성 | 양의 수익률 변동성 | 12.5% |
| | 하방 변동성 | 음의 수익률 변동성 | 8.3% |
| **통계적 검증** | Z-Score | 순열검정 Z점수 | 2.45 |
| | P-Value | 통계적 유의 수준 | 0.014 |
| **고급 분석** | WFA 효율성 | Walk-Forward 효율성 | 0.85 |
| | OOS 수익률 | Out-of-Sample 수익률 | +12.3% |
| **리스크 메트릭** | VaR (95%) | Value at Risk | -2.1% |
| | CVaR (95%) | Conditional VaR | -3.5% |

---

## 📦 구현 상세

### Phase 1: 기초 인프라 (Week 1)

#### 1.1 Streamlit 앱 기본 구조
```python
# src/web/app.py
import streamlit as st

st.set_page_config(
    page_title="Crypto Lab Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Multi-page 구조
pages = {
    "백테스트": "pages/backtest.py",
    "파라미터 최적화": "pages/optimization.py",
    "고급 분석": "pages/analysis.py",
}
```

#### 1.2 전략 레지스트리 서비스
```python
# src/web/services/strategy_registry.py
from typing import Protocol
from dataclasses import dataclass

@dataclass(frozen=True)
class StrategyInfo:
    """전략 메타데이터."""
    name: str
    class_name: str
    module_path: str
    parameters: dict[str, ParameterSpec]
    description: str

class StrategyRegistryProtocol(Protocol):
    """전략 레지스트리 인터페이스."""
    
    def list_strategies(self) -> list[StrategyInfo]: ...
    def get_strategy(self, name: str) -> type: ...
    def get_parameters(self, name: str) -> dict[str, ParameterSpec]: ...
```

#### 1.3 파라미터 스펙 정의
```python
@dataclass(frozen=True)
class ParameterSpec:
    """전략 파라미터 명세."""
    name: str
    type: Literal["int", "float", "bool", "choice"]
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    choices: list[Any] | None = None
    description: str = ""
```

### Phase 2: 사이드바 컴포넌트 (Week 2)

#### 2.1 날짜 설정 컴포넌트
```python
# src/web/components/sidebar/date_config.py
def render_date_config() -> tuple[date, date]:
    """날짜 범위 선택 UI."""
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작일",
            value=date.today() - timedelta(days=365),
            min_value=date(2017, 1, 1),
        )
    with col2:
        end_date = st.date_input(
            "종료일",
            value=date.today(),
            max_value=date.today(),
        )
    return start_date, end_date
```

#### 2.2 전략 선택 + 동적 파라미터
```python
# src/web/components/sidebar/strategy_selector.py
def render_strategy_selector(registry: StrategyRegistry) -> tuple[str, dict]:
    """전략 선택 및 파라미터 동적 렌더링."""
    strategies = registry.list_strategies()
    strategy_names = [s.name for s in strategies]
    
    selected = st.selectbox("전략 선택", strategy_names)
    
    # 선택된 전략의 파라미터 동적 렌더링
    params = registry.get_parameters(selected)
    param_values = {}
    
    st.subheader("🎛️ 전략 파라미터")
    for name, spec in params.items():
        param_values[name] = render_parameter_input(name, spec)
    
    return selected, param_values

def render_parameter_input(name: str, spec: ParameterSpec) -> Any:
    """파라미터 타입에 따른 입력 UI 렌더링."""
    match spec.type:
        case "int":
            return st.slider(
                name,
                min_value=int(spec.min_value or 1),
                max_value=int(spec.max_value or 100),
                value=int(spec.default),
                step=int(spec.step or 1),
                help=spec.description,
            )
        case "float":
            return st.number_input(
                name,
                min_value=spec.min_value,
                max_value=spec.max_value,
                value=float(spec.default),
                step=spec.step or 0.01,
                help=spec.description,
            )
        case "bool":
            return st.checkbox(name, value=spec.default, help=spec.description)
        case "choice":
            return st.selectbox(name, spec.choices, index=spec.choices.index(spec.default))
```

### Phase 3: 메트릭 및 차트 (Week 3)

#### 3.1 메트릭 계산 서비스
```python
# src/web/services/metrics_calculator.py
from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True)
class ExtendedMetrics:
    """확장 메트릭 모델."""
    # 기본
    total_return: Decimal
    cagr: Decimal
    mdd: Decimal
    
    # 리스크 조정
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 거래 통계
    total_trades: int
    win_rate: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: float
    
    # 변동성
    upside_volatility: float
    downside_volatility: float
    
    # 통계적 검증
    z_score: float | None
    p_value: float | None
    
    # 고급 분석
    wfa_efficiency: float | None
    oos_return: Decimal | None
    
    # 리스크
    var_95: Decimal
    cvar_95: Decimal
    
    # 연도별
    yearly_returns: dict[int, Decimal]
```

#### 3.2 Plotly 차트 컴포넌트
```python
# src/web/components/charts/equity_curve.py
import plotly.graph_objects as go

def render_equity_curve(
    dates: np.ndarray,
    equity: np.ndarray,
    benchmark: np.ndarray | None = None,
) -> None:
    """인터랙티브 수익률 곡선."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity,
        mode='lines',
        name='Portfolio',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Value: %{y:,.0f}<extra></extra>',
    ))
    
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=1, dash='dash'),
        ))
    
    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (KRW)',
        hovermode='x unified',
        template='plotly_white',
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

#### 3.3 월별 히트맵
```python
# src/web/components/charts/monthly_heatmap.py
import plotly.figure_factory as ff

def render_monthly_heatmap(monthly_returns: pd.DataFrame) -> None:
    """월별 수익률 히트맵."""
    # Pivot: rows=years, columns=months
    pivot = monthly_returns.pivot(index='year', columns='month', values='return')
    
    # 색상 스케일: 빨강(손실) - 흰색(0) - 녹색(수익)
    fig = ff.create_annotated_heatmap(
        z=pivot.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=pivot.index.tolist(),
        annotation_text=[[f'{v:.1f}%' for v in row] for row in pivot.values],
        colorscale='RdYlGn',
        showscale=True,
    )
    
    fig.update_layout(title='Monthly Returns Heatmap')
    st.plotly_chart(fig, use_container_width=True)
```

### Phase 4: 파라미터 최적화 (Week 4)

#### 4.1 최적화 페이지
```python
# src/web/pages/optimization.py
def render_optimization_page():
    """파라미터 최적화 페이지."""
    st.header("🔧 파라미터 최적화")
    
    # 최적화 설정
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox("최적화 방법", ["Grid Search", "Random Search"])
        metric = st.selectbox("최적화 메트릭", ["sharpe_ratio", "cagr", "calmar_ratio"])
    
    with col2:
        n_iter = st.number_input("반복 횟수", 10, 1000, 100) if method == "Random Search" else None
        n_workers = st.number_input("병렬 워커 수", 1, 8, 4)
    
    # 파라미터 범위 설정
    st.subheader("파라미터 범위")
    param_ranges = {}
    for name, spec in strategy_params.items():
        if spec.type in ("int", "float"):
            col1, col2, col3 = st.columns(3)
            with col1:
                min_val = st.number_input(f"{name} (Min)", value=spec.min_value)
            with col2:
                max_val = st.number_input(f"{name} (Max)", value=spec.max_value)
            with col3:
                step = st.number_input(f"{name} (Step)", value=spec.step)
            param_ranges[name] = list(range(int(min_val), int(max_val) + 1, int(step)))
    
    # 최적화 실행
    if st.button("🚀 최적화 시작"):
        with st.spinner("최적화 진행 중..."):
            result = run_optimization(param_ranges, method, metric, n_workers)
        
        display_optimization_results(result)
```

### Phase 5: 고급 분석 (Week 5)

#### 5.1 Walk-Forward Analysis
```python
# src/web/pages/analysis.py
def render_wfa_section():
    """Walk-Forward Analysis 섹션."""
    st.subheader("📊 Walk-Forward Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        opt_days = st.number_input("최적화 기간 (일)", 180, 730, 365)
    with col2:
        test_days = st.number_input("테스트 기간 (일)", 30, 180, 90)
    with col3:
        step_days = st.number_input("스텝 (일)", 30, 180, 90)
    
    if st.button("WFA 실행"):
        result = run_walk_forward_analysis(...)
        
        # 결과 시각화
        display_wfa_results(result)
```

#### 5.2 순열 검정
```python
def render_permutation_test():
    """순열 검정 섹션."""
    st.subheader("🎲 Permutation Test (과적합 검증)")
    
    n_shuffles = st.slider("셔플 횟수", 100, 5000, 1000)
    
    if st.button("순열 검정 실행"):
        with st.spinner(f"{n_shuffles}회 순열 검정 진행 중..."):
            result = run_permutation_test(n_shuffles)
        
        # 결과 표시
        col1, col2, col3 = st.columns(3)
        col1.metric("Z-Score", f"{result.z_score:.2f}")
        col2.metric("P-Value", f"{result.p_value:.4f}")
        col3.metric("통계적 유의성", "✅ 유의" if result.is_significant else "❌ 무의미")
        
        # 분포 차트
        render_permutation_distribution(result)
```

---

## 🔧 기술적 구현 세부사항

### 전략 레지스트리 자동 감지

```python
# src/web/services/strategy_registry.py
import inspect
from importlib import import_module
from pathlib import Path

class StrategyRegistry:
    """전략 자동 감지 및 등록."""
    
    STRATEGY_MODULES = [
        "src.strategies.volatility_breakout",
        "src.strategies.momentum",
        "src.strategies.mean_reversion",
        "src.strategies.pair_trading",
        "src.strategies.opening_range_breakout",
    ]
    
    def __init__(self):
        self._strategies: dict[str, StrategyInfo] = {}
        self._discover_strategies()
    
    def _discover_strategies(self) -> None:
        """모든 전략 모듈에서 Strategy 서브클래스 탐색."""
        for module_path in self.STRATEGY_MODULES:
            module = import_module(module_path)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Strategy) and obj is not Strategy:
                    self._register_strategy(name, obj, module_path)
    
    def _extract_parameters(self, cls: type) -> dict[str, ParameterSpec]:
        """__init__ 시그니처에서 파라미터 추출."""
        sig = inspect.signature(cls.__init__)
        params = {}
        
        for name, param in sig.parameters.items():
            if name in ('self', 'name'):
                continue
            
            # 타입 힌트 분석
            annotation = param.annotation
            default = param.default if param.default != inspect.Parameter.empty else None
            
            spec = self._infer_parameter_spec(name, annotation, default)
            if spec:
                params[name] = spec
        
        return params
```

### 캐싱 전략

```python
# Streamlit 캐싱 활용
@st.cache_data(ttl=3600)
def load_ticker_data(ticker: str, interval: str, start: date, end: date) -> pd.DataFrame:
    """OHLCV 데이터 로딩 (1시간 캐시)."""
    ...

@st.cache_resource
def get_strategy_registry() -> StrategyRegistry:
    """전략 레지스트리 싱글톤."""
    return StrategyRegistry()

# 백테스트 결과 세션 스테이트 저장
if 'backtest_result' not in st.session_state:
    st.session_state.backtest_result = None
```

### 에러 핸들링

```python
class BacktestError(Exception):
    """백테스트 실행 에러."""
    pass

def run_backtest_with_error_handling(config: BacktestConfig, strategy: Strategy):
    """에러 핸들링이 포함된 백테스트 실행."""
    try:
        with st.spinner("백테스트 실행 중..."):
            result = engine.run(strategy, data_files, start_date, end_date)
        return result
    except ValueError as e:
        st.error(f"⚠️ 설정 오류: {e}")
    except FileNotFoundError as e:
        st.error(f"📁 데이터 파일 없음: {e}")
    except Exception as e:
        st.error(f"❌ 백테스트 실패: {e}")
        logger.exception("Backtest failed")
    return None
```

---

## 📅 개발 일정

| Phase | 기간 | 주요 작업 |
|-------|------|----------|
| **Phase 1** | Week 1 | 기본 구조, 전략 레지스트리, 의존성 설정 |
| **Phase 2** | Week 2 | 사이드바 컴포넌트 (날짜, 전략, 파라미터, 자산) |
| **Phase 3** | Week 3 | 메트릭 계산, 차트 컴포넌트 (Plotly) |
| **Phase 4** | Week 4 | 파라미터 최적화 페이지 |
| **Phase 5** | Week 5 | 고급 분석 (WFA, 순열검정, VaR) |
| **Phase 6** | Week 6 | 테스트, 문서화, 최적화 |

---

## 📦 의존성 추가

```toml
# pyproject.toml [project.optional-dependencies]
web = [
    "streamlit>=1.30.0",
    "plotly>=5.18.0",
    "watchdog>=3.0.0",  # Streamlit hot reload
]
```

---

## 🚀 실행 방법

```bash
# 개발 모드
uv run streamlit run src/web/app.py --server.runOnSave true

# 프로덕션 모드
uv run streamlit run src/web/app.py --server.port 8501 --server.headless true
```

---

## ✅ 체크리스트

### 사이드바 기능
- [ ] 시작일/종료일 선택
- [ ] 캔들 인터벌 선택 (day, minute240, week, etc.)
- [ ] 수수료율 입력
- [ ] 슬리피지율 입력
- [ ] 전략 선택 (레지스트리 자동 감지)
- [ ] 동적 파라미터 편집
- [ ] 필터 설정 (추세, 노이즈)
- [ ] 자산군 멀티 선택
- [ ] 고급 설정 (초기자본, 슬롯, 스탑로스 등)

### 메트릭 표시
- [ ] 기본 메트릭 (기간, 총수익률, CAGR, MDD)
- [ ] 리스크 조정 메트릭 (Sharpe, Sortino, Calmar)
- [ ] 거래 통계 (승률, 거래수, 평균수익/손실)
- [ ] 변동성 메트릭 (상방/하방)
- [ ] 통계적 검증 (Z-score, P-value)
- [ ] 고급 분석 (WFA, OOS)
- [ ] 리스크 메트릭 (VaR, CVaR)
- [ ] 연도별 수익률

### 차트
- [ ] 수익률 곡선 (Plotly 인터랙티브)
- [ ] 언더워터 곡선 (드로다운)
- [ ] 월별 수익률 히트맵
- [ ] 연도별 수익률 막대그래프

### 추가 기능
- [ ] 파라미터 최적화 (Grid/Random Search)
- [ ] Walk-Forward Analysis
- [ ] 순열 검정
- [ ] 결과 내보내기 (CSV, HTML)
