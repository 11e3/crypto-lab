# Streamlit Backtesting Web UI

Event-driven backtesting engine with an interactive web interface built on Streamlit.

## 🚀 Quick Start

### Development Mode (Hot Reload)

```bash
# Install dependencies
uv sync --extra web

# Run the app
uv run streamlit run src/web/app.py --server.runOnSave true
```

### Production Mode

```bash
uv run streamlit run src/web/app.py --server.port 8501 --server.headless true
```

## 📁 Directory Structure

```
src/web/
├── app.py                  # Main entry point
├── config/                 # Configuration module
│   ├── __init__.py
│   └── app_settings.py     # Pydantic Settings
├── pages/                  # Multi-page structure
│   ├── __init__.py
│   ├── backtest.py         # Backtest page (Phase 2)
│   ├── optimization.py     # Optimization page (Phase 4)
│   └── analysis.py         # Advanced analysis page (Phase 5)
├── components/             # UI components
│   ├── sidebar/            # Sidebar components (Phase 2)
│   │   ├── asset_selector.py
│   │   ├── date_config.py
│   │   ├── strategy_selector.py
│   │   └── trading_config.py
│   ├── metrics/            # Metrics display components (Phase 3)
│   │   └── metrics_display.py
│   └── charts/             # Chart components (Phase 3)
│       ├── equity_curve.py
│       ├── monthly_heatmap.py
│       ├── underwater.py
│       └── yearly_bar.py
├── services/               # Business logic
│   ├── __init__.py
│   ├── parameter_models.py # Data models
│   ├── strategy_registry.py # Strategy registry
│   ├── backtest_runner.py  # Backtest execution (Phase 2)
│   ├── data_loader.py      # Data loading (Phase 2)
│   └── metrics_calculator.py # Metrics calculation (Phase 3)
└── utils/                  # Utilities
    ├── __init__.py
    ├── formatters.py       # Number/percentage formatters (Phase 2)
    └── validators.py       # Input validation (Phase 2)
```

## 🎯 Development Status

### ✅ Phase 1 Complete (Infrastructure)

- [x] Basic directory structure
- [x] Streamlit app entry point
- [x] Multi-page architecture
- [x] Pydantic Settings configuration
- [x] ParameterSpec, StrategyInfo data models
- [x] StrategyRegistry auto-detection service
- [x] Web dependencies in pyproject.toml

### ✅ Phase 2 Complete (Sidebar Components)

- [x] Date settings component (start/end date)
- [x] Trading settings component (interval, fee, slippage)
- [x] Strategy selector with dynamic parameter editor
- [x] Asset selector (multi-select)
- [x] Backtest runner service (EventDrivenBacktestEngine)
- [x] Data loader service (Upbit OHLCV)
- [x] Backtest page integration (sidebar + results)
- [x] Basic metrics display (CAGR, MDD, Sharpe, etc.)

### ✅ Phase 3 Complete (Charts & Advanced Metrics)

- [x] Plotly equity curve (interactive)
- [x] Underwater curve (drawdown visualization)
- [x] Monthly returns heatmap
- [x] Yearly returns bar chart
- [x] Extended metrics (Sortino, Calmar, VaR, CVaR, etc.)
- [x] Statistical significance display

### 📅 Phase 4-5 Planned

- [ ] Parameter optimization page (Grid/Random Search)
- [ ] Walk-Forward Analysis
- [ ] Permutation test (overfitting detection)
- [ ] Monte Carlo simulation

## 🧪 Testing

```bash
# Test strategy registry
uv run python -c "
from src.web.services import StrategyRegistry
registry = StrategyRegistry()
strategies = registry.list_strategies()
for s in strategies:
    print(f'{s.name}: {len(s.parameters)} parameters')
"

# Run web app tests
uv run pytest tests/unit/test_web/ -v
```

## 📝 Environment Variables

Configure via `.env` file:

```env
# Web server settings
WEB_SERVER_PORT=8501
WEB_SERVER_ADDRESS=localhost
WEB_SERVER_HEADLESS=false

# Cache settings
WEB_CACHE_TTL=3600
WEB_ENABLE_CACHING=true

# UI settings
WEB_DEFAULT_THEME=light
WEB_SHOW_DEBUG_INFO=false

# Backtest defaults
WEB_MAX_PARALLEL_WORKERS=4
WEB_DEFAULT_INITIAL_CAPITAL=10000000.0
WEB_DEFAULT_FEE_RATE=0.0005
WEB_DEFAULT_SLIPPAGE_RATE=0.0005
```

## 🎨 Key Features

### 📈 Backtesting (Phase 2-3)

- **Dynamic Parameter Configuration**: Real-time strategy parameter adjustment
- **Multi-Asset Backtesting**: Test strategies across multiple cryptocurrencies
- **Real-Time Metrics Display**: CAGR, MDD, Sharpe, Sortino, Calmar ratios
- **Interactive Charts**: Plotly-based equity curves, drawdown visualization

### 🔧 Parameter Optimization (Phase 4 - Planned)

- Grid Search optimization
- Random Search optimization
- Parallel processing support
- Optimization result visualization

### 📊 Advanced Analysis (Phase 5 - Planned)

- Walk-Forward Analysis for robustness testing
- Permutation testing for statistical validation
- VaR/CVaR risk metrics
- Monte Carlo simulation

## 🏗️ Architecture

### Service Layer

```
┌─────────────────────────────────────┐
│          Streamlit Pages            │
│  (backtest.py, optimization.py)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│           Components                 │
│  (sidebar/, metrics/, charts/)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│            Services                  │
│  (backtest_runner, data_loader)      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         Core Backtester              │
│  (EventDrivenBacktestEngine)         │
└─────────────────────────────────────┘
```

### Key Design Patterns

- **Strategy Registry**: Auto-discovers and registers available strategies
- **Pydantic Models**: Type-safe parameter specifications
- **Service Layer**: Separates UI from business logic
- **Component-Based UI**: Reusable Streamlit components

## 📚 Related Documentation

- [Full Planning Document](../../docs/planning/streamlit-backtest-ui-plan.md)
- [Backtester API](../../docs/api/backtester.md)
- [Strategy Guide](../../docs/guides/strategy_guide.md)
- [Architecture Overview](../../docs/architecture.md)

## 🔗 Integration with Core System

The web UI integrates with the core backtesting system:

```python
from src.backtester.models import BacktestConfig
from src.backtester.engine.event_driven import EventDrivenBacktestEngine
from src.strategies.volatility_breakout import VanillaVBO

# Configuration from web UI
config = BacktestConfig(
    initial_capital=10_000_000,
    fee_rate=0.0005,
    slippage_rate=0.0005,
)

# Strategy from registry
strategy = VanillaVBO(sma_period=4, trend_sma_period=8)

# Run backtest
engine = EventDrivenBacktestEngine(config)
result = engine.run(strategy, data_files)
```

## ⚠️ Notes

- Requires data files in `data/raw/` directory (Parquet format)
- Use `crypto-quant collect` CLI to download market data first
- Web UI is for development/analysis; use CLI for production backtests