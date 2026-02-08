# Crypto Quant System

**cryptocurrency trading platform with backtesting, live trading, and portfolio optimization.**

Part of: **[crypto-quant-system](https://github.com/11e3/crypto-quant-system)** -> [bt](https://github.com/11e3/bt) -> [crypto-bot](https://github.com/11e3/crypto-bot) -> [crypto-regime-classifier-ml](https://github.com/11e3/crypto-regime-classifier-ml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Coverage](https://img.shields.io/badge/coverage-80%25+-green.svg)](https://github.com/11e3/crypto-quant-system)
[![MyPy](https://img.shields.io/badge/mypy-97.8%25-brightgreen.svg)](https://mypy.readthedocs.io/)

## Live Trading Performance (Upbit)

<table>
  <tr>
    <td align="center"><b>2024</b><br>Return: 101.26%</td>
    <td align="center"><b>2025</b><br>Return: 19.28%</td>
    <td align="center"><b>2026 YTD</b><br>Return: 7.99%</td>
  </tr>
  <tr>
    <td><img src="docs/images/KakaoTalk_20260208_142941894_01.jpg" width="333"/></td>
    <td><img src="docs/images/KakaoTalk_20260208_142941894_02.jpg" width="333"/></td>
    <td><img src="docs/images/KakaoTalk_20260208_142941894_03.jpg" width="333"/></td>
  </tr>
</table>

## Backtest Results

<table>
  <tr>
    <td align="center"><br>Equity Curve</td>
    <td align="center"><br>Yearly Returns</td>
  </tr>
  <tr>
    <td><img src="docs/images/screencapture-localhost-8501-2026-01-17-06_29_52.png" width="500"/></td>
    <td><img src="docs/images/screencapture-localhost-8501-2026-01-17-06_30_03.png" width="500"/></td>
  </tr>
</table>

## Ecosystem Role

```
+---------------------------------------------------------------------+
|                     Crypto Quant Ecosystem                          |
+----------------------------------+----------------------------------+
|  crypto-quant-system (this repo) |  Dashboard, Backtest, Data       |
|    +-- Data Pipeline             |  - Fetches OHLCV from Upbit      |
|    +-- Backtesting Engine        |  - Event-driven & Vectorized     |
|    +-- Strategy Library          |  - VBO, ORB, Momentum, MeanRev   |
|    +-- Web Dashboard             |  - Streamlit multi-page app      |
|    +-- Bot Monitor               |  - Reads logs from GCS           |
+----------------------------------+----------------------------------+
|  crypto-bot                      |  Live Trading Bot                |
|    +-- Logs to GCS               |  - Monitored via dashboard       |
+----------------------------------+----------------------------------+
|  crypto-regime-classifier-ml     |  Market Regime Classifier        |
|    +-- Models to GCS             |  - Viewable in dashboard         |
+----------------------------------+----------------------------------+
```

## Features

### 1. Data Pipeline

```python
# CLI data download
python scripts/fetch_data.py --symbols BTC,ETH,XRP --interval day

# Or use the collector programmatically
from src.data.collector import UpbitDataCollector

collector = UpbitDataCollector()
collector.collect("KRW-BTC", "day")
```

### 2. Backtesting Engine

Dual-engine architecture for flexibility:

```python
from src.backtester import EventDrivenBacktestEngine, BacktestConfig
from src.strategies.volatility_breakout import VanillaVBO

# Event-driven for accuracy
engine = EventDrivenBacktestEngine()
strategy = VanillaVBO(sma_period=5, trend_sma_period=10)
result = engine.run(strategy, data, BacktestConfig())

# Vectorized for speed (parameter optimization)
from src.backtester import VectorizedBacktestEngine
engine = VectorizedBacktestEngine()
```

### 3. Strategy Library

Built-in strategies with composable conditions:

| Strategy | Description |
|----------|-------------|
| **VBO** | Volatility Breakout with noise/trend filters |
| **ORB** | Opening Range Breakout |
| **Momentum** | Trend-following momentum |
| **Mean Reversion** | Statistical mean reversion |

### 4. Web Dashboard

Interactive Streamlit application with:
- Data collection UI
- Backtest execution with 30+ metrics
- Parameter optimization (Grid/Random search)
- Walk-forward analysis
- Bot monitoring via GCS

### 5. Bot Monitor (GCS Integration)

Real-time monitoring of `crypto-bot` via GCS:

```python
from src.data.storage import GCSStorage

storage = GCSStorage(bucket_name="your-quant-bucket")
trades = storage.get_bot_logs(date="2025-01-16", account="Main")
positions = storage.get_bot_positions(account="Main")
```

## Quick Start

### Installation

```bash
git clone <repository-url>
cd crypto-quant-system

# Install with uv (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

### Run Dashboard

```bash
uv run streamlit run src/web/app.py
```

### Download Data

```bash
# Download specific symbols
python scripts/fetch_data.py --symbols BTC,ETH,XRP --interval day

# Update all existing data
python scripts/fetch_data.py --update

# List available data
python scripts/fetch_data.py --list

# Download with multiple intervals
python scripts/fetch_data.py --symbols BTC --interval day,minute240,minute30
```

### Run Backtest

```bash
# Via script
python scripts/backtest/run_backtest.py --mode report

# Or programmatically
python -c "
from src.backtester import run_backtest
from src.strategies.volatility_breakout import VanillaVBO
result = run_backtest(VanillaVBO(), ['KRW-BTC', 'KRW-ETH'])
print(result.summary())
"
```

## Project Structure

```
crypto-quant-system/
+-- src/                          # Main source code
|   +-- backtester/               # Backtest engines
|   |   +-- engine/               # Event-driven & vectorized engines
|   |   +-- analysis/             # Bootstrap, Monte Carlo, WFA
|   |   +-- html/                 # HTML report generation
|   |   +-- wfa/                  # Walk-forward analysis
|   |   +-- models.py             # BacktestConfig, BacktestResult
|   |   +-- metrics.py            # Performance metrics
|   |   +-- optimization.py       # Parameter optimization
|   |
|   +-- strategies/               # Trading strategies
|   |   +-- base.py               # Strategy base class
|   |   +-- volatility_breakout/  # VBO strategy
|   |   +-- opening_range_breakout/ # ORB strategy
|   |   +-- momentum/             # Momentum strategy
|   |   +-- mean_reversion/       # Mean reversion strategy
|   |
|   +-- data/                     # Data collection
|   |   +-- collector.py          # Upbit data collector
|   |   +-- storage.py            # GCS integration
|   |
|   +-- exchange/                 # Exchange abstraction
|   |   +-- upbit.py              # Upbit implementation
|   |
|   +-- execution/                # Live trading
|   |   +-- bot/                  # Trading bot
|   |   +-- order_manager.py      # Order management
|   |
|   +-- risk/                     # Risk management
|   |   +-- portfolio_optimization.py # MVO, HRP, etc.
|   |   +-- metrics_var.py        # VaR/CVaR
|   |
|   +-- web/                      # Streamlit dashboard
|   |   +-- app.py                # Main app
|   |   +-- pages/                # Multi-page app
|   |   |   +-- backtest.py       # Backtest UI
|   |   |   +-- data_collect.py   # Data collection
|   |   |   +-- optimization.py   # Parameter optimization
|   |   |   +-- analysis.py       # Advanced analysis
|   |   |   +-- monitor.py        # Bot monitor (GCS)
|   |   +-- components/           # Reusable UI components
|   |   +-- services/             # Business logic
|   |
|   +-- config/                   # Configuration
|   +-- utils/                    # Utilities
|
+-- scripts/                      # CLI tools
|   +-- fetch_data.py             # Data download CLI
|   +-- backtest/                 # Backtest scripts
|   +-- data/                     # Data collection scripts
|   +-- tools/                    # Analysis tools
|
+-- data/                         # Data storage
|   +-- raw/                      # Raw OHLCV data
|   +-- processed/                # Processed data
|
+-- tests/                        # Test suite
|   +-- unit/                     # Unit tests
|   +-- integration/              # Integration tests
|
+-- docs/                         # Documentation
+-- config/                       # Configuration examples
```

## Dashboard Pages

### Home
System overview and getting started guide.

### Data Collection
| Feature | Description |
|---------|-------------|
| Ticker selector | Choose coins from Upbit |
| Interval selector | 1min to monthly candles |
| Download button | Fetch data with progress |
| Data status | Show available data |

### Backtest
| Feature | Description |
|---------|-------------|
| Strategy selector | VBO, ORB, Momentum, MeanRev |
| Parameter config | Dynamic UI based on strategy |
| Multi-asset | Test multiple tickers |
| Metrics | CAGR, MDD, Sharpe, 30+ metrics |
| Charts | Equity curve, drawdown, heatmap |

### Optimization
| Feature | Description |
|---------|-------------|
| Grid search | Exhaustive parameter search |
| Random search | Fast exploration |
| Parallel | Multi-core support |
| Results table | Sortable by metric |

### Analysis
| Feature | Description |
|---------|-------------|
| Walk-Forward | Out-of-sample validation |
| Monte Carlo | Risk simulation |
| Bootstrap | Confidence intervals |

### Bot Monitor
| Feature | Description |
|---------|-------------|
| Live positions | Current holdings |
| Trade history | From GCS logs |
| PnL summary | Daily/weekly/monthly |
| Alerts | Error notifications |

## GCS Integration

### Configuration

```bash
# .env
GCS_BUCKET=your-quant-bucket
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Bucket Structure

```
gs://your-quant-bucket/
+-- logs/
|   +-- {account}/
|       +-- trades_2025-01-16.csv
|       +-- positions.json
+-- models/
|   +-- regime_classifier_v1.pkl
+-- data/
    +-- processed/
        +-- BTC_1d.parquet
```

## Development

### Setup

```bash
# Install with dev dependencies
uv sync --all-extras

# Setup pre-commit hooks
pre-commit install
```

### Quality Gates

```bash
# Format
uv run ruff format src/ tests/

# Lint
uv run ruff check --fix src/ tests/

# Type check
uv run mypy src/

# Test with coverage
uv run pytest --cov=src --cov-fail-under=80
```

### Run Tests

```bash
# All tests
uv run pytest

# Specific module
uv run pytest tests/unit/test_backtester/

# With coverage report
uv run pytest --cov=src --cov-report=html
```

## Performance Metrics

The backtester calculates 30+ metrics:

| Category | Metrics |
|----------|---------|
| Returns | Total Return, CAGR, Monthly/Yearly |
| Risk | MDD, Volatility, VaR, CVaR |
| Risk-Adjusted | Sharpe, Sortino, Calmar |
| Trade Stats | Win Rate, Profit Factor, Avg Trade |
| Statistical | Skewness, Kurtosis, Hit Ratio |

## Challenges & Solutions

### 1. Backtesting Accuracy vs Speed Trade-off
**Challenge**: Event-driven backtesting is accurate but slow; vectorized is fast but lacks granularity.

**Solution**: Dual-engine architecture
- **EventDrivenBacktestEngine**: Sequential bar processing for accurate fill simulation, slippage, and position tracking
- **VectorizedBacktestEngine**: NumPy-based for parameter optimization (100x faster)
- User chooses based on use case

### 2. Look-Ahead Bias Prevention
**Challenge**: Accidentally using future data in signal generation invalidates backtest results.

**Solution**:
- Strict separation of signal generation and execution
- Signals generated on bar close, executed on next bar open
- Walk-forward analysis validates out-of-sample performance

### 3. Strategy Extensibility
**Challenge**: Adding new strategies required modifying multiple files and UI components.

**Solution**: Composable Conditions pattern + Strategy Registry
```python
# Strategies auto-register via __init_subclass__
class MyStrategy(Strategy):
    entry_conditions = [RSICondition(30), MACDCondition()]
    exit_conditions = [StopLossCondition(0.02)]
# Automatically appears in web UI
```

### 4. Type Safety in Data Pipeline
**Challenge**: Runtime errors from mismatched DataFrame columns and types.

**Solution**:
- Pydantic models for all configurations
- Protocol interfaces for engine contracts
- 97.8% MyPy strict mode coverage

### 5. Live Trading Reliability
**Challenge**: Network failures, API rate limits, and order execution errors.

**Solution**: Event Bus architecture with retry logic
- Decoupled order management from strategy logic
- Exponential backoff for API calls
- GCS-based logging for audit trail

---

## Architecture Decisions

### ADR-001: Event-Driven vs Vectorized Backtesting
**Context**: Need both accuracy for final validation and speed for optimization.

**Decision**: Implement both engines with shared interfaces.

**Consequences**:
- ✅ Users can optimize quickly, then validate accurately
- ✅ Same strategy code works with both engines
- ⚠️ Maintaining two engines increases complexity

### ADR-002: Protocol-Based Dependency Injection
**Context**: Need to swap exchange implementations and mock for testing.

**Decision**: Use Python Protocols instead of ABC for loose coupling.

```python
class ExchangeProtocol(Protocol):
    def place_order(self, order: Order) -> OrderResult: ...
    def get_balance(self) -> Balance: ...
```

**Consequences**:
- ✅ Easy mocking in tests
- ✅ Can add new exchanges without modifying existing code
- ✅ Runtime duck typing with static type checking

### ADR-003: Streamlit for Dashboard
**Context**: Need interactive UI for non-technical users.

**Decision**: Streamlit over Dash/Flask for rapid development.

**Consequences**:
- ✅ 10x faster UI development
- ✅ Built-in caching and session state
- ⚠️ Limited customization compared to React
- ⚠️ Not suitable for high-frequency updates

### ADR-004: Parquet for Data Storage
**Context**: OHLCV data needs fast reads, compression, and type preservation.

**Decision**: Apache Parquet over CSV/SQLite.

**Consequences**:
- ✅ 10x smaller file size than CSV
- ✅ Column-based reads (only load needed columns)
- ✅ Native datetime and decimal type support
- ⚠️ Not human-readable

### ADR-005: GCS for Bot Monitoring
**Context**: Live bot runs separately; need centralized logging.

**Decision**: Google Cloud Storage for logs and state.

**Consequences**:
- ✅ Decoupled bot from dashboard
- ✅ Historical log retention
- ✅ Multiple accounts/bots supported
- ⚠️ Requires GCP setup

---

## Performance Metrics (Detailed)

The backtester calculates 30+ metrics across 5 categories:

### Returns
| Metric | Formula | Description |
|--------|---------|-------------|
| Total Return | `(Final - Initial) / Initial` | Overall percentage gain |
| CAGR | `(Final/Initial)^(1/years) - 1` | Annualized return |
| Monthly Returns | Per-month breakdown | Seasonality analysis |
| Best/Worst Month | Min/Max monthly | Extreme performance |

### Risk
| Metric | Formula | Description |
|--------|---------|-------------|
| Max Drawdown | `max(peak - trough) / peak` | Worst peak-to-trough decline |
| Volatility | `std(returns) * sqrt(252)` | Annualized standard deviation |
| VaR (95%) | `percentile(returns, 5)` | Value at Risk |
| CVaR (95%) | `mean(returns < VaR)` | Expected Shortfall |
| Downside Deviation | `std(negative returns)` | Downside volatility |

### Risk-Adjusted
| Metric | Formula | Description |
|--------|---------|-------------|
| Sharpe Ratio | `(return - rf) / volatility` | Return per unit risk |
| Sortino Ratio | `(return - rf) / downside_dev` | Penalizes only downside |
| Calmar Ratio | `CAGR / Max Drawdown` | Return vs worst loss |
| Information Ratio | `alpha / tracking_error` | Active return efficiency |

### Trade Statistics
| Metric | Formula | Description |
|--------|---------|-------------|
| Win Rate | `winning_trades / total_trades` | Percentage of winners |
| Profit Factor | `gross_profit / gross_loss` | Profit efficiency |
| Avg Trade | `total_pnl / num_trades` | Expected value per trade |
| Avg Winner/Loser | `avg(wins)` / `avg(losses)` | Asymmetry analysis |
| Max Consecutive Wins/Losses | Streak counting | Psychology impact |
| Holding Period | `avg(exit_time - entry_time)` | Average trade duration |

### Statistical
| Metric | Formula | Description |
|--------|---------|-------------|
| Skewness | Third moment | Return distribution asymmetry |
| Kurtosis | Fourth moment | Tail risk (fat tails) |
| Hit Ratio | Profitable days / total days | Daily success rate |
| Expectancy | `win_rate * avg_win - loss_rate * avg_loss` | Expected return per trade |

---

## Roadmap

- [ ] Real-time price streaming
- [ ] Strategy builder UI
- [ ] Regime model integration
- [ ] Multi-account aggregation
- [ ] Alert configuration

## License

MIT License

---

**Version**: 2.1.0 | **Python**: 3.10+ | **Framework**: Streamlit

## Changelog

### v2.1.0 (2026-01-22)
- **UI Improvements**
  - Equity curve now displays normalized values (1 = start) with log scale for better visualization
  - Simplified backtest results tabs (3 tabs: Equity Curve, Yearly Returns, Trade History)
  - Fixed checkbox state management in data collection page
- **Performance**
  - Improved page load speed by optimizing cache strategy (removed TTL from strategy registry)
  - Added `lru_cache` for bt availability check
- **Bug Fixes**
  - Fixed GCS environment variables not loading in Streamlit (added `load_dotenv()`)
  - Fixed parameter slider bounds when default values exceed max_value
