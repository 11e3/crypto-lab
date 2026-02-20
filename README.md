# crypto-lab

Quantitative backtesting and strategy research platform for Upbit/Binance.

- **Vectorized backtester** — event-driven simulation with realistic cost modeling (fee + slippage)
- **Parameter optimizer** — grid and random search with parallel execution
- **Walk-forward analysis** — out-of-sample validation to detect overfitting
- **Data collection** — incremental OHLCV fetching from Upbit and Binance
- **Risk analytics** — VaR, CVaR, portfolio optimization (MPT, risk parity)
- **CLI** — single `crypto-lab` command covering all workflows

---

## Installation

Requires Python 3.12+.

```bash
# Install with all dependencies
pip install -e ".[analysis,dev]"

# Or with uv
uv sync --all-extras
```

---

## Quick Start

### List registered strategies

```bash
crypto-lab list
# VBO
# VBO_DAY
```

### Run a backtest

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

### Optimize parameters

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

### Walk-forward analysis

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

### Collect data

```bash
crypto-lab collect \
  --tickers KRW-BTC KRW-ETH KRW-XRP \
  --interval day \
  --source upbit
```

---

## CLI Reference

```
crypto-lab [--log-level LEVEL] COMMAND

Commands:
  backtest   Run strategy backtest on historical data
  optimize   Grid/random parameter search
  collect    Fetch OHLCV data from exchange
  wfa        Walk-forward analysis
  list       List registered strategies
```

Common flags shared by `backtest`, `optimize`, `wfa`:

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers` | (required) | Space-separated Upbit tickers, e.g. `KRW-BTC KRW-ETH` |
| `--strategy` | (required) | Registered strategy name |
| `--start` | all data | Start date `YYYY-MM-DD` |
| `--end` | all data | End date `YYYY-MM-DD` |
| `--capital` | 1,000,000 | Initial capital (KRW) |
| `--slots` | 5 | Max concurrent positions |
| `--fee` | 0.0005 | Fee rate per trade leg (0.05%) |
| `--interval` | `day` | Candle interval (`day`, `minute240`, …) |

---

## Project Structure

```
src/
├── main.py                  # CLI entry point
├── cli/                     # CLI subcommand modules
│   ├── cmd_backtest.py      #   backtest, optimize
│   ├── cmd_data.py          #   collect
│   └── cmd_wfa.py           #   wfa
├── strategies/
│   ├── base.py              # Strategy ABC
│   ├── registry.py          # StrategyFactory singleton
│   └── volatility_breakout/
│       ├── vbo_v1.py        # VBOV1 — breakout + BTC MA filter + MA exit
│       └── vbo_day_exit.py  # VBODayExit — fixed 1-day holding period
├── backtester/
│   ├── engine/              # Vectorized + event-driven engines, run_backtest()
│   ├── optimization.py      # Grid/random parameter optimizer
│   ├── wfa/                 # Walk-forward analysis
│   ├── analysis/            # Permutation test, robustness, bootstrap
│   └── models.py            # BacktestConfig, BacktestResult, Trade
├── data/
│   ├── collector.py         # UpbitDataCollector (incremental parquet update)
│   ├── collector_factory.py # DataCollectorFactory
│   ├── upbit_source.py      # Upbit OHLCV data source
│   ├── binance_source.py    # Binance OHLCV data source
│   └── cache/               # LRU data cache
├── risk/
│   ├── position_sizing.py   # Equal, volatility-adjusted, Kelly sizing
│   ├── portfolio_methods.py # MPT + risk parity optimization
│   └── metrics*.py          # VaR, CVaR, Sharpe, Sortino, …
├── config/                  # Pydantic settings + YAML loader
└── utils/
    ├── indicators.py         # SMA, EMA, ATR, RSI, Bollinger Bands
    └── indicators_vbo.py     # VBO-specific: target price, noise range
```

---

## Writing a Strategy

Subclass `Strategy` and register with `@registry.register`:

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

`parameter_schema()` drives `optimize` and `wfa` — no extra wiring needed.

---

## Volatility Breakout Strategy (VBO)

Entry rule (both conditions must hold):

1. **Breakout**: `high ≥ open + prev_range × noise_ratio`
2. **BTC filter**: `prev BTC close > prev BTC MA(btc_ma)`

Exit rule (VBOV1): previous `close < SMA(ma_short)` → exit at next open

Exit rule (VBODayExit): always exit at the next day's open

### Best parameters (BTC+ETH, 2020–2024)

| Goal | noise_ratio | btc_ma | ma_short | Sharpe | CAGR | MDD |
|------|-------------|--------|----------|--------|------|-----|
| Balanced | 0.6 | 30 | 3 | 2.54 | +121% | −17.9% |
| Max return | 0.3 | 10 | 3 | 2.20 | +128% | −23.9% |
| Min drawdown | 0.8 | 30 | 3 | 2.05 | +101% | −16.7% |

Full sweep research results (Korean): [`src/research/results/vb_upbit/README.md`](src/research/results/vb_upbit/README.md)

---

## Development

```bash
# Run tests
pytest tests/ -x -q

# Lint + format
ruff check src/ && ruff format src/

# Type check
mypy src/ --strict

# All three (quality gate)
pytest tests/ -x -q && ruff check src/ && mypy src/ --strict
```

Coverage threshold: **80%** (currently ~84%).

### Code conventions

- Files ≤ 200 lines, functions ≤ 50 lines
- Type annotations required on all public functions (mypy strict)
- English comments only
- Lazy imports inside function bodies for heavy deps (pyupbit, ccxt, matplotlib)

---

## Data Layout

```
data/
├── upbit/          # Parquet files: KRW-BTC_day.parquet, KRW-ETH_minute240.parquet, …
└── binance/        # Parquet files: BTC_USDT_1d.parquet, …
```

Files are created/updated automatically by `crypto-lab collect`.

---

## License

MIT
