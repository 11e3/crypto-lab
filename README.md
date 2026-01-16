# Crypto Quant System

**Dashboard and data pipeline hub for the Crypto Quant Ecosystem.**

Part of: **[crypto-quant-system](https://github.com/11e3/crypto-quant-system)** → [bt](https://github.com/11e3/bt) → [crypto-bot](https://github.com/11e3/crypto-bot) → [crypto-regime-classifier-ml](https://github.com/11e3/crypto-regime-classifier-ml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

## Ecosystem Role

```
┌─────────────────────────────────────────────────────────────────┐
│                    Crypto Quant Ecosystem                       │
├─────────────────────────────────────────────────────────────────┤
│  crypto-quant-system     │  Dashboard & data pipeline          │
│  (this repo)             │                                     │
│    ├── Data download     │  - Fetches OHLCV from exchanges     │
│    ├── Data processing   │  - Cleans and formats data          │
│    ├── Backtest UI       │  - Imports bt for backtesting       │
│    └── Bot log viewer    │  - Reads logs from GCS              │
├──────────────────────────┼──────────────────────────────────────┤
│  bt                      │  Backtesting engine                 │
│    └── Imported here     │  - Strategy validation              │
├──────────────────────────┼──────────────────────────────────────┤
│  crypto-bot              │  Live trading bot                   │
│    └── Logs to GCS       │  - Monitored via dashboard          │
├──────────────────────────┼──────────────────────────────────────┤
│  crypto-regime-ml        │  Market regime classifier           │
│    └── Models to GCS     │  - Viewable in dashboard            │
└──────────────────────────┴──────────────────────────────────────┘
```

## Features

### 1. Data Pipeline

```python
# Download OHLCV data from exchanges
from data.fetcher import OHLCVFetcher

fetcher = OHLCVFetcher(exchange="upbit")
fetcher.download(symbols=["BTC", "ETH"], interval="1d")
fetcher.save_to_csv("data/")
```

### 2. Backtest Dashboard

Interactive backtesting powered by `bt` framework:

```python
# Dashboard imports bt for backtesting
from bt.framework.facade import BacktestFacade

def run_backtest(strategy: str, symbols: list, start: str, end: str):
    framework = BacktestFacade()
    data = load_processed_data(symbols, start, end)
    return framework.run_backtest(
        strategy=strategy,
        symbols=symbols,
        data=data
    )
```

### 3. Bot Log Viewer

Real-time monitoring of `crypto-bot` via GCS:

```python
# Read bot logs from GCS
from google.cloud import storage

def get_bot_logs(date: str, account: str = "Main"):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"logs/{account}/trades_{date}.csv")
    return pd.read_csv(blob.open("r"))
```

## Quick Start

### Installation

```bash
git clone <repository-url>
cd crypto-quant-system

# Install dependencies
pip install -r requirements.txt

# Install bt framework
pip install -e ../bt
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

### Download Data

```bash
# Download all supported coins
python scripts/fetch_data.py --symbols BTC,ETH,XRP --interval 1d

# Update existing data
python scripts/fetch_data.py --update
```

## Project Structure

```
crypto-quant-system/
├── dashboard/
│   ├── app.py              # Main Streamlit app
│   ├── pages/
│   │   ├── backtest.py     # Backtest UI (uses bt)
│   │   ├── data.py         # Data management
│   │   └── monitor.py      # Bot log viewer (reads GCS)
│   └── components/
│       ├── charts.py       # Plotly charts
│       └── metrics.py      # Performance cards
├── data/
│   ├── fetcher.py          # Exchange data downloader
│   ├── processor.py        # Data cleaning & formatting
│   └── storage.py          # GCS integration
├── scripts/
│   ├── fetch_data.py       # CLI data download
│   └── update_data.py      # Scheduled updates
└── config/
    └── settings.py         # Configuration
```

## Dashboard Pages

### Backtest Page

| Feature | Description |
|---------|-------------|
| Strategy selector | Choose from available strategies |
| Date range picker | Backtest period selection |
| Symbol selector | Multi-coin portfolio support |
| Performance metrics | CAGR, MDD, Sharpe, Win rate |
| Equity curve | Interactive chart |
| Trade list | Detailed trade history |

### Data Management Page

| Feature | Description |
|---------|-------------|
| Download status | Per-symbol data availability |
| Date range | Available data period |
| Update button | Fetch latest data |
| Preview | OHLCV data table |

### Bot Monitor Page

| Feature | Description |
|---------|-------------|
| Live positions | Current holdings per account |
| Trade history | Recent trades from GCS logs |
| PnL summary | Daily/weekly/monthly returns |
| Alerts | Error notifications |

## GCS Integration

### Configuration

```env
# .env
GCS_BUCKET=your-quant-bucket
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Bucket Structure

```
gs://your-quant-bucket/
├── logs/
│   └── {account}/
│       ├── trades_2025-01-16.csv
│       └── positions.json
├── models/
│   └── regime_classifier_v1.pkl
└── data/
    └── processed/
        ├── BTC_1d.parquet
        └── ETH_1d.parquet
```

## Integration with bt

```python
# Import bt for backtesting
from bt.framework.facade import BacktestFacade
from bt.strategies import VolatilityBreakout, Momentum

# Available strategies from bt
STRATEGIES = {
    "volatility_breakout": VolatilityBreakout,
    "momentum": Momentum,
    "mean_reversion": MeanReversion,
}

# Run backtest via dashboard
result = BacktestFacade().run_backtest(
    strategy=selected_strategy,
    symbols=selected_symbols,
    data=processed_data
)

# Display results
st.metric("CAGR", f"{result['performance']['cagr']:.1f}%")
st.metric("Sharpe", f"{result['performance']['sharpe']:.2f}")
st.plotly_chart(create_equity_curve(result['equity']))
```

## Development

```bash
# Run tests
pytest tests/

# Lint
ruff check .

# Format
ruff format .
```

## Roadmap

- [ ] Real-time price streaming
- [ ] Strategy parameter optimization UI
- [ ] Regime model performance tracking
- [ ] Multi-account PnL aggregation
- [ ] Alert configuration UI

## License

MIT License

---

**Version**: 1.0.0 | **Ecosystem**: Crypto Quant System | **Dashboard**: Streamlit
