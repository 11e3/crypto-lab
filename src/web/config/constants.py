"""Shared constants for the web dashboard.

Centralizes values that were previously duplicated across pages.
"""

# Default ticker list used by optimization and analysis pages
DEFAULT_TICKERS: list[str] = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]

# Extended ticker list for data collection (superset of DEFAULT_TICKERS)
DATA_COLLECT_TICKERS: list[str] = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-SOL",
    "KRW-DOGE",
    "KRW-TRX",
    "KRW-ADA",
    "KRW-AVAX",
    "KRW-SHIB",
    "KRW-LINK",
]

# Supported intervals for data collection
DATA_COLLECT_INTERVALS: list[tuple[str, str]] = [
    ("minute1", "1 min"),
    ("minute3", "3 min"),
    ("minute5", "5 min"),
    ("minute10", "10 min"),
    ("minute15", "15 min"),
    ("minute30", "30 min"),
    ("minute60", "1 hour"),
    ("minute240", "4 hours"),
    ("day", "Daily"),
    ("week", "Weekly"),
    ("month", "Monthly"),
]

# Binance ticker list for data collection (USDT + BTC pairs)
BINANCE_DATA_COLLECT_TICKERS: list[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "ETHBTC",
    "BNBBTC",
    "SOLBTC",
    "XRPBTC",
    "ADABTC",
]

# Binance intervals for data collection
BINANCE_DATA_COLLECT_INTERVALS: list[tuple[str, str]] = [
    ("1m", "1 min"),
    ("3m", "3 min"),
    ("5m", "5 min"),
    ("15m", "15 min"),
    ("30m", "30 min"),
    ("1h", "1 hour"),
    ("2h", "2 hours"),
    ("4h", "4 hours"),
    ("1d", "Daily"),
    ("1w", "Weekly"),
    ("1M", "Monthly"),
]

# Interval display mapping for backtest/analysis pages
INTERVAL_DISPLAY_MAP: dict[str, str] = {
    "minute240": "4 Hours",
    "day": "Daily",
    "week": "Weekly",
}

# Optimization metric options used by optimization and analysis pages
OPTIMIZATION_METRICS: list[tuple[str, str]] = [
    ("sharpe_ratio", "Sharpe Ratio"),
    ("cagr", "CAGR"),
    ("total_return", "Total Return"),
    ("calmar_ratio", "Calmar Ratio"),
    ("win_rate", "Win Rate"),
    ("profit_factor", "Profit Factor"),
]
