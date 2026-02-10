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
