"""
Configuration constants and settings for the trading system.

Centralizes all configuration values to avoid magic numbers and hardcoded values.
"""

from pathlib import Path
from typing import Final

# Project root directory
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
UPBIT_DATA_DIR: Final[Path] = DATA_DIR / "upbit"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"

# Upbit API Configuration
UPBIT_MAX_CANDLES_PER_REQUEST: Final[int] = 200
UPBIT_API_RATE_LIMIT_DELAY: Final[float] = 0.1  # seconds

# Binance API Configuration
BINANCE_MAX_CANDLES_PER_REQUEST: Final[int] = 1000
BINANCE_API_RATE_LIMIT_DELAY: Final[float] = 0.05  # seconds
BINANCE_DATA_DIR: Final[Path] = DATA_DIR / "binance"

# Backtest Defaults
DEFAULT_INITIAL_CAPITAL: Final[float] = 1.0
DEFAULT_FEE_RATE: Final[float] = 0.0005  # 0.05%
DEFAULT_SLIPPAGE_RATE: Final[float] = 0.0005  # 0.05%
DEFAULT_MAX_SLOTS: Final[int] = 4

# Trading Constants
ANNUALIZATION_FACTOR: Final[int] = 365  # Trading days per year for crypto
RISK_FREE_RATE: Final[float] = 0.0  # Risk-free rate for Sharpe/Sortino

# Cache Configuration
CACHE_METADATA_FILENAME: Final[str] = "_cache_metadata.json"

# Data File Naming
PARQUET_EXTENSION: Final[str] = ".parquet"


def parquet_filename(ticker: str, interval: str) -> str:
    """Generate standardized parquet filename for a ticker/interval pair."""
    return f"{ticker}_{interval}{PARQUET_EXTENSION}"


# Logging Configuration
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
