"""
Convenience functions for running backtests.

Provides high-level API for running backtests with automatic data collection.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.backtester.models import BacktestConfig, BacktestResult
from src.config import RAW_DATA_DIR
from src.data.collector import Interval
from src.data.collector_factory import DataCollectorFactory
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_backtest(
    strategy: Strategy,
    tickers: list[str],
    interval: str = "day",
    data_dir: Path | None = None,
    config: BacktestConfig | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> BacktestResult:
    """
    Convenience function to run backtest.

    Args:
        strategy: Trading strategy
        tickers: List of tickers
        interval: Data interval
        data_dir: Data directory
        config: Backtest configuration
        start_date: Start date
        end_date: End date

    Returns:
        BacktestResult
    """
    data_dir = data_dir or RAW_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    data_files = {ticker: data_dir / f"{ticker}_{interval}.parquet" for ticker in tickers}

    # Collect missing data
    missing_tickers = _find_missing_tickers(data_files)
    if missing_tickers:
        _collect_missing_data(missing_tickers, data_dir, interval)

    data_files = {k: v for k, v in data_files.items() if v.exists()}

    if not data_files:
        raise FileNotFoundError(f"No data files found for tickers: {tickers}")

    engine = VectorizedBacktestEngine(config)
    result = engine.run(strategy, data_files, start_date=start_date, end_date=end_date)
    result.interval = interval
    return result


def _find_missing_tickers(data_files: dict[str, Path]) -> list[str]:
    """Find tickers with missing or outdated data files."""
    missing = []
    for ticker, filepath in data_files.items():
        if not filepath.exists():
            missing.append(ticker)
        else:
            try:
                file_age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
                if file_age > timedelta(days=1):
                    missing.append(ticker)
                else:
                    df = pd.read_parquet(filepath)
                    if df.empty or len(df) < 10:
                        missing.append(ticker)
            except (OSError, pd.errors.EmptyDataError, ValueError) as e:
                logger.debug(f"Failed to check data file for {ticker}: {e}")
                missing.append(ticker)
    return missing


def _collect_missing_data(tickers: list[str], data_dir: Path, interval: str) -> None:
    """Collect data for missing tickers."""
    logger.info(f"Collecting data for {len(tickers)} ticker(s): {tickers}")
    collector = DataCollectorFactory.create(data_dir=data_dir)

    interval_map: dict[str, Interval] = {
        "minute240": "minute240",
        "day": "day",
        "week": "week",
    }
    interval_type: Interval = interval_map.get(interval, "day")

    for ticker in tickers:
        try:
            collector.collect(ticker, interval_type, full_refresh=False)
            logger.info(f"Collected data for {ticker}")
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to collect data for {ticker}: {e}")
