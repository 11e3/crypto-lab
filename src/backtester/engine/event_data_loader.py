"""
Event-driven backtest data loader.

Handles loading and preparation of data for event-driven backtesting.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Active position in a trade."""

    ticker: str
    entry_date: date
    entry_price: float
    amount: float
    highest_price: float  # For trailing stop tracking


def load_event_data(
    strategy: Strategy,
    data_files: dict[str, Path],
    start_date: date | None,
    end_date: date | None,
) -> dict[str, pd.DataFrame]:
    """Load and prepare data for all tickers."""
    ticker_data: dict[str, pd.DataFrame] = {}

    for ticker, filepath in data_files.items():
        try:
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue

            df = pd.read_parquet(filepath)
            df.index = pd.to_datetime(df.index)

            # Filter by date range using DatetimeIndex directly
            if start_date is not None:
                df = df[pd.to_datetime(df.index).date >= start_date]
            if end_date is not None:
                df = df[pd.to_datetime(df.index).date <= end_date]

            if df.empty:
                logger.warning(f"No data for {ticker} after date filtering")
                continue

            df = strategy.calculate_indicators(df)
            df = strategy.generate_signals(df)

            # Store date part for type-safe filtering
            df = df.copy()
            df["index_date"] = pd.Series(pd.to_datetime(df.index).date, index=df.index)

            required = ["open", "high", "low", "close", "entry_signal", "exit_signal"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                logger.error(f"{ticker}: Missing columns {missing}")
                continue

            if "entry_price" not in df.columns:
                df["entry_price"] = df.get("target", df["close"])
            if "exit_price" not in df.columns:
                df["exit_price"] = df.get("exit_price_base", df["close"])

            logger.info(
                f"Loaded {ticker}: {len(df)} rows, {df['entry_signal'].sum()} entry signals"
            )
            ticker_data[ticker] = df

        except Exception as e:
            logger.error(f"Error loading {ticker} from {filepath}: {e}")
            continue

    return ticker_data
