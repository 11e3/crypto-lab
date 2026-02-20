"""
Event-driven backtest data loader.

Handles loading and preparation of data for event-driven backtesting.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from src.backtester.engine.data_loader_base import (
    apply_strategy_signals,
    load_parquet_data,
    validate_required_columns,
)
from src.strategies.base import Strategy
from src.strategies.base_models import Position
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["Position", "load_event_data"]

_REQUIRED_COLUMNS = ["open", "high", "low", "close", "entry_signal", "exit_signal"]


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
            df = load_parquet_data(filepath)
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            continue
        except Exception as e:
            logger.error(f"Error loading {ticker} from {filepath}: {e}")
            continue

        try:
            # Filter by date range using DatetimeIndex directly
            if start_date is not None:
                df = df[pd.to_datetime(df.index).date >= start_date]
            if end_date is not None:
                df = df[pd.to_datetime(df.index).date <= end_date]

            if df.empty:
                logger.warning(f"No data for {ticker} after date filtering")
                continue

            df = apply_strategy_signals(df, strategy)

            # Store date part for type-safe filtering
            df = df.copy()
            df["index_date"] = pd.Series(pd.to_datetime(df.index).date, index=df.index)

            if not validate_required_columns(df, _REQUIRED_COLUMNS, ticker):
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
            logger.error(f"Error processing {ticker}: {e}")
            continue

    return ticker_data
