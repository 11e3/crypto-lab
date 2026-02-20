"""Data loading helpers for research experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtester.engine.data_loader import load_parquet_data


def load_parquet_ohlcv_by_symbol(data_dir: Path, interval: str) -> dict[str, pd.DataFrame]:
    """Load parquet OHLCV files into a symbol-keyed dataframe dictionary.

    Expected file naming convention: ``{symbol}_{interval}.parquet``.
    """
    files = sorted(data_dir.glob(f"*_{interval}.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir} for interval={interval}")

    data: dict[str, pd.DataFrame] = {}
    suffix = f"_{interval}"
    for file_path in files:
        symbol = file_path.stem.replace(suffix, "")
        data[symbol] = load_parquet_data(file_path)

    return data
