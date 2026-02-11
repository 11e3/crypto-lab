"""BTC data loader utility for market filter strategies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_btc_data(data_dir: Path, interval: str = "day") -> pd.DataFrame | None:
    """Load BTC parquet data for market filter."""
    file_path = data_dir / f"KRW-BTC_{interval}.parquet"
    if not file_path.exists():
        return None
    df = pd.read_parquet(file_path)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str.lower()
    return df
