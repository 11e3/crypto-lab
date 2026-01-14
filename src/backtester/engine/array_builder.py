"""
NumPy array builder for vectorized backtesting.

Converts pandas DataFrames to efficient numpy arrays for simulation.
"""

from datetime import date

import numpy as np
import pandas as pd

from src.utils.memory import get_float_dtype


def build_numpy_arrays(
    ticker_data: dict[str, pd.DataFrame],
    sorted_dates: np.ndarray,
) -> tuple[list[str], int, int, dict[str, np.ndarray]]:
    """
    Build numpy arrays from ticker DataFrames.

    Args:
        ticker_data: Dictionary of ticker -> DataFrame
        sorted_dates: Sorted array of dates

    Returns:
        Tuple of (tickers, n_tickers, n_dates, arrays_dict)
    """
    tickers = list(ticker_data.keys())
    n_tickers = len(tickers)
    n_dates = len(sorted_dates)

    float_dtype = get_float_dtype()
    arrays: dict[str, np.ndarray] = {
        "opens": np.full((n_tickers, n_dates), np.nan, dtype=float_dtype),
        "highs": np.full((n_tickers, n_dates), np.nan, dtype=float_dtype),
        "lows": np.full((n_tickers, n_dates), np.nan, dtype=float_dtype),
        "closes": np.full((n_tickers, n_dates), np.nan, dtype=float_dtype),
        "targets": np.full((n_tickers, n_dates), np.nan, dtype=float_dtype),
        "smas": np.full((n_tickers, n_dates), np.nan, dtype=float_dtype),
        "entry_signals": np.zeros((n_tickers, n_dates), dtype=bool),
        "exit_signals": np.zeros((n_tickers, n_dates), dtype=bool),
        "whipsaws": np.zeros((n_tickers, n_dates), dtype=bool),
        "entry_prices": np.full((n_tickers, n_dates), np.nan, dtype=float_dtype),
        "exit_prices": np.full((n_tickers, n_dates), np.nan, dtype=float_dtype),
        "short_noises": np.full((n_tickers, n_dates), np.nan, dtype=float_dtype),
    }

    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

    for t_idx, ticker in enumerate(tickers):
        _fill_ticker_arrays(ticker_data[ticker], t_idx, date_to_idx, arrays, float_dtype)

    return tickers, n_tickers, n_dates, arrays


def _fill_ticker_arrays(
    df: pd.DataFrame,
    t_idx: int,
    date_to_idx: dict[date, int],
    arrays: dict[str, np.ndarray],
    float_dtype: type,
) -> None:
    """Fill arrays for a single ticker."""
    df_index = pd.DatetimeIndex(df.index)
    df_dates = pd.Series(df_index.to_pydatetime()).dt.date
    df_idx = df_dates.map(date_to_idx)
    valid_mask = df_idx.notna()
    idx = df_idx[valid_mask].astype(int).to_numpy()
    mask = valid_mask.to_numpy()

    arrays["opens"][t_idx, idx] = df.loc[mask, "open"].to_numpy(dtype=float_dtype)
    arrays["highs"][t_idx, idx] = df.loc[mask, "high"].to_numpy(dtype=float_dtype)
    arrays["lows"][t_idx, idx] = df.loc[mask, "low"].to_numpy(dtype=float_dtype)
    arrays["closes"][t_idx, idx] = df.loc[mask, "close"].to_numpy(dtype=float_dtype)

    if "target" in df.columns:
        arrays["targets"][t_idx, idx] = df.loc[mask, "target"].to_numpy(dtype=float_dtype)
    else:
        arrays["targets"][t_idx, idx] = df.loc[mask, "close"].to_numpy(dtype=float_dtype)

    if "sma" in df.columns:
        arrays["smas"][t_idx, idx] = df.loc[mask, "sma"].to_numpy(dtype=float_dtype)

    arrays["entry_signals"][t_idx, idx] = df.loc[mask, "entry_signal"].astype(bool).to_numpy()
    arrays["exit_signals"][t_idx, idx] = df.loc[mask, "exit_signal"].astype(bool).to_numpy()
    arrays["whipsaws"][t_idx, idx] = df.loc[mask, "is_whipsaw"].to_numpy(dtype=float_dtype)
    arrays["entry_prices"][t_idx, idx] = df.loc[mask, "entry_price"].to_numpy(dtype=float_dtype)
    arrays["exit_prices"][t_idx, idx] = df.loc[mask, "exit_price"].to_numpy(dtype=float_dtype)

    if "short_noise" in df.columns:
        arrays["short_noises"][t_idx, idx] = df.loc[mask, "short_noise"].to_numpy(dtype=float_dtype)


def filter_valid_dates(
    sorted_dates: np.ndarray,
    arrays: dict[str, np.ndarray],
    n_dates: int,
) -> tuple[np.ndarray, int, dict[str, np.ndarray]]:
    """
    Filter to valid dates where indicators are available.

    Args:
        sorted_dates: Array of dates
        arrays: Dictionary of numpy arrays
        n_dates: Number of dates

    Returns:
        Tuple of (filtered_sorted_dates, filtered_n_dates, filtered_arrays)
    """
    n_tickers = arrays["closes"].shape[0]
    valid_date_mask = np.zeros(n_dates, dtype=bool)

    for d_idx in range(n_dates):
        has_valid_target = ~np.isnan(arrays["targets"][:, d_idx])
        has_valid_data = ~np.isnan(arrays["closes"][:, d_idx])

        if np.any(~np.isnan(arrays["smas"][:, d_idx])):
            has_valid_sma = ~np.isnan(arrays["smas"][:, d_idx])
        else:
            has_valid_sma = np.ones(n_tickers, dtype=bool)

        if np.any(has_valid_target & has_valid_sma & has_valid_data):
            valid_date_mask[d_idx] = True

    valid_indices = np.where(valid_date_mask)[0]
    if len(valid_indices) > 0:
        first_valid_idx = valid_indices[0]
        last_valid_idx = valid_indices[-1] + 1

        sorted_dates = sorted_dates[first_valid_idx:last_valid_idx]
        n_dates = len(sorted_dates)

        for key in arrays:
            arrays[key] = arrays[key][:, first_valid_idx:last_valid_idx]

    return sorted_dates, n_dates, arrays


def collect_valid_dates(ticker_data: dict[str, pd.DataFrame]) -> set[date]:
    """
    Collect all valid dates from ticker data.

    Args:
        ticker_data: Dictionary of ticker -> DataFrame

    Returns:
        Set of valid dates
    """
    all_dates: set[date] = set()

    for df in ticker_data.values():
        valid_mask = df["close"].notna()
        if "sma" in df.columns:
            valid_mask = valid_mask & df["sma"].notna()
        if "target" in df.columns:
            valid_mask = valid_mask & df["target"].notna()

        dt_index = pd.DatetimeIndex(df.index[valid_mask])
        all_dates.update({d.date() for d in dt_index.to_pydatetime()})

    return all_dates
