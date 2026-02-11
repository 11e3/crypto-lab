"""
Signal processing utilities for vectorized backtesting.

Handles entry/exit price calculations and whipsaw detection.

Slippage Architecture (dual-path):
    Vectorized engine: slippage is pre-computed here into entry_price/exit_price
    columns. The engine reads these adjusted prices directly from arrays.

    Event-driven engine: slippage is applied independently in event_exec.py
    (execute_entry/execute_exit). It does NOT use these pre-computed columns.

    Both paths apply slippage exactly once — there is no double-application.
"""

import pandas as pd

from src.backtester.models import BacktestConfig


def add_price_columns(
    df: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Add entry/exit price columns with slippage.

    Entry = target (VBO) or close + slippage, Exit = close - slippage.
    PnL% = ((exit - entry) / entry - fee_rate) * 100

    Args:
        df: DataFrame with signals
        config: Backtest configuration

    Returns:
        DataFrame with entry_price, exit_price, is_whipsaw
    """
    if (
        "target" not in df.columns
        or "entry_price" not in df.columns
        or "exit_price" not in df.columns
    ):
        df = df.copy()

    # Ensure signal columns are boolean
    if "entry_signal" in df.columns and df["entry_signal"].dtype != bool:
        df["entry_signal"] = df["entry_signal"].astype(bool)
    if "exit_signal" in df.columns and df["exit_signal"].dtype != bool:
        df["exit_signal"] = df["exit_signal"].astype(bool)

    # Whipsaw: entry & exit signal on same bar
    df["is_whipsaw"] = df["entry_signal"] & df["exit_signal"]

    # Entry price: target (VBO) or close, with slippage
    if "target" in df.columns:
        df["entry_price"] = df["target"] * (1 + config.slippage_rate)
    else:
        df["entry_price"] = df["close"] * (1 + config.slippage_rate)

    # Exit price: exit_price_base (or close) with slippage
    exit_base = df.get("exit_price_base", df["close"])
    df["exit_price"] = exit_base * (1 - config.slippage_rate)

    return df


def get_valid_dates_mask(df: pd.DataFrame) -> pd.Series:
    """
    Get mask for dates with valid indicator values.

    Args:
        df: DataFrame with indicators

    Returns:
        Boolean series indicating valid dates
    """
    valid_mask = df["close"].notna()

    if "sma" in df.columns:
        valid_mask = valid_mask & df["sma"].notna()
    if "target" in df.columns:
        valid_mask = valid_mask & df["target"].notna()

    return valid_mask
