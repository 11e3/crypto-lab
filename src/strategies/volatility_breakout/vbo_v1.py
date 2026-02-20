"""VBOV1 Strategy — faithful port of backtest_v1.py.

VBO + BTC MA20 entry, prev_close < prev_SMA exit at next open.
Fixed K=0.5, SMA includes current bar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.strategies.base import Strategy
from src.strategies.volatility_breakout.btc_data_loader import _load_btc_data


class VBOV1(Strategy):
    """V1 strategy: VBO + BTC MA20 entry, MA5 exit at open.

    Entry conditions (AND):
    - Breakout: high >= target (open + prev_range * noise_ratio)
    - BTC filter: prev BTC close > prev BTC MA(btc_ma)

    Exit condition:
    - prev_close < prev_SMA(ma_short) -> sell at today's open

    Key characteristics:
    - Fixed K value (noise_ratio=0.5)
    - SMA includes current bar (no shift)
    - Exit at open price (exit_price_base), not close
    - Exit signal uses previous day's condition (shift(1))

    Args:
        name: Strategy name
        ma_short: SMA period (default=5)
        btc_ma: BTC MA period for market filter (default=20)
        noise_ratio: Fixed K value for VBO breakout (default=0.5)
        btc_data: Pre-loaded BTC DataFrame (optional)
        data_dir: Directory containing parquet files
        interval: Data interval for BTC file lookup
    """

    def __init__(
        self,
        name: str = "VBOV1",
        ma_short: int = 5,
        btc_ma: int = 20,
        noise_ratio: float = 0.5,
        btc_data: pd.DataFrame | None = None,
        data_dir: Path | None = None,
        interval: str = "day",
        **_kwargs: Any,
    ) -> None:
        self.ma_short = ma_short
        self.btc_ma = btc_ma
        self.noise_ratio = noise_ratio
        self._btc_data = btc_data
        self._data_dir = data_dir or Path(__file__).resolve().parents[3] / "data" / "raw"
        self._interval = interval

        super().__init__(name=name)

    def _get_btc_data(self) -> pd.DataFrame | None:
        """Get BTC data, loading from disk if needed."""
        if self._btc_data is None:
            self._btc_data = _load_btc_data(self._data_dir, self._interval)
        return self._btc_data

    @property
    def exit_price_column(self) -> str:
        """VBOV1 exits at next day's open price."""
        return "exit_price_base"

    @classmethod
    def parameter_schema(cls) -> dict[str, object]:
        """Parameter schema for optimization sweep."""
        return {
            "noise_ratio": {"type": "float", "min": 0.1, "max": 0.9, "step": 0.1},
            "ma_short": {"type": "int", "min": 3, "max": 20, "step": 1},
            "btc_ma": {"type": "int", "min": 5, "max": 50, "step": 5},
        }

    def required_indicators(self) -> list[str]:
        return [
            "sma",
            "target",
            "prev_range",
            "btc_above_ma",
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VBO indicators with fixed K and current-bar SMA."""
        df = df.copy()

        # SMA includes current bar (no shift) — matches v1
        df["sma"] = df["close"].rolling(window=self.ma_short).mean()

        # Previous day range
        df["prev_high"] = df["high"].shift(1)
        df["prev_low"] = df["low"].shift(1)
        df["prev_range"] = df["prev_high"] - df["prev_low"]

        # Target with fixed K value — matches v1
        df["target"] = df["open"] + df["prev_range"] * self.noise_ratio

        # BTC market filter
        btc_df = self._get_btc_data()
        if btc_df is not None and not btc_df.empty:
            btc_close = btc_df["close"].reindex(df.index, method="ffill")
            btc_ma = btc_close.rolling(window=self.btc_ma).mean()

            # Use previous day's values to avoid look-ahead bias
            df["btc_above_ma"] = btc_close.shift(1) > btc_ma.shift(1)
        else:
            # No BTC data: don't filter (always allow)
            df["btc_above_ma"] = True

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate V1 signals: breakout+BTC entry, prev_close<prev_SMA exit at open."""
        df = df.copy()

        # Entry: breakout AND BTC filter (no SMA filter — matches v1)
        entry_breakout = df["high"] >= df["target"]
        entry_btc = df["btc_above_ma"].fillna(False)
        df["entry_signal"] = entry_breakout & entry_btc

        # Exit: previous day's close < previous day's SMA
        # When this fires on day T, we sell at day T's open
        df["exit_signal"] = df["close"].shift(1) < df["sma"].shift(1)

        # Exit at open price (not close) — the key V1 difference
        df["exit_price_base"] = df["open"]

        return df


__all__ = ["VBOV1"]
