"""VBO Portfolio Strategy with BTC Market Filter.

Multi-asset VBO strategy that uses BTC's MA20 as a market regime
filter. Only enters positions when BTC is in an uptrend.

Ported from bt framework's VBOPortfolioStrategy and adapted to
CQS vectorized paradigm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.strategies.base import Condition, Strategy
from src.strategies.volatility_breakout.conditions import (
    BreakoutCondition,
    PriceBelowSMACondition,
    SMABreakoutCondition,
)
from src.strategies.volatility_breakout.conditions_btc_filter import (
    BtcMarketExitCondition,
    BtcMarketFilterCondition,
)
from src.strategies.volatility_breakout.vbo_indicators import calculate_vbo_indicators


def _load_btc_data(data_dir: Path, interval: str = "day") -> pd.DataFrame | None:
    """Load BTC parquet data for market filter."""
    file_path = data_dir / f"KRW-BTC_{interval}.parquet"
    if not file_path.exists():
        return None
    df = pd.read_parquet(file_path)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str.lower()
    return df


class VBOPortfolio(Strategy):
    """VBO with BTC MA20 market filter.

    Entry conditions (AND):
    - Breakout: high >= target (open + range * noise_ratio)
    - SMA filter: target > SMA(ma_short)
    - BTC filter: prev BTC close > prev BTC MA(btc_ma)

    Exit conditions (OR):
    - Price below SMA: close < SMA(ma_short)
    - BTC filter exit: prev BTC close < prev BTC MA(btc_ma)

    Args:
        name: Strategy name
        ma_short: Short MA period for individual coins
        btc_ma: BTC MA period for market filter
        noise_ratio: K-factor for VBO breakout (not used as indicator param,
                     but stored for reference)
        btc_data: Pre-loaded BTC DataFrame (optional, loaded from disk if None)
        data_dir: Directory containing parquet files
        interval: Data interval for BTC file lookup
    """

    def __init__(
        self,
        name: str = "VBOPortfolio",
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

        entry_conditions: list[Condition] = [
            BreakoutCondition(),
            SMABreakoutCondition(),
            BtcMarketFilterCondition(),
        ]
        exit_conditions: list[Condition] = [
            PriceBelowSMACondition(),
            BtcMarketExitCondition(),
        ]

        super().__init__(
            name=name,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
        )

    def _get_btc_data(self) -> pd.DataFrame | None:
        """Get BTC data, loading from disk if needed."""
        if self._btc_data is None:
            self._btc_data = _load_btc_data(self._data_dir, self._interval)
        return self._btc_data

    def required_indicators(self) -> list[str]:
        return [
            "noise",
            "sma",
            "target",
            "prev_range",
            "btc_above_ma",
            "btc_below_ma",
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VBO indicators + BTC market filter columns."""
        df = calculate_vbo_indicators(
            df,
            sma_period=self.ma_short,
            trend_sma_period=self.ma_short * 2,
            short_noise_period=self.ma_short,
            long_noise_period=self.ma_short * 2,
            exclude_current=True,
        )

        # Add BTC market filter indicators
        btc_df = self._get_btc_data()
        if btc_df is not None:
            btc_close = btc_df["close"].reindex(df.index, method="ffill")
            btc_ma = btc_close.rolling(window=self.btc_ma).mean()

            # Use previous day's values to avoid look-ahead bias
            prev_btc_close = btc_close.shift(1)
            prev_btc_ma = btc_ma.shift(1)

            df["btc_above_ma"] = prev_btc_close > prev_btc_ma
            df["btc_below_ma"] = prev_btc_close < prev_btc_ma
        else:
            # No BTC data: don't filter (always allow)
            df["btc_above_ma"] = True
            df["btc_below_ma"] = False

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry/exit signals with BTC market filter."""
        df = df.copy()

        # Entry: breakout AND sma filter AND btc bull
        entry_breakout = df["high"] >= df["target"]
        entry_sma = df["target"] > df["sma"]
        entry_btc = df["btc_above_ma"].fillna(False)

        df["entry_signal"] = entry_breakout & entry_sma & entry_btc

        # Exit: price below sma OR btc bear (OR logic)
        exit_sma = df["close"] < df["sma"]
        exit_btc = df["btc_below_ma"].fillna(False)

        df["exit_signal"] = exit_sma | exit_btc

        return df


class VBOSingleCoin(VBOPortfolio):
    """Single-asset VBO with BTC MA filter and all-in allocation.

    Same logic as VBOPortfolio but intended for single-asset backtesting
    with 100% capital allocation (no 1/N split).
    """

    def __init__(
        self,
        name: str = "VBOSingleCoin",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)


class VBOPortfolioLite(VBOPortfolio):
    """VBO with BTC MA filter only (no SMA filters).

    A lighter variant of VBOPortfolio that removes SMA from entry
    conditions and BTC exit from exit conditions, relying purely on:
    - Entry: breakout + BTC bull market filter
    - Exit: price below SMA only (no BTC bear exit)

    This reduces whipsaw from SMA entry filtering and avoids
    premature exits caused by short-term BTC MA crossovers.
    """

    def __init__(
        self,
        name: str = "VBOPortfolioLite",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        # Override conditions: remove SMA from entry, remove BTC exit
        self.entry_conditions.conditions = [
            c for c in self.entry_conditions.conditions
            if not isinstance(c, SMABreakoutCondition)
        ]
        self.exit_conditions.conditions = [
            c for c in self.exit_conditions.conditions
            if not isinstance(c, BtcMarketExitCondition)
        ]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals: breakout + BTC filter entry, SMA-only exit."""
        df = df.copy()

        # Entry: breakout AND btc bull (no SMA filter)
        entry_breakout = df["high"] >= df["target"]
        entry_btc = df["btc_above_ma"].fillna(False)

        df["entry_signal"] = entry_breakout & entry_btc

        # Exit: price below SMA only (no BTC bear exit)
        df["exit_signal"] = df["close"] < df["sma"]

        return df


__all__ = [
    "VBOPortfolio",
    "VBOPortfolioLite",
    "VBOSingleCoin",
]
