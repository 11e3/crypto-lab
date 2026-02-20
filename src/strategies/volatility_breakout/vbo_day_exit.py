"""VBODayExit — VBO variant with unconditional next-day open exit.

Same entry as VBOV1 (breakout + BTC MA filter).
Exit: always exit at the next day's open after entry.
No MA-based exit — pure 1-day holding period.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.strategies.volatility_breakout.vbo_v1 import VBOV1


class VBODayExit(VBOV1):
    """VBO with fixed 1-day holding: enter at target, exit at next open.

    Entry conditions (AND):
    - Breakout: high >= open + prev_range * noise_ratio
    - BTC filter: prev BTC close > prev BTC MA(btc_ma)

    Exit condition:
    - Always exit the day after entry (at that day's open)

    Args:
        noise_ratio: Breakout coefficient K (default=0.5)
        btc_ma: BTC MA period for market filter (default=20)
        data_dir: Directory containing parquet files
        interval: Data interval for BTC file lookup
    """

    def __init__(self, name: str = "VBODayExit", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().generate_signals(df)

        # Override exit: always exit next day at open regardless of price
        df["exit_signal"] = (
            df["entry_signal"].shift(1).infer_objects(copy=False).fillna(value=False)
        )

        return df

    @classmethod
    def parameter_schema(cls) -> dict[str, object]:
        return {
            "noise_ratio": {"type": "float", "min": 0.1, "max": 0.9, "step": 0.1},
            "btc_ma": {"type": "int", "min": 5, "max": 80, "step": 5},
        }


__all__ = ["VBODayExit"]
