"""
Exit conditions for Volatility Breakout strategy.

These conditions determine when to exit a position.
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition
from src.strategies.common_conditions import PriceBelowSMACondition  # noqa: F401


class WhipsawExitCondition(Condition):
    """
    Exit condition for same-day whipsaw detection.

    Triggers when price breaks out but then falls below SMA
    on the same candle, indicating a false breakout.
    """

    def __init__(
        self,
        sma_key: str = "sma",
        name: str = "WhipsawExit",
    ) -> None:
        """Initialize condition."""
        super().__init__(name)
        self.sma_key = sma_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check for whipsaw condition."""
        target = indicators.get("target")
        sma = indicators.get(self.sma_key)

        if target is None or sma is None:
            return False

        # Whipsaw: broke out (high >= target) but closed below SMA
        breakout_occurred = current.high >= target
        closed_below_sma = current.close < sma

        return breakout_occurred and closed_below_sma
