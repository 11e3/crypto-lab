"""
Entry and exit conditions for Momentum strategy.

These conditions use momentum indicators like RSI, MACD, and moving averages
to identify trend-following opportunities.
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition
from src.strategies.common_conditions import (  # noqa: E402
    PriceAboveSMACondition,
    PriceBelowSMACondition,
    RSIOverboughtCondition,
    RSIOversoldCondition,
)

# Re-export MACD conditions for backward compatibility
from src.strategies.momentum.conditions_macd import (  # noqa: E402
    MACDBearishCondition,
    MACDBullishCondition,
)


class MomentumStrengthCondition(Condition):
    """
    Condition: Momentum strength based on price change over period.

    Filters for strong momentum moves.
    """

    def __init__(
        self,
        lookback: int = 10,
        min_change_pct: float = 0.02,
        name: str = "MomentumStrength",
    ) -> None:
        """
        Initialize momentum strength condition.

        Args:
            lookback: Lookback period for momentum calculation
            min_change_pct: Minimum price change percentage (default 2%)
            name: Condition name
        """
        super().__init__(name)
        self.lookback = lookback
        self.min_change_pct = min_change_pct

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if momentum is strong enough."""
        if len(history) < self.lookback:
            return False

        # Calculate price change over lookback period
        past_close: float = float(history.iloc[-self.lookback]["close"])
        if past_close <= 0:
            return False

        change_pct: float = float((current.close - past_close) / past_close)
        return change_pct >= self.min_change_pct


__all__ = [
    "PriceAboveSMACondition",
    "PriceBelowSMACondition",
    "MomentumStrengthCondition",
    # Re-exported from conditions_rsi
    "RSIOversoldCondition",
    "RSIOverboughtCondition",
    # Re-exported from conditions_macd
    "MACDBullishCondition",
    "MACDBearishCondition",
]
