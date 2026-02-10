"""BTC market filter conditions for VBO strategies.

Provides cross-asset market filtering using BTC price as a
market regime indicator. When BTC is above its MA, the broader
crypto market is considered bullish.

Ported from bt framework's VBOPortfolioBuyCondition/SellCondition
and adapted to CQS vectorized paradigm.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import OHLCV, Condition


class BtcMarketFilterCondition(Condition):
    """Entry filter: only trade when BTC is above its MA.

    In vectorized mode, expects 'btc_above_ma' column in DataFrame.
    In event-driven mode, checks 'btc_above_ma' indicator.

    Args:
        btc_ma_key: Key for BTC MA filter indicator
        name: Condition name
    """

    def __init__(
        self,
        btc_ma_key: str = "btc_above_ma",
        name: str = "BtcMarketFilter",
    ) -> None:
        super().__init__(name)
        self.btc_ma_key = btc_ma_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if BTC is above its MA."""
        btc_above_ma = indicators.get(self.btc_ma_key)
        if btc_above_ma is None:
            return False
        return bool(btc_above_ma)


class BtcMarketExitCondition(Condition):
    """Exit filter: sell when BTC drops below its MA.

    In vectorized mode, expects 'btc_below_ma' column in DataFrame.
    In event-driven mode, checks 'btc_below_ma' indicator.

    Args:
        btc_ma_key: Key for BTC MA exit indicator
        name: Condition name
    """

    def __init__(
        self,
        btc_ma_key: str = "btc_below_ma",
        name: str = "BtcMarketExit",
    ) -> None:
        super().__init__(name)
        self.btc_ma_key = btc_ma_key

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if BTC is below its MA (trigger exit)."""
        btc_below_ma = indicators.get(self.btc_ma_key)
        if btc_below_ma is None:
            return False
        return bool(btc_below_ma)


__all__ = [
    "BtcMarketFilterCondition",
    "BtcMarketExitCondition",
]
