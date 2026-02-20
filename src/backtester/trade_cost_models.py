"""
Trade execution data models.

Contains data structures for trade execution information.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TradeExecution:
    """Trade execution record."""

    timestamp: pd.Timestamp
    side: str  # 'buy' or 'sell'
    entry_price: float
    entry_size: float
    exit_price: float = 0.0
    exit_size: float = 0.0
    exit_time: pd.Timestamp | None = None

    # Cost components
    entry_slippage_pct: float = 0.0
    exit_slippage_pct: float = 0.0
    maker_fee_pct: float = 0.0
    taker_fee_pct: float = 0.05

    # Computed results
    gross_pnl: float = 0.0
    slippage_cost: float = 0.0
    fee_cost: float = 0.0
    net_pnl: float = 0.0
    net_pnl_pct: float = 0.0


class UpbitFeeStructure:
    """
    Upbit exchange fee structure.

    Source: Upbit official documentation

    Spot trading:
    - Maker (limit order): 0.05% (0% for some pairs)
    - Taker (market order): 0.05%

    Key features:
    - KRW fee support (fees deducted directly in KRW)
    - VIP program (volume-based discounts)
    - Last-minute trade fee rebate
    """

    # Default fee rates
    DEFAULT_MAKER_FEE = 0.05  # %
    DEFAULT_TAKER_FEE = 0.05  # %

    # Fee tiers by monthly trading volume (volume-based)
    VIP_TIERS = {
        0: {"maker": 0.05, "taker": 0.05},  # Default
        1: {"maker": 0.04, "taker": 0.05},  # monthly volume >= 100 BTC
        2: {"maker": 0.03, "taker": 0.04},  # >= 500 BTC
        3: {"maker": 0.02, "taker": 0.03},  # >= 1000 BTC
        4: {"maker": 0.01, "taker": 0.02},  # >= 5000 BTC
    }

    @staticmethod
    def get_fees(tier: int = 0) -> dict[str, float]:
        """
        Get fee rates for a given VIP tier.

        Args:
            tier: VIP tier (0-4)

        Returns:
            {'maker': 0.XX%, 'taker': 0.XX%}
        """
        tier = max(0, min(tier, 4))
        return UpbitFeeStructure.VIP_TIERS[tier].copy()


__all__ = ["TradeExecution", "UpbitFeeStructure"]
