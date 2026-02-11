"""
Advanced order data models.

Contains data structures for advanced order types:
- OrderType: Enum for order types (stop loss, take profit, trailing stop)
- AdvancedOrder: Dataclass representing a conditional order
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum


class OrderType(Enum):
    """Type of advanced order."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class AdvancedOrder:
    """
    Advanced order for risk management.

    Represents a conditional order that triggers based on price movements.
    """

    order_id: str
    ticker: str
    order_type: OrderType
    entry_price: float
    entry_date: date
    amount: float

    # Stop Loss parameters
    stop_loss_price: float | None = None
    stop_loss_pct: float | None = None  # Percentage below entry price

    # Take Profit parameters
    take_profit_price: float | None = None
    take_profit_pct: float | None = None  # Percentage above entry price

    # Trailing Stop parameters
    trailing_stop_pct: float | None = None  # Percentage to trail from peak
    trailing_stop_atr_multiplier: float | None = None  # ATR-based trailing stop
    highest_price: float | None = None  # Track highest price for trailing stop

    # Status
    is_active: bool = True
    is_triggered: bool = False
    triggered_price: float | None = None
    triggered_date: date | None = None

    def __repr__(self) -> str:
        return (
            f"AdvancedOrder({self.order_type.value}, {self.ticker}, "
            f"entry={self.entry_price:.0f}, active={self.is_active})"
        )
