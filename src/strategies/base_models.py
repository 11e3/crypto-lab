"""
Base data models for strategy abstraction.

Contains core dataclasses and enums used by trading strategies:
- SignalType: BUY/SELL/HOLD signal types
- Signal: Trading signal with metadata
- Position: Trading position representation
- OHLCV: Single bar OHLCV data
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from typing import Any


class SignalType(Enum):
    """Trading signal types."""

    BUY = auto()
    SELL = auto()
    HOLD = auto()


@dataclass
class Signal:
    """Trading signal with metadata."""

    signal_type: SignalType
    ticker: str
    price: float
    date: date
    strength: float = 1.0  # Signal strength/confidence (0-1)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Represents a trading position."""

    ticker: str
    amount: float
    entry_price: float
    entry_date: date
    highest_price: float | None = None  # For trailing stop tracking (event-driven engine)


@dataclass
class OHLCV:
    """Single OHLCV bar data."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range(self) -> float:
        """Price range (high - low)."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Candle body size."""
        return abs(self.close - self.open)


__all__ = ["SignalType", "Signal", "Position", "OHLCV"]
