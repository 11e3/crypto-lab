"""Strategies package."""

from src.strategies.base import (
    OHLCV,
    CompositeCondition,
    Condition,
    Position,
    Signal,
    SignalType,
    Strategy,
)
from src.strategies.registry import registry
from src.strategies.volatility_breakout import VBOV1, VBODayExit

# Register all strategies with the canonical registry
registry.register("VBO", VBOV1)
registry.register("VBO_DAY", VBODayExit)

__all__ = [
    "Condition",
    "CompositeCondition",
    "OHLCV",
    "Position",
    "Signal",
    "SignalType",
    "Strategy",
    "VBOV1",
    "VBODayExit",
    "registry",
]
