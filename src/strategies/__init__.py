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
from src.strategies.mean_reversion import (
    MeanReversionStrategy,
    SimpleMeanReversionStrategy,
)
from src.strategies.momentum import MomentumStrategy, SimpleMomentumStrategy
from src.strategies.pair_trading import PairTradingStrategy
from src.strategies.volatility_breakout import (
    MinimalVBO,
    StrictVBO,
    VanillaVBO,
    create_vbo_strategy,
    quick_vbo,
)

__all__ = [
    "Condition",
    "CompositeCondition",
    "OHLCV",
    "Position",
    "Signal",
    "SignalType",
    "Strategy",
    # Mean Reversion strategies
    "MeanReversionStrategy",
    "SimpleMeanReversionStrategy",
    # Pair Trading strategies
    "PairTradingStrategy",
    # Momentum strategies
    "MomentumStrategy",
    "SimpleMomentumStrategy",
    # VBO strategies
    "VanillaVBO",
    "MinimalVBO",
    "StrictVBO",
    "create_vbo_strategy",
    "quick_vbo",
]
