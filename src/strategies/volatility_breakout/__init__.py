"""Volatility Breakout Strategy package."""

from src.strategies.volatility_breakout.conditions import (
    BreakoutCondition,
    NoiseCondition,
    SMABreakoutCondition,
    TrendCondition,
    WhipsawExitCondition,
)
from src.strategies.volatility_breakout.vbo import (
    MinimalVBO,
    StrictVBO,
    VanillaVBO,
    create_vbo_strategy,
    quick_vbo,
)

__all__ = [
    "BreakoutCondition",
    "SMABreakoutCondition",
    "WhipsawExitCondition",
    "TrendCondition",
    "NoiseCondition",
    "VanillaVBO",
    "MinimalVBO",
    "StrictVBO",
    "create_vbo_strategy",
    "quick_vbo",
]
