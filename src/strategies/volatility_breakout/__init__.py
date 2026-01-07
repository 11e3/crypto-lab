"""Volatility Breakout Strategy package."""

import warnings

from src.strategies.volatility_breakout.conditions import (
    NoiseFilter,  # Backward compatibility alias (deprecated)
)
from src.strategies.volatility_breakout.conditions import (
    TrendFilter,  # Backward compatibility alias (deprecated)
)
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

# Warn when deprecated aliases are imported
if __name__ != "__main__":
    warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)

__all__ = [
    "BreakoutCondition",
    "SMABreakoutCondition",
    "WhipsawExitCondition",
    "TrendCondition",
    "NoiseCondition",
    "TrendFilter",  # Deprecated: Use TrendCondition
    "NoiseFilter",  # Deprecated: Use NoiseCondition
    "VanillaVBO",
    "MinimalVBO",
    "StrictVBO",
    "create_vbo_strategy",
    "quick_vbo",
]
