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
from src.strategies.volatility_breakout.vbo_portfolio import (
    VBOPortfolio,
    VBOPortfolioLite,
    VBOSingleCoin,
)
from src.strategies.volatility_breakout.vbo_hybrid import VBOHybrid
from src.strategies.volatility_breakout.vbo_regime import VBORegime
from src.strategies.volatility_breakout.vbo_v1 import VBOV1

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
    "VBOPortfolio",
    "VBOPortfolioLite",
    "VBOSingleCoin",
    "VBOHybrid",
    "VBORegime",
    "VBOV1",
]
