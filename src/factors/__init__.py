"""
Factor-based investment framework.

Provides:
- Factor calculation (momentum, value, quality, volatility, size)
- Cross-sectional factor scoring
- Multi-factor composite models
- Factor exposure analysis
"""

from src.factors.base import Factor, FactorScore, FactorExposure
from src.factors.momentum import MomentumFactor
from src.factors.volatility import VolatilityFactor
from src.factors.value import ValueFactor
from src.factors.quality import QualityFactor
from src.factors.composite import CompositeFactorModel, FactorWeight

__all__ = [
    # Base
    "Factor",
    "FactorScore",
    "FactorExposure",
    # Factors
    "MomentumFactor",
    "VolatilityFactor",
    "ValueFactor",
    "QualityFactor",
    # Composite
    "CompositeFactorModel",
    "FactorWeight",
]
