"""
Alpha generation pipeline.

Provides:
- Signal generation from multiple sources
- Signal combination and weighting
- Alpha decay analysis
- Signal quality metrics
"""

from src.alpha.signals import (
    AlphaSignal,
    TechnicalSignal,
    MomentumSignal,
    MeanReversionSignal,
    BreakoutSignal,
)
from src.alpha.pipeline import (
    AlphaPipeline,
    SignalWeight,
    PipelineResult,
)
from src.alpha.combination import (
    SignalCombiner,
    CombinationMethod,
)

__all__ = [
    # Signals
    "AlphaSignal",
    "TechnicalSignal",
    "MomentumSignal",
    "MeanReversionSignal",
    "BreakoutSignal",
    # Pipeline
    "AlphaPipeline",
    "SignalWeight",
    "PipelineResult",
    # Combination
    "SignalCombiner",
    "CombinationMethod",
]
