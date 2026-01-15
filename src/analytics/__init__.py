"""
Performance analytics module.

Provides:
- Performance attribution (Brinson, Factor-based)
- Risk decomposition
- Return analysis
- Benchmark comparison
"""

from src.analytics.attribution import (
    PerformanceAttribution,
    BrinsonResult,
    FactorAttributionResult,
    ReturnDecomposition,
)
from src.analytics.returns import (
    ReturnAnalyzer,
    RollingMetrics,
    DrawdownAnalysis,
)

__all__ = [
    # Attribution
    "PerformanceAttribution",
    "BrinsonResult",
    "FactorAttributionResult",
    "ReturnDecomposition",
    # Returns
    "ReturnAnalyzer",
    "RollingMetrics",
    "DrawdownAnalysis",
]
