"""
Execution algorithms for order optimization.

Provides:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Implementation Shortfall
- Market Impact modeling
"""

from src.execution.algorithms.base import (
    ExecutionAlgorithm,
    OrderSlice,
    ExecutionResult,
)
from src.execution.algorithms.twap import TWAPAlgorithm
from src.execution.algorithms.vwap import VWAPAlgorithm
from src.execution.algorithms.market_impact import MarketImpactModel

__all__ = [
    "ExecutionAlgorithm",
    "OrderSlice",
    "ExecutionResult",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "MarketImpactModel",
]
