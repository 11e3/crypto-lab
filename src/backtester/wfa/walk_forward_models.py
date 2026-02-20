"""
Walk-forward analysis data models.

Contains dataclasses for periods and results.
"""

from dataclasses import dataclass
from datetime import date

from src.backtester.models import BacktestResult
from src.backtester.optimization import OptimizationResult


@dataclass
class WalkForwardPeriod:
    """Single period in walk-forward analysis."""

    period_num: int
    optimization_start: date
    optimization_end: date
    test_start: date
    test_end: date
    optimization_result: OptimizationResult | None = None
    test_result: BacktestResult | None = None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WalkForwardPeriod({self.period_num}, "
            f"opt={self.optimization_start} to {self.optimization_end}, "
            f"test={self.test_start} to {self.test_end})"
        )


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""

    periods: list[WalkForwardPeriod]
    overall_test_result: BacktestResult | None = None

    # Aggregate statistics
    avg_test_cagr: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_test_sortino: float = 0.0
    avg_test_mdd: float = 0.0
    avg_optimization_cagr: float = 0.0

    # Consistency metrics
    positive_periods: int = 0
    total_periods: int = 0
    consistency_rate: float = 0.0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WalkForwardResult({self.total_periods} periods, "
            f"avg_test_cagr={self.avg_test_cagr:.2f}%)"
        )
