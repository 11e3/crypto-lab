"""
Data models for Robustness Analysis.

Defines RobustnessResult and RobustnessReport dataclasses.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RobustnessResult:
    """Performance result for a single parameter combination."""

    params: dict[str, Any]
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trade_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary (params merged with metrics)."""
        return {
            **self.params,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "trade_count": self.trade_count,
        }


@dataclass
class RobustnessReport:
    """Aggregate robustness analysis report."""

    optimal_params: dict[str, Any]
    results: list[RobustnessResult]

    # Return distribution statistics
    mean_return: float = 0.0
    std_return: float = 0.0
    min_return: float = 0.0
    max_return: float = 0.0

    # Fraction of ±20% neighbors achieving ≥80% of optimal return
    neighbor_success_rate: float = 0.0  # 0.0–1.0

    # Per-parameter sensitivity scores
    sensitivity_scores: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """Post-initialization to set default sensitivity scores."""
        if self.sensitivity_scores is None:
            self.sensitivity_scores = {}
