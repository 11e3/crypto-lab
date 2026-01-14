"""
Unified metrics calculation module.

Provides core metric calculations used across backtester, web, and analysis modules.
"""

from src.metrics.core import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_var,
    calculate_volatility,
)

__all__ = [
    "calculate_returns",
    "calculate_total_return",
    "calculate_cagr",
    "calculate_max_drawdown",
    "calculate_volatility",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_var",
    "calculate_cvar",
]
