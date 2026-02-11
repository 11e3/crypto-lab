"""Backtester package.

Uses lazy imports to avoid loading pyupbit unless needed for Upbit-specific functionality.
"""

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BacktestReport",
    "PerformanceMetrics",
    "Trade",
    "VectorizedBacktestEngine",
    "EventDrivenBacktestEngine",
    "generate_report",
    "run_backtest",
    "ParallelBacktestRunner",
    "ParallelBacktestTask",
    "compare_strategies",
    "optimize_parameters",
    "OptimizationResult",
    "ParameterOptimizer",
    "optimize_strategy_parameters",
    "MonteCarloResult",
    "MonteCarloSimulator",
    "run_monte_carlo",
    "WalkForwardAnalyzer",
    "WalkForwardPeriod",
    "WalkForwardResult",
    "run_walk_forward_analysis",
]

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Models
    "BacktestConfig": ("src.backtester.models", "BacktestConfig"),
    "BacktestResult": ("src.backtester.models", "BacktestResult"),
    "Trade": ("src.backtester.models", "Trade"),
    # Monte Carlo
    "MonteCarloResult": ("src.backtester.analysis.monte_carlo", "MonteCarloResult"),
    "MonteCarloSimulator": ("src.backtester.analysis.monte_carlo", "MonteCarloSimulator"),
    "run_monte_carlo": ("src.backtester.analysis.monte_carlo", "run_monte_carlo"),
    # Engine
    "BacktestEngine": ("src.backtester.engine", "BacktestEngine"),
    "EventDrivenBacktestEngine": ("src.backtester.engine", "EventDrivenBacktestEngine"),
    "VectorizedBacktestEngine": ("src.backtester.engine", "VectorizedBacktestEngine"),
    "run_backtest": ("src.backtester.engine", "run_backtest"),
    # Optimization
    "OptimizationResult": ("src.backtester.optimization", "OptimizationResult"),
    "ParameterOptimizer": ("src.backtester.optimization", "ParameterOptimizer"),
    "optimize_strategy_parameters": (
        "src.backtester.optimization",
        "optimize_strategy_parameters",
    ),
    # Parallel
    "ParallelBacktestRunner": ("src.backtester.parallel", "ParallelBacktestRunner"),
    "ParallelBacktestTask": ("src.backtester.parallel", "ParallelBacktestTask"),
    "compare_strategies": ("src.backtester.parallel", "compare_strategies"),
    "optimize_parameters": ("src.backtester.parallel", "optimize_parameters"),
    # Report
    "BacktestReport": ("src.backtester.report_pkg.report", "BacktestReport"),
    "PerformanceMetrics": ("src.backtester.report_pkg.report", "PerformanceMetrics"),
    "generate_report": ("src.backtester.report_pkg.report", "generate_report"),
    # Walk Forward Analysis
    "WalkForwardAnalyzer": ("src.backtester.wfa.walk_forward", "WalkForwardAnalyzer"),
    "WalkForwardPeriod": ("src.backtester.wfa.walk_forward", "WalkForwardPeriod"),
    "WalkForwardResult": ("src.backtester.wfa.walk_forward", "WalkForwardResult"),
    "run_walk_forward_analysis": (
        "src.backtester.wfa.walk_forward",
        "run_walk_forward_analysis",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy import to avoid loading pyupbit unless needed."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
