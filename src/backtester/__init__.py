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
    "SimpleBacktestEngine",
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

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import to avoid loading pyupbit unless needed."""
    # Models
    if name == "BacktestConfig":
        from src.backtester.models import BacktestConfig

        return BacktestConfig
    elif name == "BacktestResult":
        from src.backtester.models import BacktestResult

        return BacktestResult
    elif name == "Trade":
        from src.backtester.models import Trade

        return Trade

    # Monte Carlo
    elif name == "MonteCarloResult":
        from src.backtester.analysis.monte_carlo import MonteCarloResult

        return MonteCarloResult
    elif name == "MonteCarloSimulator":
        from src.backtester.analysis.monte_carlo import MonteCarloSimulator

        return MonteCarloSimulator
    elif name == "run_monte_carlo":
        from src.backtester.analysis.monte_carlo import run_monte_carlo

        return run_monte_carlo

    # Engine
    elif name == "BacktestEngine":
        from src.backtester.engine import BacktestEngine

        return BacktestEngine
    elif name == "EventDrivenBacktestEngine":
        from src.backtester.engine import EventDrivenBacktestEngine

        return EventDrivenBacktestEngine
    elif name == "SimpleBacktestEngine":
        from src.backtester.engine import SimpleBacktestEngine

        return SimpleBacktestEngine
    elif name == "VectorizedBacktestEngine":
        from src.backtester.engine import VectorizedBacktestEngine

        return VectorizedBacktestEngine
    elif name == "run_backtest":
        from src.backtester.engine import run_backtest

        return run_backtest

    # Optimization
    elif name == "OptimizationResult":
        from src.backtester.optimization import OptimizationResult

        return OptimizationResult
    elif name == "ParameterOptimizer":
        from src.backtester.optimization import ParameterOptimizer

        return ParameterOptimizer
    elif name == "optimize_strategy_parameters":
        from src.backtester.optimization import optimize_strategy_parameters

        return optimize_strategy_parameters

    # Parallel
    elif name == "ParallelBacktestRunner":
        from src.backtester.parallel import ParallelBacktestRunner

        return ParallelBacktestRunner
    elif name == "ParallelBacktestTask":
        from src.backtester.parallel import ParallelBacktestTask

        return ParallelBacktestTask
    elif name == "compare_strategies":
        from src.backtester.parallel import compare_strategies

        return compare_strategies
    elif name == "optimize_parameters":
        from src.backtester.parallel import optimize_parameters

        return optimize_parameters

    # Report
    elif name == "BacktestReport":
        from src.backtester.report_pkg.report import BacktestReport

        return BacktestReport
    elif name == "PerformanceMetrics":
        from src.backtester.report_pkg.report import PerformanceMetrics

        return PerformanceMetrics
    elif name == "generate_report":
        from src.backtester.report_pkg.report import generate_report

        return generate_report

    # Walk Forward Analysis
    elif name == "WalkForwardAnalyzer":
        from src.backtester.wfa.walk_forward import WalkForwardAnalyzer

        return WalkForwardAnalyzer
    elif name == "WalkForwardPeriod":
        from src.backtester.wfa.walk_forward import WalkForwardPeriod

        return WalkForwardPeriod
    elif name == "WalkForwardResult":
        from src.backtester.wfa.walk_forward import WalkForwardResult

        return WalkForwardResult
    elif name == "run_walk_forward_analysis":
        from src.backtester.wfa.walk_forward import run_walk_forward_analysis

        return run_walk_forward_analysis

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
