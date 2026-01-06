"""Backtester package."""

from src.backtester.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
    VectorizedBacktestEngine,
    run_backtest,
)
from src.backtester.monte_carlo import (
    MonteCarloResult,
    MonteCarloSimulator,
    run_monte_carlo,
)
from src.backtester.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardPeriod,
    WalkForwardResult,
    run_walk_forward_analysis,
)
from src.backtester.optimization import (
    OptimizationResult,
    ParameterOptimizer,
    optimize_strategy_parameters,
)
from src.backtester.parallel import (
    ParallelBacktestRunner,
    ParallelBacktestTask,
    compare_strategies,
    optimize_parameters,
)
from src.backtester.report import (
    BacktestReport,
    PerformanceMetrics,
    generate_report,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BacktestReport",
    "PerformanceMetrics",
    "Trade",
    "VectorizedBacktestEngine",
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
