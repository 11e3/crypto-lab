"""
Search algorithms for parameter optimization.

Contains grid search and random search implementations.
"""

import random
from collections.abc import Callable
from itertools import product
from typing import TYPE_CHECKING, Any

from src.backtester.models import BacktestConfig, BacktestResult
from src.backtester.optimization_models import OptimizationResult
from src.backtester.parallel import ParallelBacktestRunner, ParallelBacktestTask
from src.strategies.base import Strategy
from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def grid_search(
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    param_grid: dict[str, list[Any]],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    metric: str,
    maximize: bool,
    n_workers: int | None = None,
) -> OptimizationResult:
    """Perform grid search over parameter space.

    Tests all combinations of parameters from param_grid.

    Args:
        strategy_factory: Function that creates a strategy from parameters
        param_grid: Parameter names to value lists mapping
        tickers: List of tickers to backtest
        interval: Data interval
        config: Backtest configuration
        metric: Metric to optimize
        maximize: If True, maximize metric
        n_workers: Number of parallel workers

    Returns:
        OptimizationResult with best parameters
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    logger.info(f"Grid search: {len(combinations)} parameter combinations")

    tasks = []
    for combo in combinations:
        params = dict(zip(param_names, combo, strict=True))
        strategy = strategy_factory(params)
        task_name = f"{strategy.name}_{'_'.join(str(v) for v in combo)}"
        tasks.append(
            ParallelBacktestTask(
                name=task_name,
                strategy=strategy,
                tickers=tickers,
                interval=interval,
                config=config,
                params=params,
            )
        )

    runner = ParallelBacktestRunner(n_workers=n_workers)
    results = runner.run(tasks)

    return _collect_results(tasks, results, metric, maximize)


def random_search(
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    param_grid: dict[str, list[Any]],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    metric: str,
    maximize: bool,
    n_iter: int,
    n_workers: int | None = None,
) -> OptimizationResult:
    """Perform random search over parameter space.

    Randomly samples n_iter parameter combinations.

    Args:
        strategy_factory: Function that creates a strategy from parameters
        param_grid: Parameter names to value lists mapping
        tickers: List of tickers to backtest
        interval: Data interval
        config: Backtest configuration
        metric: Metric to optimize
        maximize: If True, maximize metric
        n_iter: Number of random iterations
        n_workers: Number of parallel workers

    Returns:
        OptimizationResult with best parameters
    """
    logger.info(f"Random search: {n_iter} iterations")

    tasks = []
    for i in range(n_iter):
        params = {name: random.choice(values) for name, values in param_grid.items()}
        strategy = strategy_factory(params)
        task_name = f"{strategy.name}_iter{i}"
        tasks.append(
            ParallelBacktestTask(
                name=task_name,
                strategy=strategy,
                tickers=tickers,
                interval=interval,
                config=config,
                params=params,
            )
        )

    runner = ParallelBacktestRunner(n_workers=n_workers)
    results = runner.run(tasks)

    return _collect_results(tasks, results, metric, maximize)


def _collect_results(
    tasks: list[ParallelBacktestTask],
    results: dict[str, BacktestResult],
    metric: str,
    maximize: bool,
) -> OptimizationResult:
    """Collect and sort optimization results."""
    all_results: list[tuple[dict[str, Any], BacktestResult, float]] = []

    for task in tasks:
        result = results.get(task.name)
        if result and task.params:
            score = _extract_metric(result, metric)
            all_results.append((task.params, result, score))

    all_results.sort(key=lambda x: x[2], reverse=maximize)

    best_params, best_result, best_score = (
        all_results[0] if all_results else ({}, BacktestResult(), 0.0)
    )

    return OptimizationResult(
        best_params=best_params,
        best_result=best_result,
        best_score=best_score,
        all_results=all_results,
        optimization_metric=metric,
    )


def _extract_metric(result: BacktestResult, metric: str) -> float:
    """Extract metric value from backtest result."""
    metric_map = {
        "sharpe_ratio": "sharpe_ratio",
        "cagr": "cagr",
        "total_return": "total_return",
        "calmar_ratio": "calmar_ratio",
        "win_rate": "win_rate",
        "profit_factor": "profit_factor",
    }
    attr = metric_map.get(metric, "sharpe_ratio")
    return getattr(result, attr, 0.0)


__all__ = ["grid_search", "random_search"]
