"""
Parameter optimization for trading strategies.

Provides optimization methods:
- Grid search
- Random search
- Bayesian optimization (future)
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.backtester.engine import BacktestConfig, BacktestResult
from src.backtester.parallel import ParallelBacktestRunner, ParallelBacktestTask
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    best_params: dict[str, Any]
    best_result: BacktestResult
    best_score: float
    all_results: list[tuple[dict[str, Any], BacktestResult, float]]
    optimization_metric: str

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(metric={self.optimization_metric}, "
            f"best_score={self.best_score:.2f})"
        )


class ParameterOptimizer:
    """
    Optimize strategy parameters using various search methods.
    """

    def __init__(
        self,
        strategy_factory: Callable[[dict[str, Any]], Strategy],
        tickers: list[str],
        interval: str,
        config: BacktestConfig,
        n_workers: int | None = None,
    ) -> None:
        """
        Initialize parameter optimizer.

        Args:
            strategy_factory: Function that creates a strategy from parameters
            tickers: List of tickers to backtest
            interval: Data interval
            config: Backtest configuration
            n_workers: Number of parallel workers
        """
        self.strategy_factory = strategy_factory
        self.tickers = tickers
        self.interval = interval
        self.config = config
        self.n_workers = n_workers

    def optimize(
        self,
        param_grid: dict[str, list[Any]],
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        method: str = "grid",
        n_iter: int = 100,
    ) -> OptimizationResult:
        """
        Optimize parameters using specified method.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            metric: Metric to optimize (e.g., 'sharpe_ratio', 'cagr', 'total_return')
            maximize: If True, maximize metric; if False, minimize
            method: Optimization method ('grid' or 'random')
            n_iter: Number of iterations for random search

        Returns:
            OptimizationResult with best parameters and results
        """
        if method == "grid":
            return self._grid_search(param_grid, metric, maximize)
        elif method == "random":
            return self._random_search(param_grid, metric, maximize, n_iter)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _grid_search(
        self,
        param_grid: dict[str, list[Any]],
        metric: str,
        maximize: bool,
    ) -> OptimizationResult:
        """Perform grid search over parameter space."""
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Grid search: {len(combinations)} parameter combinations")

        tasks = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            strategy = self.strategy_factory(params)
            task_name = f"{strategy.name}_{'_'.join(str(v) for v in combo)}"
            tasks.append(
                ParallelBacktestTask(
                    name=task_name,
                    strategy=strategy,
                    tickers=self.tickers,
                    interval=self.interval,
                    config=self.config,
                )
            )

        runner = ParallelBacktestRunner(n_workers=self.n_workers)
        results = runner.run(tasks)

        # Extract scores and find best
        all_results = []
        for task in tasks:
            task_name = task.name
            result = results.get(task_name)
            if result:
                score = self._extract_metric(result, metric)
                # Extract params from task name or store separately
                # For simplicity, parse from name (could be improved)
                params = self._parse_params_from_name(task_name, param_names)
                all_results.append((params, result, score))

        # Sort by score
        all_results.sort(key=lambda x: x[2], reverse=maximize)

        best_params, best_result, best_score = all_results[0] if all_results else (
            {},
            BacktestResult(),
            0.0,
        )

        return OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=metric,
        )

    def _random_search(
        self,
        param_grid: dict[str, list[Any]],
        metric: str,
        maximize: bool,
        n_iter: int,
    ) -> OptimizationResult:
        """Perform random search over parameter space."""
        import random

        param_names = list(param_grid.keys())
        tasks = []

        logger.info(f"Random search: {n_iter} iterations")

        for i in range(n_iter):
            # Randomly sample parameters
            params = {
                name: random.choice(values) for name, values in param_grid.items()
            }
            strategy = self.strategy_factory(params)
            task_name = f"{strategy.name}_iter{i}"
            tasks.append(
                ParallelBacktestTask(
                    name=task_name,
                    strategy=strategy,
                    tickers=self.tickers,
                    interval=self.interval,
                    config=self.config,
                    params=params,  # Store params for later retrieval
                )
            )

        runner = ParallelBacktestRunner(n_workers=self.n_workers)
        results = runner.run(tasks)

        # Extract scores (params are stored in task.params)
        all_results = []
        for task in tasks:
            result = results.get(task.name)
            if result and task.params:
                score = self._extract_metric(result, metric)
                all_results.append((task.params, result, score))

        # Sort by score
        all_results.sort(key=lambda x: x[2], reverse=maximize)

        best_params, best_result, best_score = all_results[0] if all_results else (
            {},
            BacktestResult(),
            0.0,
        )

        return OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=metric,
        )

    def _extract_metric(self, result: BacktestResult, metric: str) -> float:
        """
        Extract metric value from backtest result.

        Args:
            result: BacktestResult
            metric: Metric name

        Returns:
            Metric value
        """
        if metric == "sharpe_ratio":
            return result.sharpe_ratio if hasattr(result, "sharpe_ratio") else 0.0
        elif metric == "cagr":
            return result.cagr if hasattr(result, "cagr") else 0.0
        elif metric == "total_return":
            return result.total_return if hasattr(result, "total_return") else 0.0
        elif metric == "calmar_ratio":
            return result.calmar_ratio if hasattr(result, "calmar_ratio") else 0.0
        elif metric == "win_rate":
            return result.win_rate if hasattr(result, "win_rate") else 0.0
        elif metric == "profit_factor":
            return result.profit_factor if hasattr(result, "profit_factor") else 0.0
        else:
            logger.warning(f"Unknown metric: {metric}, using sharpe_ratio")
            return result.sharpe_ratio if hasattr(result, "sharpe_ratio") else 0.0

    def _parse_params_from_name(
        self, name: str, param_names: list[str]
    ) -> dict[str, Any]:
        """Parse parameters from task name (simplified)."""
        # This is a simplified parser - in practice, store params separately
        parts = name.split("_")
        params = {}
        # Try to extract numeric values
        for part in parts:
            try:
                value = float(part)
                if param_names:
                    # Assign to first unassigned param (simplified)
                    for pname in param_names:
                        if pname not in params:
                            params[pname] = int(value) if value.is_integer() else value
                            break
            except ValueError:
                pass
        return params


def optimize_strategy_parameters(
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    param_grid: dict[str, list[Any]],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    metric: str = "sharpe_ratio",
    maximize: bool = True,
    method: str = "grid",
    n_iter: int = 100,
    n_workers: int | None = None,
) -> OptimizationResult:
    """
    Optimize strategy parameters.

    Args:
        strategy_factory: Function that creates a strategy from parameters
        param_grid: Dictionary mapping parameter names to lists of values
        tickers: List of tickers to backtest
        interval: Data interval
        config: Backtest configuration
        metric: Metric to optimize
        maximize: If True, maximize metric
        method: Optimization method ('grid' or 'random')
        n_iter: Number of iterations for random search
        n_workers: Number of parallel workers

    Returns:
        OptimizationResult

    Example:
        Optimize VBO parameters::

            from src.strategies.volatility_breakout import create_vbo_strategy
            from src.backtester import BacktestConfig

            def create_strategy(params):
                return create_vbo_strategy(
                    name=f"VBO_{params['sma']}_{params['trend']}",
                    sma_period=params['sma'],
                    trend_sma_period=params['trend'],
                )

            param_grid = {
                'sma': [4, 5, 6],
                'trend': [8, 10, 12],
            }

            result = optimize_strategy_parameters(
                strategy_factory=create_strategy,
                param_grid=param_grid,
                tickers=["KRW-BTC"],
                interval="day",
                config=BacktestConfig(),
                metric="sharpe_ratio",
            )

            print(f"Best parameters: {result.best_params}")
            print(f"Best Sharpe ratio: {result.best_score}")
    """
    optimizer = ParameterOptimizer(
        strategy_factory=strategy_factory,
        tickers=tickers,
        interval=interval,
        config=config,
        n_workers=n_workers,
    )

    return optimizer.optimize(
        param_grid=param_grid,
        metric=metric,
        maximize=maximize,
        method=method,
        n_iter=n_iter,
    )
