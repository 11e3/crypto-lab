"""
Parallel backtesting support.

Enables concurrent execution of multiple backtests for:
- Multiple strategies
- Parameter optimization
- Strategy comparison
"""

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.backtester.engine import BacktestConfig, BacktestResult, run_backtest
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParallelBacktestTask:
    """Single backtest task for parallel execution."""

    name: str
    strategy: Strategy
    tickers: list[str]
    interval: str
    config: BacktestConfig
    params: dict[str, Any] | None = None  # Store parameters for optimization
    start_date: date | None = None  # Start date for date filtering
    end_date: date | None = None  # End date for date filtering

    def __repr__(self) -> str:
        return f"ParallelBacktestTask(name={self.name}, tickers={self.tickers})"


def _run_single_backtest(task: ParallelBacktestTask) -> tuple[str, BacktestResult]:
    """
    Run a single backtest (worker function for multiprocessing).

    Args:
        task: Backtest task to execute

    Returns:
        Tuple of (task_name, BacktestResult)
    """
    try:
        logger.info(f"Starting backtest: {task.name}")
        result = run_backtest(
            strategy=task.strategy,
            tickers=task.tickers,
            interval=task.interval,
            config=task.config,
            start_date=task.start_date,
            end_date=task.end_date,
        )
        logger.info(f"Completed backtest: {task.name}")
        return (task.name, result)
    except Exception as e:
        logger.error(f"Error in backtest {task.name}: {e}", exc_info=True)
        # Return empty result on error
        empty_result = BacktestResult()
        empty_result.strategy_name = task.name
        return (task.name, empty_result)


class ParallelBacktestRunner:
    """
    Runs multiple backtests in parallel.

    Supports concurrent execution of multiple strategies or parameter combinations.
    """

    def __init__(self, n_workers: int | None = None) -> None:
        """
        Initialize parallel backtest runner.

        Args:
            n_workers: Number of parallel workers. Defaults to CPU count - 1.
        """
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        self.n_workers = n_workers
        logger.info(f"Initialized ParallelBacktestRunner with {n_workers} workers")

    def run(
        self,
        tasks: list[ParallelBacktestTask],
        progress_callback: Callable[[str, BacktestResult], None] | None = None,
    ) -> dict[str, BacktestResult]:
        """
        Run multiple backtests in parallel.

        Args:
            tasks: List of backtest tasks to execute
            progress_callback: Optional callback function called after each task completes
                              Signature: (task_name: str, result: BacktestResult) -> None

        Returns:
            Dictionary mapping task names to BacktestResult objects
        """
        if not tasks:
            logger.warning("No tasks provided to parallel backtest runner")
            return {}

        logger.info(f"Running {len(tasks)} backtests with {self.n_workers} workers")

        # Use multiprocessing Pool for parallel execution
        with mp.Pool(processes=self.n_workers) as pool:
            results = pool.map(_run_single_backtest, tasks)

        # Convert to dictionary
        results_dict: dict[str, BacktestResult] = {}
        for task_name, result in results:
            results_dict[task_name] = result
            if progress_callback:
                progress_callback(task_name, result)

        logger.info(f"Completed {len(results_dict)} backtests")
        return results_dict

    def run_sequential(
        self,
        tasks: list[ParallelBacktestTask],
        progress_callback: Callable[[str, BacktestResult], None] | None = None,
    ) -> dict[str, BacktestResult]:
        """
        Run backtests sequentially (useful for debugging).

        Args:
            tasks: List of backtest tasks to execute
            progress_callback: Optional callback function called after each task completes

        Returns:
            Dictionary mapping task names to BacktestResult objects
        """
        if not tasks:
            logger.warning("No tasks provided to sequential backtest runner")
            return {}

        logger.info(f"Running {len(tasks)} backtests sequentially")

        results_dict: dict[str, BacktestResult] = {}
        for task in tasks:
            task_name, result = _run_single_backtest(task)
            results_dict[task_name] = result
            if progress_callback:
                progress_callback(task_name, result)

        logger.info(f"Completed {len(results_dict)} backtests")
        return results_dict


def compare_strategies(
    strategies: list[Strategy],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    n_workers: int | None = None,
) -> dict[str, BacktestResult]:
    """
    Compare multiple strategies using parallel backtesting.

    Args:
        strategies: List of strategies to compare
        tickers: List of tickers to backtest
        interval: Data interval
        config: Backtest configuration
        n_workers: Number of parallel workers

    Returns:
        Dictionary mapping strategy names to BacktestResult objects

    Example:
        Compare VBO variants::

            from src.strategies.volatility_breakout import VanillaVBO, MinimalVBO
            from src.backtester import BacktestConfig

            strategies = [
                VanillaVBO(name="Vanilla"),
                MinimalVBO(name="Minimal"),
            ]
            config = BacktestConfig(initial_capital=1_000_000)

            results = compare_strategies(
                strategies=strategies,
                tickers=["KRW-BTC", "KRW-ETH"],
                interval="day",
                config=config,
            )
    """
    tasks = [
        ParallelBacktestTask(
            name=strategy.name,
            strategy=strategy,
            tickers=tickers,
            interval=interval,
            config=config,
        )
        for strategy in strategies
    ]

    runner = ParallelBacktestRunner(n_workers=n_workers)
    return runner.run(tasks)


def optimize_parameters(
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    param_grid: dict[str, list[Any]],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    n_workers: int | None = None,
) -> dict[str, BacktestResult]:
    """
    Optimize strategy parameters using grid search with parallel backtesting.

    Args:
        strategy_factory: Function that creates a strategy from parameters
                         Signature: (params: dict) -> Strategy
        param_grid: Dictionary mapping parameter names to lists of values to test
        tickers: List of tickers to backtest
        interval: Data interval
        config: Backtest configuration
        n_workers: Number of parallel workers

    Returns:
        Dictionary mapping parameter combination names to BacktestResult objects

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

            results = optimize_parameters(
                strategy_factory=create_strategy,
                param_grid=param_grid,
                tickers=["KRW-BTC"],
                interval="day",
                config=BacktestConfig(),
            )
    """
    from itertools import product

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    tasks: list[ParallelBacktestTask] = []
    for combo in combinations:
        params = dict(zip(param_names, combo))
        strategy = strategy_factory(params)
        task_name = f"{strategy.name}_{'_'.join(str(v) for v in combo)}"
        tasks.append(
            ParallelBacktestTask(
                name=task_name,
                strategy=strategy,
                tickers=tickers,
                interval=interval,
                config=config,
            )
        )

    logger.info(f"Generated {len(tasks)} parameter combinations for optimization")

    runner = ParallelBacktestRunner(n_workers=n_workers)
    return runner.run(tasks)
