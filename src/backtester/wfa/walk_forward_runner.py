"""
Walk-forward analysis execution utilities.

Contains period generation, optimization, and testing functions.
"""

from collections.abc import Callable
from datetime import date, timedelta
from itertools import product
from typing import Any

import pandas as pd

from src.backtester.engine import run_backtest
from src.backtester.models import BacktestConfig, BacktestResult
from src.backtester.optimization import OptimizationResult
from src.backtester.wfa.walk_forward_models import WalkForwardPeriod
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_periods(
    start_date: date,
    end_date: date,
    optimization_days: int,
    test_days: int,
    step_days: int,
    gap_days: int = 1,
) -> list[WalkForwardPeriod]:
    """
    Generate walk-forward periods.

    Args:
        start_date: Analysis start date
        end_date: Analysis end date
        optimization_days: Length of optimization period
        test_days: Length of test period
        step_days: Step size between periods
        gap_days: Gap between optimization end and test start to prevent data leakage

    Returns:
        List of WalkForwardPeriod objects
    """
    periods: list[WalkForwardPeriod] = []
    period_num = 1
    current_date = start_date

    while current_date < end_date:
        # Optimization period
        opt_start = current_date
        opt_end = opt_start + timedelta(days=optimization_days)

        # Test period (with gap after optimization to prevent data leakage)
        test_start = opt_end + timedelta(days=gap_days)
        test_end = test_start + timedelta(days=test_days)

        # Skip if test period extends beyond end_date
        if test_end > end_date:
            break

        periods.append(
            WalkForwardPeriod(
                period_num=period_num,
                optimization_start=opt_start,
                optimization_end=opt_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        current_date += timedelta(days=step_days)
        period_num += 1

    return periods


def optimize_period(
    period: WalkForwardPeriod,
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    param_grid: dict[str, list[Any]],
    metric: str,
    n_workers: int | None,
    ticker_data: dict[str, pd.DataFrame] | None = None,
) -> OptimizationResult | None:
    """
    Optimize parameters on optimization period.

    If ``ticker_data`` is provided (pre-loaded raw OHLCV DataFrames keyed by
    ticker), uses the lightweight ``simple_backtest`` path — no subprocess
    spawning, no per-combination disk I/O.  This is typically 10-100× faster
    than the full-engine multiprocessing path.

    If ``ticker_data`` is None, falls back to the original full-engine
    multiprocessing path for backward compatibility.

    Args:
        period: Walk-forward period
        strategy_factory: Function to create strategy from params
        tickers: List of tickers
        interval: Data interval
        config: Backtest configuration
        param_grid: Parameter grid
        metric: Optimization metric
        n_workers: Number of parallel workers (only used in fallback path)
        ticker_data: Pre-loaded raw OHLCV data keyed by ticker symbol

    Returns:
        OptimizationResult or None if failed
    """
    if ticker_data is not None:
        return _optimize_period_fast(
            period=period,
            strategy_factory=strategy_factory,
            config=config,
            param_grid=param_grid,
            metric=metric,
            ticker_data=ticker_data,
        )

    # --- fallback: original full-engine multiprocessing path ---
    from src.backtester.parallel import ParallelBacktestRunner, ParallelBacktestTask

    try:
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

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
                    start_date=period.optimization_start,
                    end_date=period.optimization_end,
                )
            )

        runner = ParallelBacktestRunner(n_workers=n_workers)
        results = runner.run(tasks)

        all_results: list[tuple[dict[str, Any], BacktestResult, float]] = []
        for task in tasks:
            result = results.get(task.name)
            if result and task.params:
                score = extract_metric(result, metric)
                all_results.append((task.params, result, score))

        all_results.sort(key=lambda x: x[2], reverse=True)

        if not all_results:
            return None

        best_params, best_result, best_score = all_results[0]

        return OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=metric,
        )
    except Exception as e:
        logger.error(f"Error optimizing period {period.period_num}: {e}", exc_info=True)
        return None


def _optimize_period_fast(
    period: WalkForwardPeriod,
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    config: BacktestConfig,
    param_grid: dict[str, list[Any]],
    metric: str,
    ticker_data: dict[str, pd.DataFrame],
) -> OptimizationResult | None:
    """
    Fast optimization using simple_backtest (no subprocess overhead).

    Data is sliced to the optimization window for each call.  Indicators are
    computed inside simple_backtest; the first few rows with NaN indicators
    produce no signals, so the warmup period is handled automatically.
    """
    from src.backtester.wfa.walk_forward_backtest import simple_backtest

    try:
        # Slice each ticker's data to the optimization window
        opt_start = pd.Timestamp(period.optimization_start)
        opt_end = pd.Timestamp(period.optimization_end)
        period_data: dict[str, pd.DataFrame] = {}
        for ticker, df in ticker_data.items():
            sliced = df[(df.index >= opt_start) & (df.index <= opt_end)]
            if len(sliced) > 0:
                period_data[ticker] = sliced

        if not period_data:
            logger.warning(f"No data for optimization period {period.period_num}")
            return None

        param_names = list(param_grid.keys())
        combinations = list(product(*param_grid.values()))

        all_results: list[tuple[dict[str, Any], BacktestResult, float]] = []

        for combo in combinations:
            params = dict(zip(param_names, combo, strict=True))
            strategy = strategy_factory(params)

            scores: list[float] = []
            last_result: BacktestResult | None = None

            for df in period_data.values():
                r = simple_backtest(df, strategy, config.initial_capital)
                scores.append(getattr(r, metric, 0.0))
                last_result = r

            if last_result is None or not scores:
                continue

            avg_score = sum(scores) / len(scores)
            all_results.append((params, last_result, avg_score))

        all_results.sort(key=lambda x: x[2], reverse=True)

        if not all_results:
            return None

        best_params, best_result, best_score = all_results[0]

        return OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=metric,
        )
    except Exception as e:
        logger.error(f"Error in fast optimization period {period.period_num}: {e}", exc_info=True)
        return None


def run_test_period(
    period: WalkForwardPeriod,
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    best_params: dict[str, Any],
) -> BacktestResult | None:
    """
    Test optimized parameters on test period.

    Args:
        period: Walk-forward period
        strategy_factory: Function to create strategy from params
        tickers: List of tickers
        interval: Data interval
        config: Backtest configuration
        best_params: Best parameters from optimization

    Returns:
        BacktestResult or None if failed
    """
    try:
        strategy = strategy_factory(best_params)
        result = run_backtest(
            strategy=strategy,
            tickers=tickers,
            interval=interval,
            config=config,
            start_date=period.test_start,
            end_date=period.test_end,
        )
        return result
    except Exception as e:
        logger.error(f"Error testing period {period.period_num}: {e}", exc_info=True)
        return None


def extract_metric(result: BacktestResult, metric: str) -> float:
    """
    Extract metric value from backtest result.

    Args:
        result: Backtest result
        metric: Metric name

    Returns:
        Metric value
    """
    metric_map = {
        "sharpe_ratio": lambda r: r.sharpe_ratio if hasattr(r, "sharpe_ratio") else 0.0,
        "sortino_ratio": lambda r: r.sortino_ratio if hasattr(r, "sortino_ratio") else 0.0,
        "cagr": lambda r: r.cagr if hasattr(r, "cagr") else 0.0,
        "total_return": lambda r: r.total_return if hasattr(r, "total_return") else 0.0,
        "calmar_ratio": lambda r: r.calmar_ratio if hasattr(r, "calmar_ratio") else 0.0,
        "win_rate": lambda r: r.win_rate if hasattr(r, "win_rate") else 0.0,
        "profit_factor": lambda r: r.profit_factor if hasattr(r, "profit_factor") else 0.0,
    }

    if metric in metric_map:
        return metric_map[metric](result)

    logger.warning(f"Unknown metric: {metric}, using sharpe_ratio")
    return result.sharpe_ratio if hasattr(result, "sharpe_ratio") else 0.0
