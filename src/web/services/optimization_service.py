"""Optimization service — pure computation logic for parameter optimization.

Extracted from web/pages/optimization.py to decouple from Streamlit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from src.backtester import BacktestConfig, optimize_strategy_parameters
from src.utils.logger import get_logger
from src.web.services.vbo_backtest_runner import (
    VboBacktestResult,
    run_vbo_backtest_service,
)

logger = get_logger(__name__)


@dataclass
class VboOptimizationResult:
    """Result of VBO strategy optimization."""

    best_params: dict[str, Any]
    best_score: float
    all_params: list[dict[str, Any]] = field(default_factory=list)
    all_scores: list[float] = field(default_factory=list)


def get_default_param_range(spec: Any) -> str:
    """Generate default parameter range string from a ParameterSpec.

    Args:
        spec: ParameterSpec object with type, default, min_value, max_value, step

    Returns:
        Comma-separated default values string
    """
    default = spec.default
    param_type = spec.type

    if param_type == "int":
        min_int = int(spec.min_value or 1)
        max_int = int(spec.max_value or 100)
        step_int = int(spec.step or 1)

        int_values: list[int] = []
        for v in range(min_int, max_int + 1, step_int):
            if abs(v - default) <= step_int * 3:
                int_values.append(v)

        if not int_values:
            int_values = [int(default)]

        return ",".join(str(v) for v in int_values)

    elif param_type == "float":
        min_float = float(spec.min_value or 0.0)
        max_float = float(spec.max_value or 1.0)
        step_float = float(spec.step or 0.1)

        float_values: list[float] = []
        current_float: float = min_float
        while current_float <= max_float and len(float_values) < 5:
            float_values.append(round(current_float, 4))
            current_float += step_float

        return ",".join(str(fv) for fv in float_values)

    elif param_type == "bool":
        return "True,False"

    return str(default)


def parse_dynamic_param_grid(
    param_ranges: dict[str, str],
    param_specs: dict[str, Any],
) -> dict[str, list[Any]]:
    """Parse dynamic parameter ranges from user input strings.

    Args:
        param_ranges: Parameter name to range string mapping
        param_specs: Parameter specifications from strategy

    Returns:
        Parameter grid dictionary

    Raises:
        ValueError: If parsing error occurs
    """
    param_grid: dict[str, list[Any]] = {}

    for param_name, range_str in param_ranges.items():
        if not range_str.strip():
            raise ValueError(f"Please enter values for {param_name}")

        spec = param_specs.get(param_name)
        param_type = spec.type if spec else "int"

        values: list[Any] = []
        for val_str in range_str.split(","):
            val_str = val_str.strip()
            if not val_str:
                continue

            try:
                if param_type == "int":
                    values.append(int(val_str))
                elif param_type == "float":
                    values.append(float(val_str))
                elif param_type == "bool":
                    values.append(val_str.lower() in ("true", "1", "yes"))
                else:
                    values.append(val_str)
            except ValueError as e:
                raise ValueError(f"Invalid value '{val_str}' for {param_name}") from e

        if not values:
            raise ValueError(f"No valid values for {param_name}")

        param_grid[param_name] = values

    return param_grid


def extract_vbo_metric(result: VboBacktestResult, metric: str) -> float:
    """Extract metric value from VBO backtest result.

    Args:
        result: VboBacktestResult object
        metric: Metric name

    Returns:
        Metric value
    """
    if metric == "calmar_ratio":
        mdd = abs(result.mdd) if result.mdd != 0 else 1.0
        return result.cagr / mdd

    metric_map = {
        "sharpe_ratio": "sharpe_ratio",
        "cagr": "cagr",
        "total_return": "total_return",
        "win_rate": "win_rate",
        "profit_factor": "profit_factor",
        "sortino_ratio": "sortino_ratio",
    }

    attr = metric_map.get(metric, "sharpe_ratio")
    return getattr(result, attr, 0.0)


def execute_native_optimization(
    strategy_class: type,
    param_grid: dict[str, list[Any]],
    tickers: list[str],
    interval: str,
    metric: str,
    method: str,
    n_iter: int,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    workers: int,
) -> Any:
    """Run native strategy optimization (no Streamlit dependency).

    Args:
        strategy_class: Strategy class to instantiate
        param_grid: Parameter grid
        tickers: List of tickers
        interval: Data interval
        metric: Optimization metric
        method: Search method (grid/random)
        n_iter: Number of iterations for random search
        initial_capital: Initial capital
        fee_rate: Fee rate
        max_slots: Maximum position slots
        workers: Number of parallel workers

    Returns:
        OptimizationResult from backtester

    Raises:
        Exception: If optimization fails
    """

    def create_strategy(params: dict[str, Any]) -> Any:
        return strategy_class(**params)

    config = BacktestConfig(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=fee_rate,
        max_slots=max_slots,
        use_cache=True,
    )

    return optimize_strategy_parameters(
        strategy_factory=create_strategy,
        param_grid=param_grid,
        tickers=tickers,
        interval=interval,
        config=config,
        metric=metric,
        maximize=True,
        method=method,
        n_iter=n_iter,
        n_workers=workers,
    )


def execute_vbo_optimization(
    strategy_name: str,
    param_grid: dict[str, list[Any]],
    symbols: list[str],
    metric: str,
    method: str,
    n_iter: int,
    initial_capital: int,
    fee_rate: float,
    on_progress: Any | None = None,
) -> VboOptimizationResult:
    """Run VBO strategy optimization (no Streamlit dependency).

    Args:
        strategy_name: VBO strategy name
        param_grid: Parameter grid
        symbols: List of symbols (without KRW- prefix)
        metric: Optimization metric
        method: Search method (grid/random)
        n_iter: Number of iterations for random search
        initial_capital: Initial capital in KRW
        fee_rate: Fee rate
        on_progress: Optional callback(current, total) for progress updates

    Returns:
        VboOptimizationResult

    Raises:
        RuntimeError: If all backtests fail
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    if method == "grid":
        combinations = list(product(*param_values))
    else:
        all_combinations = list(product(*param_values))
        n_iter = min(n_iter, len(all_combinations))
        combinations = random.sample(all_combinations, n_iter)

    total = len(combinations)

    all_results: list[tuple[dict[str, Any], VboBacktestResult | None, float]] = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo, strict=False))

        try:
            result = run_vbo_backtest_service(
                symbols=tuple(symbols),
                interval="day",
                initial_cash=initial_capital,
                fee=fee_rate,
                slippage=fee_rate,
                multiplier=params.get("multiplier", 2),
                lookback=params.get("lookback", 5),
            )

            if result:
                score = extract_vbo_metric(result, metric)
                all_results.append((params, result, score))
            else:
                all_results.append((params, None, float("-inf")))

        except Exception as e:
            logger.warning(f"VBO backtest failed for {params}: {e}")
            all_results.append((params, None, float("-inf")))

        if on_progress:
            on_progress(i + 1, total)

    all_results.sort(key=lambda x: x[2], reverse=True)

    if not all_results or all_results[0][1] is None:
        raise RuntimeError("All backtests failed")

    best_params, _, best_score = all_results[0]

    return VboOptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_params=[r[0] for r in all_results if r[1] is not None],
        all_scores=[r[2] for r in all_results if r[1] is not None],
    )
