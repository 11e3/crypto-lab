"""
Statistical functions for Robustness Analysis.

Provides neighbor parameter analysis and sensitivity calculation.
"""

from typing import Any

import numpy as np

from src.backtester.analysis.robustness_models import RobustnessResult


def find_neighbors(
    optimal_params: dict[str, Any],
    results: list[RobustnessResult],
    tolerance: float = 0.20,
) -> list[RobustnessResult]:
    """
    Find parameter combinations within ±tolerance% of the optimal values.

    Example: optimal_param=4, tolerance=0.20 → accepts range 3.2–4.8

    Args:
        optimal_params: Optimal parameter values
        results: All test results
        tolerance: Allowed deviation (0.20 = ±20%)

    Returns:
        Results whose parameters are all within tolerance of optimal
    """
    neighbors: list[RobustnessResult] = []

    for result in results:
        is_neighbor = True

        for param_name, optimal_value in optimal_params.items():
            if param_name not in result.params:
                continue

            actual_value = result.params[param_name]

            # Only compare numeric parameters
            if not isinstance(actual_value, int | float):
                continue

            if optimal_value == 0:
                is_neighbor = actual_value == 0
            else:
                change_pct = abs(actual_value - optimal_value) / abs(optimal_value)

                if change_pct > tolerance:
                    is_neighbor = False
                    break

        if is_neighbor:
            neighbors.append(result)

    return neighbors


def calculate_sensitivity(results: list[RobustnessResult]) -> dict[str, float]:
    """
    Compute per-parameter sensitivity scores in the range [0.0, 1.0].

    Score near 1.0 means performance is highly sensitive to that parameter.
    Score near 0.0 means performance is stable regardless of the parameter value.

    Args:
        results: All test results

    Returns:
        Sensitivity score per parameter (0.0–1.0)
    """
    if not results:
        return {}

    sensitivity: dict[str, float] = {}

    param_names = list(results[0].params.keys())

    for param_name in param_names:
        param_values: list[float] = []
        returns: list[float] = []

        for result in results:
            if param_name in result.params:
                value = result.params[param_name]

                if isinstance(value, int | float):
                    param_values.append(float(value))
                    returns.append(result.total_return)

        if len(set(param_values)) < 2:
            # Not enough variation to compute correlation
            sensitivity[param_name] = 0.0
            continue

        # Correlation between parameter value and return measures sensitivity
        correlation = abs(np.corrcoef(param_values, returns)[0, 1])

        if np.isnan(correlation):
            sensitivity[param_name] = 0.0
        else:
            sensitivity[param_name] = float(correlation)

    return sensitivity


def calculate_neighbor_success_rate(
    optimal_params: dict[str, Any],
    results: list[RobustnessResult],
    tolerance: float = 0.20,
) -> float:
    """
    Fraction of neighbor parameters that achieve at least 80% of the optimal return.

    Args:
        optimal_params: Optimal parameter values
        results: All test results
        tolerance: Neighbor range (0.20 = ±20%)

    Returns:
        Success rate (0.0–1.0)
    """
    neighbor_results = find_neighbors(optimal_params, results, tolerance)

    if not neighbor_results:
        return 0.0

    optimal_returns = [r.total_return for r in results if r.params == optimal_params]

    if not optimal_returns:
        return 0.0

    optimal_return = max(optimal_returns)
    threshold = optimal_return * 0.80

    successful_neighbors = sum(1 for r in neighbor_results if r.total_return >= threshold)

    return successful_neighbors / len(neighbor_results)
