"""
Walk-forward statistics calculation utilities.
"""

import numpy as np

from src.backtester.wfa.walk_forward_models import WalkForwardPeriod, WalkForwardResult

__all__ = ["calculate_walk_forward_statistics"]


def calculate_walk_forward_statistics(
    periods: list[WalkForwardPeriod],
) -> WalkForwardResult:
    """
    Calculate aggregate statistics from walk-forward periods.

    Args:
        periods: List of walk-forward periods with results

    Returns:
        WalkForwardResult with aggregated statistics
    """
    test_cagrs: list[float] = []
    test_sharpes: list[float] = []
    test_sortinos: list[float] = []
    test_mdds: list[float] = []
    opt_cagrs: list[float] = []

    positive_count = 0
    total_count = 0

    for period in periods:
        if period.test_result:
            test_cagrs.append(period.test_result.cagr)
            test_sharpes.append(period.test_result.sharpe_ratio)
            test_sortinos.append(period.test_result.sortino_ratio)
            test_mdds.append(period.test_result.mdd)
            total_count += 1
            if period.test_result.cagr > 0:
                positive_count += 1

        if period.optimization_result and period.optimization_result.best_result:
            opt_cagrs.append(period.optimization_result.best_result.cagr)

    avg_test_cagr = float(np.mean(test_cagrs)) if test_cagrs else 0.0
    avg_test_sharpe = float(np.mean(test_sharpes)) if test_sharpes else 0.0
    avg_test_sortino = float(np.mean(test_sortinos)) if test_sortinos else 0.0
    avg_test_mdd = float(np.mean(test_mdds)) if test_mdds else 0.0
    avg_optimization_cagr = float(np.mean(opt_cagrs)) if opt_cagrs else 0.0
    consistency_rate = (positive_count / total_count * 100) if total_count > 0 else 0.0

    return WalkForwardResult(
        periods=periods,
        avg_test_cagr=avg_test_cagr,
        avg_test_sharpe=avg_test_sharpe,
        avg_test_sortino=avg_test_sortino,
        avg_test_mdd=avg_test_mdd,
        avg_optimization_cagr=avg_optimization_cagr,
        positive_periods=positive_count,
        total_periods=total_count,
        consistency_rate=consistency_rate,
    )
