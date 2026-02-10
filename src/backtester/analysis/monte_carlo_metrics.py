"""Monte Carlo simulation metrics calculation."""

import numpy as np

from src.utils.metrics_core import calculate_mdd, calculate_sharpe_ratio


def calculate_simulation_metrics(
    simulated_equities: np.ndarray,
    simulated_returns: np.ndarray,
    initial_capital: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate CAGR, MDD, and Sharpe ratio for each simulation.

    Args:
        simulated_equities: Array of equity curves (n_simulations x n_periods)
        simulated_returns: Array of returns (n_simulations x n_periods)
        initial_capital: Starting capital

    Returns:
        Tuple of (simulated_cagrs, simulated_mdds, simulated_sharpes)
    """
    n_simulations = simulated_returns.shape[0]
    n_periods = simulated_returns.shape[1]
    annualization_factor = 365.0

    simulated_cagrs = np.zeros(n_simulations)
    simulated_mdds = np.zeros(n_simulations)
    simulated_sharpes = np.zeros(n_simulations)

    for i in range(n_simulations):
        equity_curve = simulated_equities[i]
        returns = simulated_returns[i]

        simulated_cagrs[i] = _calculate_cagr(
            equity_curve, initial_capital, n_periods, annualization_factor
        )
        simulated_mdds[i] = calculate_mdd(equity_curve)
        simulated_sharpes[i] = calculate_sharpe_ratio(returns, annualization_factor)

    return simulated_cagrs, simulated_mdds, simulated_sharpes


def _calculate_cagr(
    equity_curve: np.ndarray,
    initial_capital: float,
    n_periods: int,
    annualization_factor: float,
) -> float:
    """Calculate CAGR for an equity curve."""
    if len(equity_curve) > 0 and equity_curve[-1] > 0 and n_periods > 0:
        result: float = (
            (equity_curve[-1] / initial_capital) ** (annualization_factor / n_periods) - 1
        ) * 100
        return result
    return -100.0


def calculate_statistics(
    simulated_cagrs: np.ndarray,
    simulated_mdds: np.ndarray,
    simulated_sharpes: np.ndarray,
) -> dict[str, float]:
    """
    Calculate summary statistics for simulation results.

    Returns:
        Dictionary with mean, std, and confidence intervals
    """

    return {
        "mean_cagr": float(np.mean(simulated_cagrs)),
        "std_cagr": float(np.std(simulated_cagrs)),
        "mean_mdd": float(np.mean(simulated_mdds)),
        "std_mdd": float(np.std(simulated_mdds)),
        "mean_sharpe": float(np.mean(simulated_sharpes)),
        "std_sharpe": float(np.std(simulated_sharpes)),
        "cagr_ci_lower": float(np.percentile(simulated_cagrs, 2.5)),
        "cagr_ci_upper": float(np.percentile(simulated_cagrs, 97.5)),
        "mdd_ci_lower": float(np.percentile(simulated_mdds, 2.5)),
        "mdd_ci_upper": float(np.percentile(simulated_mdds, 97.5)),
        "sharpe_ci_lower": float(np.percentile(simulated_sharpes, 2.5)),
        "sharpe_ci_upper": float(np.percentile(simulated_sharpes, 97.5)),
    }


def calculate_percentiles(data: np.ndarray) -> dict[float, float]:
    """Calculate standard percentiles for data."""
    percentiles = [5, 25, 50, 75, 95]
    return {float(p): float(np.percentile(data, p)) for p in percentiles}
