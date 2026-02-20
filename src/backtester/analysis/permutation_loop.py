"""
Permutation test loop helpers.

Provides helper functions for running permutation simulations.
"""

from collections.abc import Callable

import pandas as pd

from src.backtester.analysis.permutation_stats import shuffle_data
from src.backtester.wfa.walk_forward_backtest import simple_backtest
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["run_permutation_loop"]


def run_permutation_loop(
    data: pd.DataFrame,
    strategy_factory: Callable[[], Strategy],
    initial_capital: float,
    num_shuffles: int,
    shuffle_columns: list[str],
    verbose: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """
    Run permutation loop for shuffled data backtests.

    Args:
        data: OHLCV data
        strategy_factory: Function to create strategy instances
        initial_capital: Initial capital for backtest
        num_shuffles: Number of shuffles to run
        shuffle_columns: Columns to shuffle
        verbose: Whether to log progress

    Returns:
        Tuple of (shuffled_returns, shuffled_sharpes, shuffled_win_rates)
    """
    shuffled_returns: list[float] = []
    shuffled_sharpes: list[float] = []
    shuffled_win_rates: list[float] = []

    for i in range(num_shuffles):
        try:
            shuffled_data = shuffle_data(data, shuffle_columns)
            strategy_shuffled = strategy_factory()
            shuffled_result = simple_backtest(shuffled_data, strategy_shuffled, initial_capital)

            shuffled_returns.append(shuffled_result.total_return)
            shuffled_sharpes.append(getattr(shuffled_result, "sharpe_ratio", 0.0))

            if hasattr(shuffled_result, "win_rate"):
                shuffled_win_rates.append(shuffled_result.win_rate)

            if verbose and (i + 1) % max(1, num_shuffles // 10) == 0:
                logger.info(f"  Completed {i + 1}/{num_shuffles} permutations")

        except Exception as e:
            logger.debug(f"Permutation {i} failed: {e}")
            continue

    return shuffled_returns, shuffled_sharpes, shuffled_win_rates
