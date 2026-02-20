"""Statistical utilities for Permutation Test."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.backtester.analysis.permutation_test import PermutationTestResult

logger = get_logger(__name__)


def shuffle_data(data: pd.DataFrame, columns_to_shuffle: list[str]) -> pd.DataFrame:
    """
    Block-bootstrap shuffle preserving OHLC consistency.

    Resamples returns in blocks of 5 bars so autocorrelation structure is
    partially preserved, then reconstructs prices from the resampled returns.
    The date index is kept intact; only the price sequence changes.

    Args:
        data: OHLCV DataFrame
        columns_to_shuffle: Columns to shuffle (e.g. ['close', 'volume'])

    Returns:
        Shuffled DataFrame with the same index
    """
    shuffled = data.copy()

    # Compute daily returns from close prices
    returns = shuffled["close"].pct_change().fillna(0).values

    # Block-bootstrap: resample returns in blocks of 5 to preserve short-term structure
    block_size = 5
    n = len(returns)
    resampled_returns: list[float] = []
    i = 0
    while i < n:
        start = np.random.randint(0, max(1, n - block_size))
        block = returns[start : start + block_size]
        resampled_returns.extend(block.tolist())
        i += block_size
    resampled_array = np.array(resampled_returns[:n])

    # Reconstruct close prices from resampled returns
    base_price = float(shuffled["close"].iloc[0])
    new_close: list[float] = [base_price]
    for r in resampled_array[1:]:
        new_close.append(new_close[-1] * (1 + r))

    shuffled["close"] = new_close

    # Rebuild OHLC to maintain internal consistency
    shuffled["open"] = shuffled["close"].shift(1).fillna(shuffled["close"].iloc[0])
    shuffled["high"] = shuffled[["open", "close"]].max(axis=1) * 1.002
    shuffled["low"] = shuffled[["open", "close"]].min(axis=1) * 0.998

    if "volume" in columns_to_shuffle and "volume" in shuffled.columns:
        volume_values = shuffled["volume"].values
        volume_array = np.array(volume_values, dtype=np.float64).copy()
        np.random.shuffle(volume_array)
        shuffled["volume"] = volume_array

    return shuffled


def compute_statistics(
    original_return: float,
    original_sharpe: float,
    original_win_rate: float,
    shuffled_returns: list[float],
    shuffled_sharpes: list[float],
    shuffled_win_rates: list[float],
    result_class: type[PermutationTestResult],
) -> PermutationTestResult:
    """
    Compute Z-score and p-value comparing original vs shuffled performance.

    Args:
        original_return: Return on original data
        original_sharpe: Sharpe ratio on original data
        original_win_rate: Win rate on original data
        shuffled_returns: Returns from each shuffled run
        shuffled_sharpes: Sharpe ratios from each shuffled run
        shuffled_win_rates: Win rates from each shuffled run
        result_class: PermutationTestResult class to instantiate

    Returns:
        PermutationTestResult with statistical significance metrics
    """
    result = result_class(
        original_return=original_return,
        original_sharpe=original_sharpe,
        original_win_rate=original_win_rate,
        shuffled_returns=shuffled_returns,
        shuffled_sharpes=shuffled_sharpes,
        shuffled_win_rates=shuffled_win_rates,
    )

    if not shuffled_returns:
        logger.error("No valid shuffled results")
        return result

    mean_shuffled = float(np.mean(shuffled_returns))
    std_shuffled = float(np.std(shuffled_returns))

    result.mean_shuffled_return = mean_shuffled
    result.std_shuffled_return = std_shuffled

    # Z-score = (X - μ) / σ
    if std_shuffled > 0:
        result.z_score = (original_return - mean_shuffled) / std_shuffled
    else:
        result.z_score = 0.0

    # Two-tailed p-value: probability of achieving original performance by chance
    result.p_value = float(2 * (1 - stats.norm.cdf(abs(result.z_score))))

    if result.p_value < 0.01:
        result.confidence_level = "1%"
        result.is_statistically_significant = True
    elif result.p_value < 0.05:
        result.confidence_level = "5%"
        result.is_statistically_significant = True
    else:
        result.confidence_level = "not significant"
        result.is_statistically_significant = False

    result.interpretation = interpret_results(result)

    return result


def interpret_results(result: PermutationTestResult) -> str:
    """Build a human-readable interpretation string for the permutation test result."""
    if result.z_score < 0:
        return (
            f"Original return ({result.original_return:.2%}) is below the shuffled mean "
            f"({result.mean_shuffled_return:.2%}). Strategy does not appear to capture real signal."
        )
    elif result.z_score < 1.0:
        return (
            f"Z-score={result.z_score:.2f} < 1.0: not statistically significant "
            f"(p-value={result.p_value:.4f}). Performance is likely due to chance — possible overfitting."
        )
    elif result.z_score < 2.0:
        return (
            f"Z-score={result.z_score:.2f}: weakly significant (p-value={result.p_value:.4f}). "
            f"Some signal present, but overfitting risk remains."
        )
    elif result.z_score < 3.0:
        return (
            f"Z-score={result.z_score:.2f} ({result.confidence_level} significance level): "
            f"statistically significant — strategy likely captures real signal."
        )
    else:
        return (
            f"Z-score={result.z_score:.2f} ({result.confidence_level} significance level): "
            f"very strong statistical significance — high signal quality."
        )
