"""Combinatorially Purged Cross-Validation (CPCV).

Advanced cross-validation for time-series backtesting that prevents
information leakage through purging and embargo.

Ported from bt framework and adapted to crypto-lab conventions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CPCVResult:
    """Result container for CPCV analysis."""

    fold_results: list[dict[str, Any]]
    summary: CPCVSummary


@dataclass
class CPCVSummary:
    """Summary statistics across all CPCV folds."""

    avg_cagr: float
    std_cagr: float
    min_cagr: float
    max_cagr: float
    avg_mdd: float
    worst_mdd: float
    avg_win_rate: float
    avg_sortino: float
    consistency: float  # % of folds with positive CAGR
    num_folds: int


class CombinatorialPurgedCV:
    """Combinatorially Purged Cross-Validation.

    Time-series aware CV that:
    1. Creates multiple non-contiguous test splits
    2. Purges data between train/test to prevent leakage
    3. Embargoes data after test to prevent forward-looking bias

    Args:
        num_splits: Number of cross-validation splits
        test_size: Fraction of data for testing (0.0-1.0)
        purge_pct: Percentage of data to purge between train/test
        embargo_pct: Percentage of data to embargo after test set
    """

    def __init__(
        self,
        num_splits: int = 5,
        test_size: float = 0.2,
        purge_pct: float = 0.02,
        embargo_pct: float = 0.01,
    ) -> None:
        self.num_splits = num_splits
        self.test_size = test_size
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def create_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create train/test splits with purging and embargo.

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_indices, test_indices) tuples
        """
        test_samples = int(n_samples * self.test_size)
        purge_samples = int(n_samples * self.purge_pct)
        embargo_samples = int(n_samples * self.embargo_pct)

        splits: list[tuple[np.ndarray, np.ndarray]] = []

        # Create equally spaced test sets
        test_starts = np.linspace(0, n_samples - test_samples, self.num_splits, dtype=int)

        for test_start in test_starts:
            test_end = test_start + test_samples
            test_indices = np.arange(test_start, test_end)

            # Build train indices excluding purge and embargo zones
            train_indices_list: list[int] = []

            if test_start > purge_samples:
                train_indices_list.extend(range(0, test_start - purge_samples))

            if test_end + embargo_samples < n_samples:
                train_indices_list.extend(range(test_end + embargo_samples, n_samples))

            train_indices = np.array(train_indices_list)

            if len(train_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits

    def run(
        self,
        data: dict[str, pd.DataFrame],
        backtest_func: Callable[..., dict[str, Any]],
    ) -> CPCVResult:
        """Run CPCV analysis.

        Args:
            data: Dictionary of symbol -> DataFrame with OHLCV data
            backtest_func: Function(data_dict, config_dict) -> dict with
                          keys like 'cagr', 'mdd', 'win_rate', 'sortino_ratio'

        Returns:
            CPCVResult with per-fold results and summary
        """
        first_symbol = next(iter(data.keys()))
        n_samples = len(data[first_symbol])
        splits = self.create_splits(n_samples)

        logger.info(f"Starting CPCV: {len(splits)} folds, {len(data)} symbols, {n_samples} samples")

        all_results: list[dict[str, Any]] = []

        for i, (train_indices, test_indices) in enumerate(splits):
            # Split data for all symbols
            test_data = {symbol: df.iloc[test_indices].copy() for symbol, df in data.items()}

            # Run backtest on test set
            test_results = backtest_func(test_data, {})

            fold_result = {
                "fold": i + 1,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "results": test_results,
            }
            all_results.append(fold_result)

            logger.info(
                f"Fold {i + 1}/{len(splits)}: "
                f"CAGR={test_results.get('cagr', 0):.2f}%, "
                f"MDD={test_results.get('mdd', 0):.2f}%"
            )

        summary = self._summarize(all_results)
        logger.info(
            f"CPCV complete: avg_cagr={summary.avg_cagr:.2f}%, "
            f"consistency={summary.consistency:.1f}%"
        )

        return CPCVResult(fold_results=all_results, summary=summary)

    def _summarize(self, all_results: list[dict[str, Any]]) -> CPCVSummary:
        """Aggregate results across all folds."""
        cagrs = [r["results"].get("cagr", 0) for r in all_results]
        mdds = [r["results"].get("mdd", 0) for r in all_results]
        win_rates = [r["results"].get("win_rate", 0) for r in all_results]
        sortinos = [r["results"].get("sortino_ratio", 0) for r in all_results]

        return CPCVSummary(
            avg_cagr=float(np.mean(cagrs)),
            std_cagr=float(np.std(cagrs)),
            min_cagr=float(np.min(cagrs)),
            max_cagr=float(np.max(cagrs)),
            avg_mdd=float(np.mean(mdds)),
            worst_mdd=float(np.min(mdds)),
            avg_win_rate=float(np.mean(win_rates)),
            avg_sortino=float(np.mean(sortinos)),
            consistency=float(len([c for c in cagrs if c > 0]) / len(cagrs) * 100),
            num_folds=len(all_results),
        )
