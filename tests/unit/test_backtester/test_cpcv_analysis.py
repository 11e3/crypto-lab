"""Tests for backtester.analysis.cpcv."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtester.analysis.cpcv import CombinatorialPurgedCV, CPCVResult


def _make_data(n: int = 100) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.997,
            "close": close,
            "volume": rng.uniform(1e6, 5e6, n),
        },
        index=idx,
    )
    return {"KRW-BTC": df}


def _dummy_backtest(data_dict: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, Any]:
    return {"cagr": 8.0, "mdd": -4.0, "win_rate": 0.6, "sortino_ratio": 1.3}


class TestCreateSplits:
    def test_returns_at_most_num_splits(self) -> None:
        cv = CombinatorialPurgedCV(num_splits=4, test_size=0.2)
        splits = cv.create_splits(100)
        assert len(splits) <= 4

    def test_returns_list_of_tuples(self) -> None:
        cv = CombinatorialPurgedCV(num_splits=3, test_size=0.25)
        splits = cv.create_splits(200)
        for item in splits:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_train_and_test_indices_disjoint(self) -> None:
        cv = CombinatorialPurgedCV(num_splits=3, test_size=0.25, purge_pct=0.02, embargo_pct=0.01)
        splits = cv.create_splits(200)
        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_test_indices_are_contiguous(self) -> None:
        cv = CombinatorialPurgedCV(num_splits=3, test_size=0.2)
        splits = cv.create_splits(100)
        for _, test_idx in splits:
            if len(test_idx) > 1:
                diffs = np.diff(test_idx)
                assert np.all(diffs == 1)

    def test_handles_small_dataset_gracefully(self) -> None:
        cv = CombinatorialPurgedCV(num_splits=5, test_size=0.5)
        splits = cv.create_splits(10)
        assert isinstance(splits, list)


class TestRun:
    def test_run_returns_cpcv_result(self) -> None:
        cv = CombinatorialPurgedCV(num_splits=3, test_size=0.2)
        data = _make_data()
        result = cv.run(data, _dummy_backtest)
        assert isinstance(result, CPCVResult)

    def test_fold_results_count_matches_splits(self) -> None:
        cv = CombinatorialPurgedCV(num_splits=3, test_size=0.2)
        data = _make_data()
        result = cv.run(data, _dummy_backtest)
        assert len(result.fold_results) == result.summary.num_folds

    def test_summary_consistency_in_valid_range(self) -> None:
        cv = CombinatorialPurgedCV(num_splits=3, test_size=0.2)
        data = _make_data()
        result = cv.run(data, _dummy_backtest)
        assert 0.0 <= result.summary.consistency <= 100.0

    def test_summary_avg_cagr_matches_dummy(self) -> None:
        cv = CombinatorialPurgedCV(num_splits=3, test_size=0.2)
        data = _make_data()
        result = cv.run(data, _dummy_backtest)
        # All folds return cagr=8.0, so avg should be 8.0
        assert result.summary.avg_cagr == 8.0
