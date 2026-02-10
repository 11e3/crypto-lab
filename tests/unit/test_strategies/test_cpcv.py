"""Tests for Combinatorially Purged Cross-Validation (CPCV)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.backtester.analysis.cpcv import (
    CombinatorialPurgedCV,
    CPCVResult,
    CPCVSummary,
)


class TestCombinatorialPurgedCVInit:
    """Test CPCV initialization."""

    def test_default_parameters(self) -> None:
        cpcv = CombinatorialPurgedCV()
        assert cpcv.num_splits == 5
        assert cpcv.test_size == 0.2
        assert cpcv.purge_pct == 0.02
        assert cpcv.embargo_pct == 0.01

    def test_custom_parameters(self) -> None:
        cpcv = CombinatorialPurgedCV(
            num_splits=3, test_size=0.3, purge_pct=0.05, embargo_pct=0.02
        )
        assert cpcv.num_splits == 3
        assert cpcv.test_size == 0.3
        assert cpcv.purge_pct == 0.05
        assert cpcv.embargo_pct == 0.02


class TestCreateSplits:
    """Test CPCV split creation."""

    def test_creates_correct_number_of_splits(self) -> None:
        cpcv = CombinatorialPurgedCV(num_splits=5)
        splits = cpcv.create_splits(n_samples=500)
        assert len(splits) == 5

    def test_train_test_no_overlap(self) -> None:
        """Train and test sets must not overlap."""
        cpcv = CombinatorialPurgedCV(num_splits=3, purge_pct=0.02, embargo_pct=0.01)
        splits = cpcv.create_splits(n_samples=300)

        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, "Train and test indices must not overlap"

    def test_purge_gap_between_train_and_test(self) -> None:
        """Purge zone must separate train (before) from test."""
        cpcv = CombinatorialPurgedCV(
            num_splits=3, purge_pct=0.05, embargo_pct=0.0
        )
        splits = cpcv.create_splits(n_samples=200)

        purge_samples = int(200 * 0.05)  # 10

        for train_idx, test_idx in splits:
            test_start = test_idx.min()
            # Train indices before test should end before test_start - purge
            train_before = train_idx[train_idx < test_start]
            if len(train_before) > 0:
                assert train_before.max() < test_start - purge_samples + 1

    def test_embargo_gap_after_test(self) -> None:
        """Embargo zone must separate test from train (after)."""
        cpcv = CombinatorialPurgedCV(
            num_splits=3, purge_pct=0.0, embargo_pct=0.05
        )
        splits = cpcv.create_splits(n_samples=200)

        embargo_samples = int(200 * 0.05)  # 10

        for train_idx, test_idx in splits:
            test_end = test_idx.max()
            # Train indices after test should start after test_end + embargo
            train_after = train_idx[train_idx > test_end]
            if len(train_after) > 0:
                assert train_after.min() >= test_end + embargo_samples

    def test_test_size_fraction(self) -> None:
        """Test set size should approximate the specified fraction."""
        cpcv = CombinatorialPurgedCV(num_splits=3, test_size=0.2)
        splits = cpcv.create_splits(n_samples=500)

        expected_test_size = int(500 * 0.2)  # 100
        for _, test_idx in splits:
            assert len(test_idx) == expected_test_size

    def test_small_sample(self) -> None:
        """Should work with small sample sizes."""
        cpcv = CombinatorialPurgedCV(
            num_splits=2, test_size=0.3, purge_pct=0.01, embargo_pct=0.01
        )
        splits = cpcv.create_splits(n_samples=50)
        assert len(splits) >= 1

    def test_single_split(self) -> None:
        cpcv = CombinatorialPurgedCV(num_splits=1)
        splits = cpcv.create_splits(n_samples=100)
        assert len(splits) == 1


class TestCPCVRun:
    """Test CPCV run method."""

    @pytest.fixture()
    def sample_data(self) -> dict[str, pd.DataFrame]:
        """Create sample multi-symbol OHLCV data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        data: dict[str, pd.DataFrame] = {}
        for symbol in ["KRW-BTC", "KRW-ETH"]:
            prices = 50000 + np.cumsum(np.random.randn(n) * 500)
            data[symbol] = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * 1.02,
                    "low": prices * 0.98,
                    "close": prices,
                    "volume": np.random.rand(n) * 1000,
                },
                index=dates,
            )
        return data

    @staticmethod
    def _dummy_backtest(
        data: dict[str, pd.DataFrame], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Dummy backtest function for testing."""
        first_df = next(iter(data.values()))
        returns = first_df["close"].pct_change().dropna()
        cagr = float(returns.mean() * 365 * 100)
        mdd = float(returns.cumsum().cummin().min() * 100)
        return {
            "cagr": cagr,
            "mdd": mdd,
            "win_rate": 0.55,
            "sortino_ratio": 1.2,
        }

    def test_run_returns_cpcv_result(self, sample_data: dict[str, pd.DataFrame]) -> None:
        cpcv = CombinatorialPurgedCV(num_splits=3)
        result = cpcv.run(sample_data, self._dummy_backtest)

        assert isinstance(result, CPCVResult)
        assert isinstance(result.summary, CPCVSummary)
        assert len(result.fold_results) == 3

    def test_fold_results_have_required_keys(
        self, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        cpcv = CombinatorialPurgedCV(num_splits=2)
        result = cpcv.run(sample_data, self._dummy_backtest)

        for fold in result.fold_results:
            assert "fold" in fold
            assert "train_size" in fold
            assert "test_size" in fold
            assert "results" in fold
            assert "cagr" in fold["results"]

    def test_summary_statistics(self, sample_data: dict[str, pd.DataFrame]) -> None:
        cpcv = CombinatorialPurgedCV(num_splits=3)
        result = cpcv.run(sample_data, self._dummy_backtest)

        summary = result.summary
        assert summary.num_folds == 3
        assert summary.min_cagr <= summary.avg_cagr <= summary.max_cagr
        assert summary.std_cagr >= 0
        assert 0 <= summary.consistency <= 100


class TestCPCVSummary:
    """Test CPCVSummary dataclass."""

    def test_fields(self) -> None:
        summary = CPCVSummary(
            avg_cagr=10.0,
            std_cagr=2.0,
            min_cagr=5.0,
            max_cagr=15.0,
            avg_mdd=-10.0,
            worst_mdd=-20.0,
            avg_win_rate=0.55,
            avg_sortino=1.5,
            consistency=80.0,
            num_folds=5,
        )
        assert summary.avg_cagr == 10.0
        assert summary.consistency == 80.0
        assert summary.num_folds == 5
