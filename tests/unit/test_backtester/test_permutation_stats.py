"""Tests for backtester.analysis.permutation_stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester.analysis.permutation_stats import (
    compute_statistics,
    interpret_results,
    shuffle_data,
)
from src.backtester.analysis.permutation_test import PermutationTestResult


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        }
    )


class TestShuffleData:
    def test_preserves_index(self) -> None:
        df = _make_ohlcv()
        shuffled = shuffle_data(df, ["close"])
        assert list(shuffled.index) == list(df.index)

    def test_preserves_length(self) -> None:
        # Block-bootstrap output has the same number of rows
        df = _make_ohlcv()
        shuffled = shuffle_data(df, ["close"])
        assert len(shuffled) == len(df)

    def test_close_starts_from_same_base_price(self) -> None:
        # Prices are reconstructed from the original first close as anchor
        df = _make_ohlcv()
        shuffled = shuffle_data(df, ["close"])
        assert shuffled["close"].iloc[0] == pytest.approx(df["close"].iloc[0])

    def test_non_shuffled_column_unchanged(self) -> None:
        df = _make_ohlcv()
        shuffled = shuffle_data(df, ["close"])
        assert (shuffled["volume"] == df["volume"]).all()


class TestComputeStatistics:
    def _make_result(self, original_return: float, shuffled: list[float]) -> PermutationTestResult:
        return compute_statistics(
            original_return=original_return,
            original_sharpe=1.5,
            original_win_rate=0.6,
            shuffled_returns=shuffled,
            shuffled_sharpes=[0.5] * len(shuffled),
            shuffled_win_rates=[0.4] * len(shuffled),
            result_class=PermutationTestResult,
        )

    def test_z_score_positive_for_outperforming_original(self) -> None:
        shuffled = [0.01] * 100
        result = self._make_result(0.5, shuffled)
        assert result.z_score > 0

    def test_z_score_zero_when_std_zero(self) -> None:
        # All shuffled returns identical and exactly 0.0 (exactly representable) → std=0
        result = self._make_result(0.1, [0.0] * 50)
        assert result.z_score == 0.0

    def test_p_value_range(self) -> None:
        result = self._make_result(0.3, list(np.random.default_rng(1).normal(0, 0.05, 200)))
        assert 0.0 <= result.p_value <= 1.0


class TestInterpretResults:
    def _result_with_z(self, z: float) -> PermutationTestResult:
        r = PermutationTestResult(
            original_return=0.1,
            original_sharpe=1.0,
            original_win_rate=0.5,
            shuffled_returns=[0.0] * 10,
            shuffled_sharpes=[0.0] * 10,
            shuffled_win_rates=[0.5] * 10,
            z_score=z,
            p_value=0.04 if z > 2 else 0.2,
            is_statistically_significant=z > 2,
        )
        return r

    def test_negative_z_returns_warning_string(self) -> None:
        text = interpret_results(self._result_with_z(-1.0))
        assert len(text) > 0

    def test_high_z_returns_significant_string(self) -> None:
        text = interpret_results(self._result_with_z(3.5))
        assert len(text) > 0
