"""Tests for backtester.analysis.permutation_test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtester.analysis.permutation_test import PermutationTester, PermutationTestResult
from src.backtester.models import BacktestConfig
from src.strategies.volatility_breakout.vbo_v1 import VBOV1


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 50_000_000.0 + rng.normal(0, 500_000, n).cumsum()
    close = np.maximum(close, 1_000_000.0)
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.997,
            "close": close,
            "volume": rng.uniform(1e9, 5e9, n),
        },
        index=pd.date_range("2022-01-01", periods=n, freq="D"),
    )
    df.index.name = "datetime"
    return df


class TestPermutationTestResult:
    def test_default_values(self) -> None:
        r = PermutationTestResult(
            original_return=0.1,
            original_sharpe=1.0,
            original_win_rate=0.5,
            shuffled_returns=[],
            shuffled_sharpes=[],
            shuffled_win_rates=[],
        )
        assert r.z_score == 0.0
        assert r.p_value == 0.0
        assert r.is_statistically_significant is False
        assert r.confidence_level == ""


class TestPermutationTester:
    def _make_tester(self, data: pd.DataFrame) -> PermutationTester:
        config = BacktestConfig(initial_capital=1_000_000.0, max_slots=1, fee_rate=0.0005)
        return PermutationTester(
            data=data,
            strategy_factory=lambda: VBOV1(noise_ratio=0.5, btc_ma=20, ma_short=3),
            backtest_config=config,
        )

    def test_run_returns_permutation_test_result(self) -> None:
        data = _make_ohlcv()
        tester = self._make_tester(data)
        result = tester.run(num_shuffles=5, verbose=False)
        assert isinstance(result, PermutationTestResult)

    def test_result_has_original_return(self) -> None:
        data = _make_ohlcv()
        tester = self._make_tester(data)
        result = tester.run(num_shuffles=3, verbose=False)
        assert isinstance(result.original_return, float)

    def test_p_value_in_range(self) -> None:
        data = _make_ohlcv()
        tester = self._make_tester(data)
        result = tester.run(num_shuffles=5, verbose=False)
        assert 0.0 <= result.p_value <= 1.0
