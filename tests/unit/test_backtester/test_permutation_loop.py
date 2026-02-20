"""Tests for backtester.analysis.permutation_loop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtester.analysis.permutation_loop import run_permutation_loop
from src.strategies.volatility_breakout.vbo_v1 import VBOV1


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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


def _factory() -> VBOV1:
    return VBOV1(noise_ratio=0.5, btc_ma=20, ma_short=3)


class TestRunPermutationLoop:
    def test_returns_three_lists(self) -> None:
        data = _make_ohlcv()
        returns, sharpes, win_rates = run_permutation_loop(
            data=data,
            strategy_factory=_factory,
            initial_capital=1_000_000.0,
            num_shuffles=5,
            shuffle_columns=["close"],
            verbose=False,
        )
        assert isinstance(returns, list)
        assert isinstance(sharpes, list)
        assert isinstance(win_rates, list)

    def test_at_most_num_shuffles_results(self) -> None:
        data = _make_ohlcv()
        returns, _, _ = run_permutation_loop(
            data=data,
            strategy_factory=_factory,
            initial_capital=1_000_000.0,
            num_shuffles=3,
            shuffle_columns=["close"],
            verbose=False,
        )
        assert len(returns) <= 3
