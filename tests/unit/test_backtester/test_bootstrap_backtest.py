"""Tests for backtester.analysis.bootstrap_backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtester.analysis.bootstrap_backtest import simple_backtest_vectorized
from src.backtester.models import BacktestResult
from src.strategies.volatility_breakout.vbo_v1 import VBOV1


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(5)
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


class TestSimpleBacktestVectorized:
    def test_returns_backtest_result(self) -> None:
        data = _make_ohlcv()
        strategy = VBOV1(noise_ratio=0.5, btc_ma=20, ma_short=3)
        result = simple_backtest_vectorized(data, strategy, initial_capital=1_000_000.0)
        assert isinstance(result, BacktestResult)

    def test_total_return_is_float(self) -> None:
        data = _make_ohlcv()
        strategy = VBOV1(noise_ratio=0.5, btc_ma=20, ma_short=3)
        result = simple_backtest_vectorized(data, strategy, initial_capital=1_000_000.0)
        assert isinstance(result.total_return, float)

    def test_win_rate_in_valid_range(self) -> None:
        data = _make_ohlcv()
        strategy = VBOV1(noise_ratio=0.5, btc_ma=20, ma_short=3)
        result = simple_backtest_vectorized(data, strategy, initial_capital=1_000_000.0)
        assert 0.0 <= result.win_rate <= 1.0
