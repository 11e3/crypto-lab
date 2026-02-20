"""Tests for backtester.analysis.bootstrap_analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtester.analysis.bootstrap_analysis import BootstrapAnalyzer, BootstrapResult
from src.backtester.models import BacktestConfig
from src.strategies.volatility_breakout.vbo_v1 import VBOV1


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(3)
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
        index=pd.date_range("2021-01-01", periods=n, freq="D"),
    )
    df.index.name = "datetime"
    return df


class TestBootstrapResult:
    def test_default_fields(self) -> None:
        r = BootstrapResult(returns=[], sharpes=[], mdds=[])
        assert r.mean_return == 0.0
        assert r.ci_return_95 == (0.0, 0.0)


class TestBootstrapAnalyzer:
    def _make_analyzer(self, data: pd.DataFrame) -> BootstrapAnalyzer:
        config = BacktestConfig(initial_capital=1_000_000.0, max_slots=1, fee_rate=0.0005)
        return BootstrapAnalyzer(
            data=data,
            strategy_factory=lambda: VBOV1(noise_ratio=0.5, btc_ma=20, ma_short=3),
            backtest_config=config,
        )

    def test_analyze_returns_bootstrap_result(self) -> None:
        data = _make_ohlcv()
        analyzer = self._make_analyzer(data)
        result = analyzer.analyze(n_samples=5, block_size=30)
        assert isinstance(result, BootstrapResult)

    def test_ci_tuple_has_two_elements(self) -> None:
        data = _make_ohlcv()
        analyzer = self._make_analyzer(data)
        result = analyzer.analyze(n_samples=5, block_size=30)
        assert len(result.ci_return_95) == 2

    def test_mean_return_is_float(self) -> None:
        data = _make_ohlcv()
        analyzer = self._make_analyzer(data)
        result = analyzer.analyze(n_samples=5, block_size=30)
        assert isinstance(result.mean_return, float)
