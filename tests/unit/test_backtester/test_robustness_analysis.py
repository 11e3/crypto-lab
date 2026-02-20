"""Tests for backtester.analysis.robustness_analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtester.analysis.robustness_analysis import RobustnessAnalyzer
from src.backtester.analysis.robustness_models import RobustnessReport
from src.backtester.models import BacktestConfig
from src.strategies.volatility_breakout.vbo_v1 import VBOV1


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
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
        index=pd.date_range("2021-01-01", periods=n, freq="D"),
    )
    df.index.name = "datetime"
    return df


class TestRobustnessAnalyzer:
    def _make_analyzer(self, data: pd.DataFrame) -> RobustnessAnalyzer:
        config = BacktestConfig(initial_capital=1_000_000.0, max_slots=1, fee_rate=0.0005)
        return RobustnessAnalyzer(
            data=data,
            strategy_factory=lambda p: VBOV1(**p),
            backtest_config=config,
        )

    def test_analyze_returns_robustness_report(self) -> None:
        data = _make_ohlcv()
        analyzer = self._make_analyzer(data)
        report = analyzer.analyze(
            optimal_params={"noise_ratio": 0.5, "btc_ma": 20, "ma_short": 3},
            parameter_ranges={
                "noise_ratio": [0.4, 0.5, 0.6],
                "btc_ma": [20],
                "ma_short": [3],
            },
            verbose=False,
        )
        assert isinstance(report, RobustnessReport)

    def test_results_list_length_matches_combinations(self) -> None:
        data = _make_ohlcv()
        analyzer = self._make_analyzer(data)
        report = analyzer.analyze(
            optimal_params={"noise_ratio": 0.5, "btc_ma": 20, "ma_short": 3},
            parameter_ranges={
                "noise_ratio": [0.4, 0.6],
                "btc_ma": [20],
                "ma_short": [3],
            },
            verbose=False,
        )
        # 2 * 1 * 1 = 2 combinations
        assert len(report.results) == 2

    def test_mean_return_is_float(self) -> None:
        data = _make_ohlcv()
        analyzer = self._make_analyzer(data)
        report = analyzer.analyze(
            optimal_params={"noise_ratio": 0.5, "btc_ma": 20, "ma_short": 3},
            parameter_ranges={"noise_ratio": [0.5], "btc_ma": [20], "ma_short": [3]},
            verbose=False,
        )
        assert isinstance(report.mean_return, float)

    def test_sensitivity_scores_is_dict(self) -> None:
        data = _make_ohlcv()
        analyzer = self._make_analyzer(data)
        report = analyzer.analyze(
            optimal_params={"noise_ratio": 0.5, "btc_ma": 20, "ma_short": 3},
            parameter_ranges={
                "noise_ratio": [0.4, 0.5, 0.6],
                "btc_ma": [20],
                "ma_short": [3],
            },
            verbose=False,
        )
        assert isinstance(report.sensitivity_scores, dict)
