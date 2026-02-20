"""Tests for strategy analysis module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.backtester.analysis.strategy_analysis import (
    BenchmarkMetrics,
    GoLiveCheck,
    StrategyAnalysisResult,
    _build_robustness_ranges,
    _compute_go_live_checks,
    _extract_optimal_params,
    run_strategy_analysis,
)
from src.backtester.models import BacktestResult


# =========================================================================
# BenchmarkMetrics
# =========================================================================


class TestBenchmarkMetrics:
    def test_fields(self) -> None:
        bm = BenchmarkMetrics(cagr=20.0, mdd=30.0, sharpe=0.8, excess_return=-5.0)
        assert bm.cagr == 20.0
        assert bm.mdd == 30.0
        assert bm.sharpe == 0.8
        assert bm.excess_return == -5.0


# =========================================================================
# _compute_go_live_checks
# =========================================================================


def _good_backtest() -> BacktestResult:
    """BacktestResult that passes Sharpe and MDD checks."""
    return BacktestResult(sharpe_ratio=1.5, mdd=10.0, cagr=25.0)


def _bad_backtest() -> BacktestResult:
    """BacktestResult that fails Sharpe and MDD checks."""
    return BacktestResult(sharpe_ratio=0.5, mdd=35.0, cagr=5.0)


def _good_permutation() -> MagicMock:
    perm = MagicMock()
    perm.z_score = 2.5
    perm.is_statistically_significant = True
    return perm


def _bad_permutation() -> MagicMock:
    perm = MagicMock()
    perm.z_score = 1.0
    perm.is_statistically_significant = False
    return perm


def _good_bootstrap() -> MagicMock:
    bs = MagicMock()
    bs.ci_return_95 = (5.0, 25.0)
    return bs


def _bad_bootstrap() -> MagicMock:
    bs = MagicMock()
    bs.ci_return_95 = (-10.0, 8.0)
    return bs


def _good_robustness() -> MagicMock:
    rob = MagicMock()
    rob.neighbor_success_rate = 0.75
    return rob


def _bad_robustness() -> MagicMock:
    rob = MagicMock()
    rob.neighbor_success_rate = 0.40
    return rob


def _good_performance() -> MagicMock:
    perf = MagicMock()
    perf.sortino_ratio = 2.5
    return perf


def _bad_performance() -> MagicMock:
    perf = MagicMock()
    perf.sortino_ratio = 0.3
    return perf


class TestGoLiveAllPass:
    def test_all_pass(self) -> None:
        checks = _compute_go_live_checks(
            backtest=_good_backtest(),
            permutation=_good_permutation(),
            bootstrap=_good_bootstrap(),
            robustness=_good_robustness(),
            performance=_good_performance(),
        )
        assert len(checks) == 5
        assert all(c.passed for c in checks)
        assert all(c.level == "PASS" for c in checks)


class TestGoLiveAllFail:
    def test_all_fail(self) -> None:
        checks = _compute_go_live_checks(
            backtest=_bad_backtest(),
            permutation=_bad_permutation(),
            bootstrap=_bad_bootstrap(),
            robustness=_bad_robustness(),
            performance=_bad_performance(),
        )
        assert len(checks) == 5
        assert all(not c.passed for c in checks)

    def test_skipped_checks_are_warn(self) -> None:
        checks = _compute_go_live_checks(
            backtest=_bad_backtest(),
            permutation=None,
            bootstrap=None,
            robustness=None,
        )
        # Permutation and robustness/bootstrap skipped → WARN
        assert checks[0].level == "WARN"  # permutation
        assert checks[3].level == "WARN"  # robustness
        assert checks[4].level == "WARN"  # bootstrap


# =========================================================================
# _build_robustness_ranges
# =========================================================================


class TestBuildRobustnessRanges:
    def test_float_range_clips_to_schema(self) -> None:
        schema = {"noise_ratio": {"type": "float", "min": 0.1, "max": 0.9, "step": 0.1}}
        optimal = {"noise_ratio": 0.2}
        ranges = _build_robustness_ranges(optimal, schema, n_steps=2)
        # optimal 0.2, ±2 steps of 0.1 → [0.0, 0.1, 0.2, 0.3, 0.4]
        # but 0.0 is clipped by min=0.1 → [0.1, 0.2, 0.3, 0.4]
        vals = ranges["noise_ratio"]
        assert min(vals) >= 0.1
        assert max(vals) <= 0.9
        assert 0.2 in vals

    def test_int_range_clips_to_schema(self) -> None:
        schema = {"ma_short": {"type": "int", "min": 3, "max": 20, "step": 1}}
        optimal = {"ma_short": 4}
        ranges = _build_robustness_ranges(optimal, schema, n_steps=2)
        # optimal 4, ±2 steps of 1 → [2, 3, 4, 5, 6]
        # 2 is clipped by min=3 → [3, 4, 5, 6]
        vals = ranges["ma_short"]
        assert min(vals) >= 3
        assert max(vals) <= 20
        assert 4 in vals

    def test_missing_param_excluded(self) -> None:
        schema = {"noise_ratio": {"type": "float", "min": 0.1, "max": 0.9, "step": 0.1}}
        optimal = {}  # param not in optimal
        ranges = _build_robustness_ranges(optimal, schema, n_steps=2)
        assert "noise_ratio" not in ranges


# =========================================================================
# _extract_optimal_params
# =========================================================================


class TestExtractOptimalParams:
    def test_extracts_via_getattr(self) -> None:
        strategy = MagicMock()
        strategy.noise_ratio = 0.5
        strategy.ma_short = 5
        schema = {
            "noise_ratio": {"type": "float", "min": 0.1, "max": 0.9, "step": 0.1},
            "ma_short": {"type": "int", "min": 3, "max": 20, "step": 1},
        }
        params = _extract_optimal_params(strategy, schema)
        assert params["noise_ratio"] == 0.5
        assert params["ma_short"] == 5

    def test_missing_attr_excluded(self) -> None:
        strategy = MagicMock(spec=[])  # no attributes
        schema = {"noise_ratio": {"type": "float", "min": 0.1, "max": 0.9, "step": 0.1}}
        params = _extract_optimal_params(strategy, schema)
        assert "noise_ratio" not in params


# =========================================================================
# run_strategy_analysis (skip-all fast path)
# =========================================================================


class TestRunStrategyAnalysisSkipAll:
    def test_skip_perm_and_robust_returns_backtest_only(self) -> None:
        """With skip_perm + skip_robust, permutation/robustness/bootstrap are None."""
        mock_result = BacktestResult(
            cagr=15.0, sharpe_ratio=1.2, mdd=8.0, total_trades=30
        )

        with patch(
            "src.backtester.analysis.strategy_analysis.run_backtest",
            return_value=mock_result,
        ), patch(
            "src.backtester.analysis.strategy_analysis._load_ticker_data",
            return_value=None,
        ):
            strategy = MagicMock()
            strategy.parameter_schema.return_value = {}

            result = run_strategy_analysis(
                strategy_name="TestStrat",
                strategy_factory_0=lambda: strategy,
                strategy_factory_p=lambda p: strategy,
                tickers=["KRW-BTC"],
                interval="day",
                config=MagicMock(),
                run_permutation=False,
                run_robustness=False,
            )

        assert isinstance(result, StrategyAnalysisResult)
        assert result.permutation is None
        assert result.robustness is None
        assert result.bootstrap is None
        assert result.backtest is mock_result
        # go_live_checks still produced (with WARN for skipped)
        assert len(result.go_live_checks) == 5
