"""
Integration tests for CLI pipeline stage contracts.

Verifies data contracts and common bug patterns across stages:
  collect → backtest → optimize → wfa → analyze

Complements test_pipeline.py (engine-level) by testing:
  - BacktestResult field population (sortino_ratio)
  - Cache key correctness (critical bug regression)
  - Different params → different results
  - WFA period structure (no data leakage)
  - analyze skip flags
"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from src.backtester.analysis.strategy_analysis import run_strategy_analysis
from src.backtester.engine import BacktestConfig
from src.backtester.engine.data_loader import get_cache_params
from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.backtester.wfa.walk_forward_runner import generate_periods
from src.strategies.volatility_breakout.vbo_v1 import VBOV1
from tests.fixtures.data.sample_ohlcv import generate_trending_data

# ── Shared helpers ────────────────────────────────────────────────────────────


def _write_btc_parquet(dest: Path) -> Path:
    """Write 800-bar synthetic BTC parquet to dest. Returns filepath."""
    df = generate_trending_data(periods=800, trend=0.001, seed=42)
    filepath = dest / "KRW-BTC_day.parquet"
    df.to_parquet(filepath)
    return filepath


# ── Stage 1: BacktestResult field population ─────────────────────────────────


class TestBacktestResultFields:
    """BacktestResult must have all numeric fields populated after an engine run."""

    def test_sortino_ratio_is_float(self, tmp_path: Path) -> None:
        """sortino_ratio must be a float after VectorizedBacktestEngine.run()."""
        filepath = _write_btc_parquet(tmp_path)
        strategy = VBOV1(ma_short=5, btc_ma=20, noise_ratio=0.5)
        config = BacktestConfig(initial_capital=10_000_000, fee_rate=0.0005, max_slots=1)
        engine = VectorizedBacktestEngine(config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result.sortino_ratio, float)


# ── Stage 2: Cache key correctness ───────────────────────────────────────────


class TestCacheKeyCorrectness:
    """get_cache_params() must produce distinct keys for distinct parameter values.

    Regression: previously used hardcoded attribute names (sma_period, etc.) that
    don't exist on VBOV1, causing all instances to share one cache key regardless
    of their actual parameters.
    """

    def test_different_params_yield_different_keys(self) -> None:
        s1 = VBOV1(noise_ratio=0.3, ma_short=5, btc_ma=20)
        s2 = VBOV1(noise_ratio=0.7, ma_short=5, btc_ma=20)

        p1 = get_cache_params(s1)
        p2 = get_cache_params(s2)

        assert p1 != p2
        assert p1["noise_ratio"] == pytest.approx(0.3)
        assert p2["noise_ratio"] == pytest.approx(0.7)

    def test_all_schema_params_in_key(self) -> None:
        """Every parameter in parameter_schema() must appear in the cache key."""
        strategy = VBOV1(noise_ratio=0.4, ma_short=8, btc_ma=30)
        schema = VBOV1.parameter_schema()

        params = get_cache_params(strategy)

        for param_name in schema:
            assert param_name in params, (
                f"Schema param {param_name!r} missing from cache key — "
                "get_cache_params() may be using hardcoded attribute names"
            )


# ── Stage 3: Optimize — parameter diversity ───────────────────────────────────


class TestOptimizeDiversity:
    """Different parameter combinations must produce different backtest results.

    If all results are identical, it indicates a cache key collision bug.
    """

    def test_different_params_different_results(self, tmp_path: Path) -> None:
        filepath = _write_btc_parquet(tmp_path)
        config = BacktestConfig(initial_capital=10_000_000, fee_rate=0.0005, max_slots=1)
        engine = VectorizedBacktestEngine(config)

        param_sets = [
            {"noise_ratio": 0.2, "ma_short": 3, "btc_ma": 10},
            {"noise_ratio": 0.8, "ma_short": 15, "btc_ma": 40},
        ]
        results = [engine.run(VBOV1(**p), {"KRW-BTC": filepath}) for p in param_sets]

        returns = [r.total_return for r in results]
        assert len(set(returns)) > 1, (
            "All parameter combinations produced identical total_return — "
            "possible cache key collision"
        )


# ── Stage 4: WFA — period structure integrity ─────────────────────────────────


class TestWFAPeriodStructure:
    """Walk-forward period generation must satisfy date contracts."""

    def test_gap_between_opt_and_test(self) -> None:
        """test_start must equal optimization_end + 1 day (data leakage prevention)."""
        periods = generate_periods(
            date(2020, 1, 1), date(2023, 12, 31),
            optimization_days=365, test_days=90, step_days=90,
        )

        assert len(periods) > 0, "Expected at least one walk-forward period"
        for p in periods:
            expected = p.optimization_end + timedelta(days=1)
            assert p.test_start == expected, (
                f"Period {p.period_num}: test_start {p.test_start} != "
                f"opt_end+1 {expected} — data leakage gap missing"
            )

    def test_opt_and_test_windows_do_not_overlap(self) -> None:
        periods = generate_periods(
            date(2020, 1, 1), date(2023, 12, 31),
            optimization_days=365, test_days=90, step_days=90,
        )
        for p in periods:
            assert p.optimization_end < p.test_start

    def test_all_test_periods_within_end_date(self) -> None:
        end = date(2023, 12, 31)
        periods = generate_periods(
            date(2020, 1, 1), end,
            optimization_days=365, test_days=90, step_days=90,
        )
        for p in periods:
            assert p.test_end <= end


# ── Stage 5: Analyze — skip flags ────────────────────────────────────────────


class TestAnalyzeSkipFlags:
    """run_strategy_analysis with skip flags must complete and return correct fields."""

    def test_skip_perm_and_robust_returns_backtest_only(self, tmp_path: Path) -> None:
        """With both skip flags, permutation and robustness should be None."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_btc_parquet(data_dir)  # KRW-BTC_day.parquet

        config = BacktestConfig(initial_capital=10_000_000, fee_rate=0.0005, max_slots=1)

        result = run_strategy_analysis(
            strategy_name="VBOV1",
            strategy_factory_0=lambda: VBOV1(ma_short=5, btc_ma=20, noise_ratio=0.5),
            strategy_factory_p=lambda p: VBOV1(**p),
            tickers=["KRW-BTC"],
            interval="day",
            config=config,
            run_permutation=False,
            run_robustness=False,
            data_dir=data_dir,
        )

        assert result.backtest is not None
        assert result.permutation is None
        assert result.bootstrap is None
        assert result.robustness is None
        assert len(result.go_live_checks) >= 1
