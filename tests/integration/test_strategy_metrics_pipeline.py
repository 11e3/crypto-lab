"""Integration tests: Strategy → Backtest → Extended Metrics pipeline.

Tests the complete flow from strategy configuration through backtesting
to metrics calculation, verifying that the pipeline produces consistent
and valid results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.backtester.models import BacktestConfig
from src.strategies.volatility_breakout.vbo_v1 import VBOV1
from src.web.services.metrics_calculator import (
    ExtendedMetrics,
    calculate_extended_metrics,
)


@pytest.fixture()
def engine_config() -> BacktestConfig:
    return BacktestConfig(
        initial_capital=10_000_000,
        fee_rate=0.0005,
        slippage_rate=0.0002,
        max_slots=3,
    )


@pytest.fixture()
def data_files(
    temp_data_dir: Path,
    trending_ohlcv_data: pd.DataFrame,
) -> dict[str, Path]:
    filepath = temp_data_dir / "KRW-BTC_day.parquet"
    trending_ohlcv_data.to_parquet(filepath)
    return {"KRW-BTC": filepath}


class TestStrategyToMetricsPipeline:
    """End-to-end: Strategy → VectorizedEngine → ExtendedMetrics."""

    def test_backtest_result_feeds_metrics(
        self,
        engine_config: BacktestConfig,
        data_files: dict[str, Path],
    ) -> None:
        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(engine_config)
        result = engine.run(strategy, data_files)

        # Feed result into extended metrics
        metrics = calculate_extended_metrics(
            equity=result.equity_curve,
            dates=result.dates,
        )

        assert isinstance(metrics, ExtendedMetrics)
        assert metrics.trading_days == len(result.equity_curve)
        assert np.isfinite(metrics.total_return_pct)
        assert np.isfinite(metrics.sharpe_ratio)
        assert metrics.max_drawdown_pct >= 0.0

    def test_metrics_with_trade_returns(
        self,
        engine_config: BacktestConfig,
        data_files: dict[str, Path],
    ) -> None:
        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(engine_config)
        result = engine.run(strategy, data_files)

        trade_returns = [t.pnl_pct for t in result.trades]

        metrics = calculate_extended_metrics(
            equity=result.equity_curve,
            dates=result.dates,
            trade_returns=trade_returns,
        )

        assert metrics.num_trades == len(result.trades)
        if result.total_trades > 0:
            assert metrics.win_rate_pct >= 0.0
            assert metrics.win_rate_pct <= 100.0

    def test_metrics_total_return_matches_engine(
        self,
        engine_config: BacktestConfig,
        data_files: dict[str, Path],
    ) -> None:
        """Extended metrics total return should match engine result."""
        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(engine_config)
        result = engine.run(strategy, data_files)

        metrics = calculate_extended_metrics(
            equity=result.equity_curve,
            dates=result.dates,
        )

        # Both metrics compute from the same equity curve, so total return
        # should match (metrics uses equity[0] and equity[-1] directly)
        expected = (result.equity_curve[-1] / result.equity_curve[0] - 1) * 100
        assert metrics.total_return_pct == pytest.approx(expected, abs=0.01)

    def test_var_cvar_relationship(
        self,
        engine_config: BacktestConfig,
        data_files: dict[str, Path],
    ) -> None:
        """CVaR should always be >= VaR at same confidence level."""
        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(engine_config)
        result = engine.run(strategy, data_files)

        metrics = calculate_extended_metrics(
            equity=result.equity_curve,
            dates=result.dates,
        )

        # CVaR (Expected Shortfall) >= VaR
        assert metrics.cvar_95_pct >= metrics.var_95_pct
        assert metrics.cvar_99_pct >= metrics.var_99_pct
        # VaR99 >= VaR95
        assert metrics.var_99_pct >= metrics.var_95_pct


class TestMultiTickerMetrics:
    """Multi-ticker backtest → metrics pipeline."""

    def test_portfolio_metrics(
        self,
        engine_config: BacktestConfig,
        sample_parquet_files: dict[str, Path],
    ) -> None:
        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(engine_config)
        result = engine.run(strategy, sample_parquet_files)

        metrics = calculate_extended_metrics(
            equity=result.equity_curve,
            dates=result.dates,
            trade_returns=[t.pnl_pct for t in result.trades],
        )

        assert metrics.trading_days > 0
        assert np.isfinite(metrics.volatility_pct)
        assert np.isfinite(metrics.sharpe_ratio)
        assert 0.0 <= metrics.p_value <= 1.0


class TestStrategyComparisonMetrics:
    """Compare different strategy configurations through metrics."""

    def test_different_params_both_valid(
        self,
        engine_config: BacktestConfig,
        data_files: dict[str, Path],
    ) -> None:
        """Different strategy params should both produce valid metrics."""
        strategy_a = VBOV1(ma_short=3, btc_ma=10)
        strategy_b = VBOV1(ma_short=10, btc_ma=30)

        engine = VectorizedBacktestEngine(engine_config)
        result_a = engine.run(strategy_a, data_files)
        result_b = engine.run(strategy_b, data_files)

        metrics_a = calculate_extended_metrics(equity=result_a.equity_curve, dates=result_a.dates)
        metrics_b = calculate_extended_metrics(equity=result_b.equity_curve, dates=result_b.dates)

        # Both should produce valid, finite results
        for m in [metrics_a, metrics_b]:
            assert np.isfinite(m.total_return_pct)
            assert np.isfinite(m.sharpe_ratio)
            assert m.max_drawdown_pct >= 0.0
            assert 0.0 <= m.p_value <= 1.0
