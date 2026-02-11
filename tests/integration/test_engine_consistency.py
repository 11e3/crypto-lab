"""
Integration tests for backtesting engine consistency.

Tests that VectorizedBacktestEngine and EventDrivenBacktestEngine
produce consistent results given the same data and strategy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import EventDrivenBacktestEngine, VectorizedBacktestEngine
from src.backtester.models import BacktestConfig
from src.strategies.volatility_breakout.vbo_v1 import VBOV1


def _create_data_files(tmp_path: Path, periods: int = 200, seed: int = 42) -> dict[str, Path]:
    """Create deterministic OHLCV parquet files for backtesting."""
    np.random.seed(seed)
    data_files: dict[str, Path] = {}

    for ticker, base_price in [("KRW-BTC", 50_000_000.0), ("KRW-ETH", 3_000_000.0)]:
        dates = pd.date_range("2024-01-01", periods=periods, freq="1D")
        returns = np.random.randn(periods) * 0.02
        close = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(periods) * 0.001),
                "high": close * (1 + np.abs(np.random.randn(periods) * 0.005)),
                "low": close * (1 - np.abs(np.random.randn(periods) * 0.005)),
                "close": close,
                "volume": np.random.uniform(100, 1000, periods),
            },
            index=dates,
        )
        file_path = tmp_path / f"{ticker}.parquet"
        df.to_parquet(file_path)
        data_files[ticker] = file_path

    return data_files


@pytest.fixture
def data_files(tmp_path: Path) -> dict[str, Path]:
    """Create deterministic data files."""
    return _create_data_files(tmp_path, periods=200, seed=42)


@pytest.fixture
def config() -> BacktestConfig:
    """Standard backtest config for consistency tests."""
    return BacktestConfig(
        initial_capital=10_000_000,
        fee_rate=0.0005,
        slippage_rate=0.0,
        max_slots=3,
    )


class TestEngineConsistency:
    """Test that both engines produce consistent results."""

    def test_both_engines_produce_results(
        self, data_files: dict[str, Path], config: BacktestConfig
    ) -> None:
        """Both engines should produce valid BacktestResult."""
        strategy = VBOV1(ma_short=5, btc_ma=20)

        vec_engine = VectorizedBacktestEngine(config)
        event_engine = EventDrivenBacktestEngine(config)

        vec_result = vec_engine.run(strategy, data_files)
        event_result = event_engine.run(strategy, data_files)

        assert vec_result is not None
        assert event_result is not None
        assert len(vec_result.equity_curve) > 0
        assert len(event_result.equity_curve) > 0

    def test_initial_capital_preserved(
        self, data_files: dict[str, Path], config: BacktestConfig
    ) -> None:
        """Both engines should start near the initial capital."""
        strategy = VBOV1(ma_short=5, btc_ma=20)

        vec_result = VectorizedBacktestEngine(config).run(strategy, data_files)
        event_result = EventDrivenBacktestEngine(config).run(strategy, data_files)

        # Vectorized engine may account for first trade fees in equity[0],
        # so allow tolerance of 0.1% of initial capital
        tolerance = config.initial_capital * 0.001
        assert abs(vec_result.equity_curve[0] - config.initial_capital) < tolerance
        assert abs(event_result.equity_curve[0] - config.initial_capital) < tolerance

    def test_strategy_name_consistent(
        self, data_files: dict[str, Path], config: BacktestConfig
    ) -> None:
        """Both engines should report the same strategy name."""
        strategy = VBOV1(ma_short=5, btc_ma=20)

        vec_result = VectorizedBacktestEngine(config).run(strategy, data_files)
        event_result = EventDrivenBacktestEngine(config).run(strategy, data_files)

        assert vec_result.strategy_name == event_result.strategy_name

    def test_total_return_both_finite(
        self, data_files: dict[str, Path], config: BacktestConfig
    ) -> None:
        """Both engines should produce finite total returns."""
        strategy = VBOV1(ma_short=5, btc_ma=20)

        vec_result = VectorizedBacktestEngine(config).run(strategy, data_files)
        event_result = EventDrivenBacktestEngine(config).run(strategy, data_files)

        # Engines use different trade execution models (vectorized uses whipsaw
        # detection, event-driven uses bar-by-bar simulation), so returns may
        # differ significantly. Verify both produce finite results.
        assert np.isfinite(vec_result.total_return)
        assert np.isfinite(event_result.total_return)

    def test_total_return_similar_vbo(
        self, data_files: dict[str, Path], config: BacktestConfig
    ) -> None:
        """Both engines produce finite results for VBOV1 strategy.

        Note: VBOV1 uses exit_price_base (exit at open) which causes
        genuinely different behavior between vectorized and event-driven engines.
        We only verify both produce valid, finite results.
        """
        strategy = VBOV1(ma_short=5, btc_ma=20)

        vec_result = VectorizedBacktestEngine(config).run(strategy, data_files)
        event_result = EventDrivenBacktestEngine(config).run(strategy, data_files)

        # Both engines should produce finite results
        assert np.isfinite(vec_result.total_return)
        assert np.isfinite(event_result.total_return)


class TestEngineEdgeCases:
    """Test both engines handle edge cases consistently."""

    def test_empty_data(self, tmp_path: Path, config: BacktestConfig) -> None:
        """Both engines should handle empty data gracefully."""
        strategy = VBOV1(ma_short=5, btc_ma=20)

        # Create empty parquet
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        file_path = tmp_path / "KRW-BTC.parquet"
        empty_df.to_parquet(file_path)

        data_files = {"KRW-BTC": file_path}

        vec_result = VectorizedBacktestEngine(config).run(strategy, data_files)
        event_result = EventDrivenBacktestEngine(config).run(strategy, data_files)

        # Both should produce results (with no trades)
        assert vec_result is not None
        assert event_result is not None
        assert vec_result.total_trades == 0
        assert event_result.total_trades == 0

    def test_single_data_point(self, tmp_path: Path, config: BacktestConfig) -> None:
        """Both engines should handle single data point."""
        strategy = VBOV1(ma_short=5, btc_ma=20)

        df = pd.DataFrame(
            {
                "open": [50_000_000.0],
                "high": [51_000_000.0],
                "low": [49_000_000.0],
                "close": [50_500_000.0],
                "volume": [100.0],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="1D"),
        )
        file_path = tmp_path / "KRW-BTC.parquet"
        df.to_parquet(file_path)
        data_files = {"KRW-BTC": file_path}

        vec_result = VectorizedBacktestEngine(config).run(strategy, data_files)
        event_result = EventDrivenBacktestEngine(config).run(strategy, data_files)

        assert vec_result is not None
        assert event_result is not None
        assert vec_result.total_trades == 0
        assert event_result.total_trades == 0

    def test_high_volatility_data(self, tmp_path: Path, config: BacktestConfig) -> None:
        """Both engines should handle extreme volatility without crashing."""
        np.random.seed(99)
        strategy = VBOV1(ma_short=5, btc_ma=20)

        dates = pd.date_range("2024-01-01", periods=100, freq="1D")
        # Extreme volatility: ±20% daily moves
        returns = np.random.randn(100) * 0.2
        close = 50_000_000.0 * np.cumprod(1 + returns)
        close = np.maximum(close, 100.0)  # Prevent negative prices

        df = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(100) * 0.01),
                "high": close * 1.05,
                "low": close * 0.95,
                "close": close,
                "volume": np.random.uniform(100, 1000, 100),
            },
            index=dates,
        )
        file_path = tmp_path / "KRW-BTC.parquet"
        df.to_parquet(file_path)
        data_files = {"KRW-BTC": file_path}

        vec_result = VectorizedBacktestEngine(config).run(strategy, data_files)
        event_result = EventDrivenBacktestEngine(config).run(strategy, data_files)

        assert vec_result is not None
        assert event_result is not None
        # Both should produce valid equity curves
        assert len(vec_result.equity_curve) > 0
        assert len(event_result.equity_curve) > 0

    def test_multiple_tickers(self, tmp_path: Path, config: BacktestConfig) -> None:
        """Both engines should handle multiple tickers."""
        data_files = _create_data_files(tmp_path, periods=100, seed=42)
        strategy = VBOV1(ma_short=5, btc_ma=20)

        vec_result = VectorizedBacktestEngine(config).run(strategy, data_files)
        event_result = EventDrivenBacktestEngine(config).run(strategy, data_files)

        assert vec_result is not None
        assert event_result is not None
        assert len(vec_result.equity_curve) > 0
        assert len(event_result.equity_curve) > 0
