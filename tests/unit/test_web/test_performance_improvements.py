"""Tests for web UI performance improvements.

Tests for:
- VectorizedBacktestEngine integration
- Metrics calculation determinism
- Parallel data loading
- Chart data downsampling
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import VectorizedBacktestEngine
from src.backtester.models import BacktestConfig
from src.strategies.mean_reversion import MeanReversionStrategy

# Import web services only when streamlit is available
try:
    from src.web.services.data_loader import load_multiple_tickers_parallel
    from src.web.services.metrics_calculator import calculate_extended_metrics

    WEB_SERVICES_AVAILABLE = True
except ImportError:
    WEB_SERVICES_AVAILABLE = False


class TestVectorizedEngineIntegration:
    """Test VectorizedBacktestEngine produces valid results."""

    @pytest.fixture
    def mock_data_files(self, tmp_path: Path) -> dict[str, Path]:
        """Create mock data files for testing."""
        data_files = {}
        for ticker in ["KRW-BTC", "KRW-ETH"]:
            dates = pd.date_range("2024-01-01", periods=1000, freq="1h")
            df = pd.DataFrame(
                {
                    "open": np.random.uniform(50000, 60000, 1000),
                    "high": np.random.uniform(60000, 70000, 1000),
                    "low": np.random.uniform(40000, 50000, 1000),
                    "close": np.random.uniform(50000, 60000, 1000),
                    "volume": np.random.uniform(100, 1000, 1000),
                },
                index=dates,
            )
            file_path = tmp_path / f"{ticker}.parquet"
            df.to_parquet(file_path)
            data_files[ticker] = file_path

        return data_files

    def test_vectorized_engine_produces_valid_results(
        self, mock_data_files: dict[str, Path]
    ) -> None:
        """Test that VectorizedBacktestEngine produces valid results."""
        config = BacktestConfig(initial_capital=10_000_000)
        vectorized_engine = VectorizedBacktestEngine(config)

        strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        result = vectorized_engine.run(strategy, mock_data_files)

        assert result is not None
        assert result.strategy_name == "MeanReversionStrategy"
        assert len(result.equity_curve) > 0


@pytest.mark.skipif(not WEB_SERVICES_AVAILABLE, reason="Streamlit not available")
class TestMetricsCalculationCaching:
    """Test metrics calculation caching."""

    def test_calculate_extended_metrics_is_deterministic(self) -> None:
        """Test that calculate_extended_metrics produces same results."""
        equity = np.linspace(1_000_000, 1_500_000, 1000)
        trade_returns = [0.05, -0.02, 0.03, -0.01, 0.04]

        # Calculate twice
        metrics1 = calculate_extended_metrics(equity, trade_returns)
        metrics2 = calculate_extended_metrics(equity, trade_returns)

        # Should be identical (for caching to work)
        assert metrics1.total_return_pct == metrics2.total_return_pct
        assert metrics1.sharpe_ratio == metrics2.sharpe_ratio
        assert metrics1.max_drawdown_pct == metrics2.max_drawdown_pct

    def test_metrics_handles_large_dataset(self) -> None:
        """Test that metrics calculation handles large datasets without error."""
        equity = np.linspace(1_000_000, 1_500_000, 10000)
        trade_returns = np.random.normal(0.01, 0.05, 1000).tolist()

        metrics = calculate_extended_metrics(equity, trade_returns)

        assert metrics.num_trades == 1000
        assert np.isfinite(metrics.total_return_pct)
        assert np.isfinite(metrics.sharpe_ratio)


@pytest.mark.skipif(not WEB_SERVICES_AVAILABLE, reason="Streamlit not available")
class TestParallelDataLoading:
    """Test parallel data loading for multiple tickers."""

    @pytest.fixture
    def mock_data_dir(self, tmp_path: Path) -> Path:
        """Create mock data directory with multiple ticker files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        for ticker in ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA"]:
            dates = pd.date_range("2024-01-01", periods=1000, freq="1h")
            df = pd.DataFrame(
                {
                    "open": np.random.uniform(50000, 60000, 1000),
                    "high": np.random.uniform(60000, 70000, 1000),
                    "low": np.random.uniform(40000, 50000, 1000),
                    "close": np.random.uniform(50000, 60000, 1000),
                    "volume": np.random.uniform(100, 1000, 1000),
                },
                index=dates,
            )
            file_path = data_dir / f"{ticker}_minute60.parquet"
            df.to_parquet(file_path)

        return data_dir

    def test_parallel_loading_returns_all_tickers(self, mock_data_dir: Path) -> None:
        """Test that parallel loading returns data for all available tickers."""
        tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA"]
        interval = "minute60"

        # Mock RAW_DATA_DIR
        with patch("src.web.services.data_loader.RAW_DATA_DIR", mock_data_dir):
            # Parallel loading
            parallel_data = load_multiple_tickers_parallel(tickers, interval)

            # All tickers should be loaded
            assert len(parallel_data) == 4
            for ticker in tickers:
                assert ticker in parallel_data
                assert len(parallel_data[ticker]) == 1000

    def test_parallel_loading_handles_missing_files(self, mock_data_dir: Path) -> None:
        """Test that parallel loading handles missing files gracefully."""
        tickers = ["KRW-BTC", "KRW-MISSING", "KRW-ETH"]
        interval = "minute60"

        with patch("src.web.services.data_loader.RAW_DATA_DIR", mock_data_dir):
            data = load_multiple_tickers_parallel(tickers, interval)

            # Only existing tickers should be loaded
            assert len(data) == 2
            assert "KRW-BTC" in data
            assert "KRW-ETH" in data
            assert "KRW-MISSING" not in data


@pytest.mark.skipif(not WEB_SERVICES_AVAILABLE, reason="Streamlit not available")
class TestChartDataDownsampling:
    """Test chart data downsampling for large datasets."""

    def test_downsample_large_dataset(self) -> None:
        """Test downsampling reduces data points while preserving shape."""
        from src.web.utils.chart_utils import downsample_timeseries

        # Generate large dataset (10k points)
        dates = pd.date_range("2020-01-01", periods=10000, freq="1h")
        equity = np.linspace(1_000_000, 2_000_000, 10000)

        # Downsample to 1000 points
        downsampled_dates, downsampled_equity = downsample_timeseries(
            dates, equity, max_points=1000
        )

        # Should have ~1000 points
        assert len(downsampled_dates) <= 1100  # Allow some margin
        assert len(downsampled_dates) >= 900
        assert len(downsampled_dates) == len(downsampled_equity)

        # Should preserve start and end values
        np.testing.assert_almost_equal(downsampled_equity[0], equity[0])
        np.testing.assert_almost_equal(downsampled_equity[-1], equity[-1])

    def test_downsample_small_dataset_unchanged(self) -> None:
        """Test that small datasets are not downsampled."""
        from src.web.utils.chart_utils import downsample_timeseries

        dates = pd.date_range("2024-01-01", periods=500, freq="1h")
        equity = np.linspace(1_000_000, 1_500_000, 500)

        downsampled_dates, downsampled_equity = downsample_timeseries(
            dates, equity, max_points=1000
        )

        # Should be unchanged
        assert len(downsampled_dates) == 500
        np.testing.assert_array_equal(downsampled_dates, dates)
        np.testing.assert_array_equal(downsampled_equity, equity)
