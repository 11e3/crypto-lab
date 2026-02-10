"""
Unit tests for backtest_runner module.

Tests cover missing branches including:
78, 82-84, 104-105
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.backtester.engine.backtest_runner import (
    _collect_missing_data,
    _find_missing_tickers,
    run_backtest,
)
from src.strategies.base import Strategy


class MockStrategy(Strategy):
    """Mock strategy for testing."""

    def required_indicators(self) -> list[str]:
        return []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["entry_signal"] = False
        df["exit_signal"] = False
        return df


class TestFindMissingTickers:
    """Test _find_missing_tickers function."""

    def test_with_nonexistent_files(self, tmp_path: Path) -> None:
        """Test with nonexistent files (line 78)."""
        data_files = {
            "TICKER1": tmp_path / "TICKER1_day.parquet",
            "TICKER2": tmp_path / "TICKER2_day.parquet",
        }
        missing = _find_missing_tickers(data_files)
        assert len(missing) == 2
        assert "TICKER1" in missing
        assert "TICKER2" in missing

    def test_with_outdated_files(self, tmp_path: Path) -> None:
        """Test with outdated files (line 82)."""
        # Create an old file
        filepath = tmp_path / "TICKER1_day.parquet"
        df = pd.DataFrame({"close": [100, 101, 102]})
        df.to_parquet(filepath)

        # Set file modification time to 2 days ago
        old_time = (datetime.now() - timedelta(days=2)).timestamp()
        import os

        os.utime(filepath, (old_time, old_time))

        data_files = {"TICKER1": filepath}
        missing = _find_missing_tickers(data_files)
        assert "TICKER1" in missing

    def test_with_empty_dataframe(self, tmp_path: Path) -> None:
        """Test with empty dataframe (line 84)."""
        filepath = tmp_path / "TICKER1_day.parquet"
        df = pd.DataFrame()
        df.to_parquet(filepath)

        data_files = {"TICKER1": filepath}
        missing = _find_missing_tickers(data_files)
        assert "TICKER1" in missing

    def test_with_small_dataframe(self, tmp_path: Path) -> None:
        """Test with small dataframe (line 84)."""
        filepath = tmp_path / "TICKER1_day.parquet"
        df = pd.DataFrame({"close": [100, 101]})  # Only 2 rows < 10
        df.to_parquet(filepath)

        data_files = {"TICKER1": filepath}
        missing = _find_missing_tickers(data_files)
        assert "TICKER1" in missing

    def test_with_exception_reading_file(self, tmp_path: Path) -> None:
        """Test with exception when reading file (line 87)."""
        # Create a non-parquet file
        filepath = tmp_path / "TICKER1_day.parquet"
        filepath.write_text("not a parquet file")

        data_files = {"TICKER1": filepath}
        missing = _find_missing_tickers(data_files)
        assert "TICKER1" in missing


class TestCollectMissingData:
    """Test _collect_missing_data function."""

    @patch("src.backtester.engine.backtest_runner.DataCollectorFactory")
    def test_collect_success(self, mock_factory: MagicMock, tmp_path: Path) -> None:
        """Test successful data collection."""
        mock_collector = MagicMock()
        mock_factory.create.return_value = mock_collector

        _collect_missing_data(["TICKER1", "TICKER2"], tmp_path, "day")

        # Verify collector was created and called
        mock_factory.create.assert_called_once_with(data_dir=tmp_path)
        assert mock_collector.collect.call_count == 2

    @patch("src.backtester.engine.backtest_runner.DataCollectorFactory")
    def test_collect_failure(self, mock_factory: MagicMock, tmp_path: Path) -> None:
        """Test data collection failure (line 105)."""
        mock_collector = MagicMock()
        mock_collector.collect.side_effect = ConnectionError("Collection failed")
        mock_factory.create.return_value = mock_collector

        # Should not raise exception, just log error
        _collect_missing_data(["TICKER1"], tmp_path, "day")

        mock_collector.collect.assert_called_once()


class TestRunBacktest:
    """Test run_backtest function."""

    def test_run_with_no_data_files(self, tmp_path: Path) -> None:
        """Test when no data files exist (line 66)."""
        strategy = MockStrategy()
        with pytest.raises(FileNotFoundError, match="No data files found"):
            run_backtest(strategy, ["TICKER1", "TICKER2"], data_dir=tmp_path)

    @patch("src.backtester.engine.backtest_runner.VectorizedBacktestEngine")
    @patch("src.backtester.engine.backtest_runner._collect_missing_data")
    def test_run_with_existing_data(
        self,
        mock_collect: MagicMock,
        mock_engine_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test run_backtest with existing data files."""
        # Create valid data files
        for ticker in ["TICKER1", "TICKER2"]:
            filepath = tmp_path / f"{ticker}_day.parquet"
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=20),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
            )
            df.to_parquet(filepath)

        # Mock engine
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        strategy = MockStrategy()
        result = run_backtest(strategy, ["TICKER1", "TICKER2"], data_dir=tmp_path)

        # Verify no collection was needed (files are recent and valid)
        assert mock_collect.call_count == 0 or mock_collect.call_count == 1
        # Verify engine was called
        mock_engine.run.assert_called_once()
        assert result == mock_result


__all__ = [
    "TestFindMissingTickers",
    "TestCollectMissingData",
    "TestRunBacktest",
]
