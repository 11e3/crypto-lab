"""Tests for data.binance_collector module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.binance_collector import BinanceDataCollector


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory."""
    data_dir = tmp_path / "data" / "binance"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def collector(temp_data_dir: Path) -> BinanceDataCollector:
    """Create BinanceDataCollector instance for testing."""
    return BinanceDataCollector(data_dir=temp_data_dir)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [42000.0 + i * 100 for i in range(5)],
            "high": [42500.0 + i * 100 for i in range(5)],
            "low": [41800.0 + i * 100 for i in range(5)],
            "close": [42200.0 + i * 100 for i in range(5)],
            "volume": [100.0 + i * 10 for i in range(5)],
        },
        index=dates,
    )
    df.index.name = "datetime"
    return df


class TestBinanceDataCollector:
    """Test cases for BinanceDataCollector class."""

    def test_initialization(self, temp_data_dir: Path) -> None:
        """Test BinanceDataCollector initialization."""
        collector = BinanceDataCollector(data_dir=temp_data_dir)
        assert collector.data_dir == temp_data_dir
        assert temp_data_dir.exists()

    def test_initialization_creates_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates directory if it doesn't exist."""
        data_dir = tmp_path / "new_binance_data"
        BinanceDataCollector(data_dir=data_dir)
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_get_parquet_path(self, collector: BinanceDataCollector) -> None:
        """Test parquet path generation."""
        path = collector._get_parquet_path("BTCUSDT", "1d")
        assert path == collector.data_dir / "BTCUSDT_1d.parquet"

    def test_load_existing_data_exists(
        self, collector: BinanceDataCollector, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test loading existing data when file exists."""
        filepath = collector._get_parquet_path("BTCUSDT", "1d")
        sample_ohlcv_data.to_parquet(filepath)

        result = collector._load_existing_data(filepath)
        assert result is not None
        assert len(result) == 5

    def test_load_existing_data_not_exists(self, collector: BinanceDataCollector) -> None:
        """Test loading data when file doesn't exist returns None."""
        filepath = collector._get_parquet_path("BTCUSDT", "1d")
        result = collector._load_existing_data(filepath)
        assert result is None


class TestBinanceCollectorCollect:
    """Test collect method."""

    @patch("src.data.binance_collector.fetch_all_binance_candles")
    def test_collect_new_data(
        self,
        mock_fetch: MagicMock,
        collector: BinanceDataCollector,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test collecting new data with no existing file."""
        mock_fetch.return_value = sample_ohlcv_data

        count = collector.collect("BTCUSDT", "1d")
        assert count == 5

        # Verify file was saved
        filepath = collector._get_parquet_path("BTCUSDT", "1d")
        assert filepath.exists()
        loaded = pd.read_parquet(filepath)
        assert len(loaded) == 5

    @patch("src.data.binance_collector.fetch_all_binance_candles")
    def test_collect_incremental_update(
        self,
        mock_fetch: MagicMock,
        collector: BinanceDataCollector,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test incremental update with existing data."""
        # Save existing data (first 3 rows)
        filepath = collector._get_parquet_path("BTCUSDT", "1d")
        existing = sample_ohlcv_data.iloc[:3].copy()
        existing.index.name = "datetime"
        existing.to_parquet(filepath)

        # New data: last 3 rows (overlaps at index 2)
        new_data = sample_ohlcv_data.iloc[2:].copy()
        mock_fetch.return_value = new_data

        count = collector.collect("BTCUSDT", "1d")
        assert count == 3

        # Verify merged data
        loaded = pd.read_parquet(filepath)
        assert len(loaded) == 5

    @patch("src.data.binance_collector.fetch_all_binance_candles")
    def test_collect_full_refresh(
        self,
        mock_fetch: MagicMock,
        collector: BinanceDataCollector,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test full refresh ignores existing data."""
        # Save existing data
        filepath = collector._get_parquet_path("BTCUSDT", "1d")
        sample_ohlcv_data.to_parquet(filepath)

        # New data (different)
        new_dates = pd.date_range("2024-02-01", periods=3, freq="D")
        new_data = pd.DataFrame(
            {
                "open": [50000.0] * 3,
                "high": [51000.0] * 3,
                "low": [49000.0] * 3,
                "close": [50500.0] * 3,
                "volume": [200.0] * 3,
            },
            index=new_dates,
        )
        new_data.index.name = "datetime"
        mock_fetch.return_value = new_data

        count = collector.collect("BTCUSDT", "1d", full_refresh=True)
        assert count == 3

        # Full refresh should not include existing data
        loaded = pd.read_parquet(filepath)
        assert len(loaded) == 3

    @patch("src.data.binance_collector.fetch_all_binance_candles")
    def test_collect_no_new_data(
        self,
        mock_fetch: MagicMock,
        collector: BinanceDataCollector,
    ) -> None:
        """Test returns 0 when no new data available."""
        mock_fetch.return_value = None

        count = collector.collect("BTCUSDT", "1d")
        assert count == 0

    @patch("src.data.binance_collector.fetch_all_binance_candles")
    def test_collect_empty_dataframe(
        self,
        mock_fetch: MagicMock,
        collector: BinanceDataCollector,
    ) -> None:
        """Test returns 0 when fetch returns empty DataFrame."""
        mock_fetch.return_value = pd.DataFrame()

        count = collector.collect("BTCUSDT", "1d")
        assert count == 0


class TestBinanceCollectorCollectMultiple:
    """Test collect_multiple method."""

    @patch("src.data.binance_collector.time.sleep")
    @patch("src.data.binance_collector.fetch_all_binance_candles")
    def test_collect_multiple_success(
        self,
        mock_fetch: MagicMock,
        mock_sleep: MagicMock,
        collector: BinanceDataCollector,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test collecting multiple symbols and intervals."""
        mock_fetch.return_value = sample_ohlcv_data

        results = collector.collect_multiple(
            tickers=["BTCUSDT", "ETHUSDT"],
            intervals=["1d"],
        )

        assert "BTCUSDT_1d" in results
        assert "ETHUSDT_1d" in results
        assert results["BTCUSDT_1d"] == 5
        assert results["ETHUSDT_1d"] == 5

    @patch("src.data.binance_collector.time.sleep")
    @patch("src.data.binance_collector.fetch_all_binance_candles")
    def test_collect_multiple_with_error(
        self,
        mock_fetch: MagicMock,
        mock_sleep: MagicMock,
        collector: BinanceDataCollector,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test collect_multiple handles errors gracefully."""
        mock_fetch.side_effect = [
            sample_ohlcv_data,
            ConnectionError("API error"),
        ]

        results = collector.collect_multiple(
            tickers=["BTCUSDT", "ETHUSDT"],
            intervals=["1d"],
        )

        assert results["BTCUSDT_1d"] == 5
        assert results["ETHUSDT_1d"] == -1
