"""
Unit tests for data collector module.

Tests both the UpbitDataCollector class and module-level fetch functions.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.collector import UpbitDataCollector
from src.data.collector_fetch import fetch_all_candles, fetch_candles


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def collector(temp_data_dir: Path) -> UpbitDataCollector:
    """Create UpbitDataCollector instance for testing."""
    return UpbitDataCollector(data_dir=temp_data_dir)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(5)],
            "high": [105.0 + i for i in range(5)],
            "low": [95.0 + i for i in range(5)],
            "close": [102.0 + i for i in range(5)],
            "volume": [1000.0 + i * 10 for i in range(5)],
        },
        index=dates,
    )


class TestUpbitDataCollector:
    """Test cases for UpbitDataCollector class."""

    def test_initialization(self, temp_data_dir: Path) -> None:
        """Test UpbitDataCollector initialization."""
        collector = UpbitDataCollector(data_dir=temp_data_dir)

        assert collector.data_dir == temp_data_dir
        assert temp_data_dir.exists()

    def test_initialization_creates_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates directory if it doesn't exist."""
        data_dir = tmp_path / "new_data"
        UpbitDataCollector(data_dir=data_dir)

        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_get_parquet_path(self, collector: UpbitDataCollector) -> None:
        """Test parquet path generation."""
        path = collector._get_parquet_path("KRW-BTC", "day")

        assert path == collector.data_dir / "KRW-BTC_day.parquet"

    def test_load_existing_data_exists(
        self, collector: UpbitDataCollector, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test loading existing data when file exists."""
        filepath = collector._get_parquet_path("KRW-BTC", "day")
        sample_ohlcv_data.to_parquet(filepath)

        result = collector._load_existing_data(filepath)

        assert result is not None
        assert len(result) == len(sample_ohlcv_data)
        assert "open" in result.columns

    def test_load_existing_data_not_exists(self, collector: UpbitDataCollector) -> None:
        """Test loading existing data when file doesn't exist."""
        filepath = collector._get_parquet_path("KRW-BTC", "day")

        result = collector._load_existing_data(filepath)

        assert result is None


class TestFetchFunctions:
    """Test cases for module-level fetch functions."""

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    def test_fetch_candles(
        self,
        mock_get_ohlcv: MagicMock,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test fetching candles from API."""
        mock_get_ohlcv.return_value = sample_ohlcv_data

        result = fetch_candles("KRW-BTC", "day", count=5)

        assert result is not None
        assert len(result) == len(sample_ohlcv_data)
        mock_get_ohlcv.assert_called_once()

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    def test_fetch_candles_error(self, mock_get_ohlcv: MagicMock) -> None:
        """Test fetching candles when API returns error."""
        mock_get_ohlcv.side_effect = Exception("API Error")

        result = fetch_candles("KRW-BTC", "day")

        assert result is None

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    def test_fetch_candles_none(self, mock_get_ohlcv: MagicMock) -> None:
        """Test fetching candles when API returns None."""
        mock_get_ohlcv.return_value = None

        result = fetch_candles("KRW-BTC", "day")

        assert result is None

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    @patch("src.data.collector_fetch.time.sleep")  # Mock sleep to speed up tests
    def test_fetch_all_candles(self, mock_sleep: MagicMock, mock_get_ohlcv: MagicMock) -> None:
        """Test fetching all candles with pagination."""
        from src.config.constants import UPBIT_MAX_CANDLES_PER_REQUEST

        # Create mock data for pagination (first page has max candles, second has fewer)
        dates1 = pd.date_range("2024-01-06", periods=UPBIT_MAX_CANDLES_PER_REQUEST, freq="D")
        dates2 = pd.date_range("2024-01-01", periods=3, freq="D")
        df1 = pd.DataFrame(
            {
                "open": [103.0] * UPBIT_MAX_CANDLES_PER_REQUEST,
                "high": [108.0] * UPBIT_MAX_CANDLES_PER_REQUEST,
                "low": [98.0] * UPBIT_MAX_CANDLES_PER_REQUEST,
                "close": [105.0] * UPBIT_MAX_CANDLES_PER_REQUEST,
                "volume": [1030.0] * UPBIT_MAX_CANDLES_PER_REQUEST,
            },
            index=dates1,
        )
        df2 = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [95.0, 96.0, 97.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [1000.0, 1010.0, 1020.0],
            },
            index=dates2,
        )

        # First call returns df1 (max candles), second call returns df2 (fewer, stops loop)
        mock_get_ohlcv.side_effect = [df1, df2]

        result = fetch_all_candles("KRW-BTC", "day")

        assert result is not None
        assert len(result) == UPBIT_MAX_CANDLES_PER_REQUEST + 3  # Combined data

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    @patch("src.data.collector_fetch.time.sleep")
    def test_fetch_all_candles_with_since(
        self, mock_sleep: MagicMock, mock_get_ohlcv: MagicMock
    ) -> None:
        """Test fetching all candles with since parameter."""
        since = datetime(2024, 1, 3)
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0 + i for i in range(5)],
                "high": [105.0 + i for i in range(5)],
                "low": [95.0 + i for i in range(5)],
                "close": [102.0 + i for i in range(5)],
                "volume": [1000.0 + i * 10 for i in range(5)],
            },
            index=dates,
        )
        mock_get_ohlcv.return_value = df

        result = fetch_all_candles("KRW-BTC", "day", since=since)

        assert result is not None
        # Only rows after 'since' should be included
        assert all(result.index > since)

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    @patch("src.data.collector_fetch.time.sleep")
    def test_fetch_all_candles_with_since_empty_after_filter(
        self, mock_sleep: MagicMock, mock_get_ohlcv: MagicMock
    ) -> None:
        """Test fetching all candles with since parameter when filtered result is empty."""
        since = datetime(2024, 1, 10)  # Future date, so all data will be filtered out
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0 + i for i in range(5)],
                "high": [105.0 + i for i in range(5)],
                "low": [95.0 + i for i in range(5)],
                "close": [102.0 + i for i in range(5)],
                "volume": [1000.0 + i * 10 for i in range(5)],
            },
            index=dates,
        )
        mock_get_ohlcv.return_value = df

        result = fetch_all_candles("KRW-BTC", "day", since=since)

        # Result should be None since all data is filtered out (df.empty triggers break)
        assert result is None


class TestUpbitDataCollectorCollect:
    """Test cases for UpbitDataCollector.collect methods."""

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    @patch("src.data.collector_fetch.time.sleep")
    def test_collect_new_data(
        self,
        mock_sleep: MagicMock,
        mock_get_ohlcv: MagicMock,
        collector: UpbitDataCollector,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test collect with new data (no existing file)."""
        mock_get_ohlcv.return_value = sample_ohlcv_data

        count = collector.collect("KRW-BTC", "day")

        assert count == len(sample_ohlcv_data)
        filepath = collector._get_parquet_path("KRW-BTC", "day")
        assert filepath.exists()

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    @patch("src.data.collector_fetch.time.sleep")
    def test_collect_incremental_update(
        self,
        mock_sleep: MagicMock,
        mock_get_ohlcv: MagicMock,
        collector: UpbitDataCollector,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test collect with incremental update (existing file)."""
        # Save existing data
        filepath = collector._get_parquet_path("KRW-BTC", "day")
        existing_df = sample_ohlcv_data.iloc[:3].copy()
        existing_df.to_parquet(filepath)

        # New data (after existing)
        dates = pd.date_range("2024-01-04", periods=2, freq="D")
        new_df = pd.DataFrame(
            {
                "open": [103.0, 104.0],
                "high": [108.0, 109.0],
                "low": [98.0, 99.0],
                "close": [105.0, 106.0],
                "volume": [1030.0, 1040.0],
            },
            index=dates,
        )
        mock_get_ohlcv.return_value = new_df

        count = collector.collect("KRW-BTC", "day")

        assert count == 2  # 2 new rows
        # Verify combined data
        loaded_df = pd.read_parquet(filepath)
        assert len(loaded_df) == 5  # 3 existing + 2 new

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    @patch("src.data.collector_fetch.time.sleep")
    def test_collect_full_refresh(
        self,
        mock_sleep: MagicMock,
        mock_get_ohlcv: MagicMock,
        collector: UpbitDataCollector,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test collect with full_refresh=True."""
        # Save existing data
        filepath = collector._get_parquet_path("KRW-BTC", "day")
        sample_ohlcv_data.to_parquet(filepath)

        # New data
        new_df = sample_ohlcv_data.copy()
        mock_get_ohlcv.return_value = new_df

        count = collector.collect("KRW-BTC", "day", full_refresh=True)

        assert count == len(new_df)

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    @patch("src.data.collector_fetch.time.sleep")
    def test_collect_no_new_data(
        self, mock_sleep: MagicMock, mock_get_ohlcv: MagicMock, collector: UpbitDataCollector
    ) -> None:
        """Test collect when no new data available."""
        mock_get_ohlcv.return_value = None

        count = collector.collect("KRW-BTC", "day")

        assert count == 0

    @patch("src.data.collector_fetch.pyupbit.get_ohlcv")
    @patch("src.data.collector_fetch.time.sleep")
    def test_collect_multiple(
        self,
        mock_sleep: MagicMock,
        mock_get_ohlcv: MagicMock,
        collector: UpbitDataCollector,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test collect_multiple method."""
        mock_get_ohlcv.return_value = sample_ohlcv_data

        results = collector.collect_multiple(["KRW-BTC", "KRW-ETH"], ["day"])

        assert len(results) == 2
        assert "KRW-BTC_day" in results
        assert "KRW-ETH_day" in results
        assert results["KRW-BTC_day"] == len(sample_ohlcv_data)
        assert results["KRW-ETH_day"] == len(sample_ohlcv_data)

    @patch("src.data.collector_fetch.time.sleep")
    def test_collect_multiple_error(
        self, mock_sleep: MagicMock, collector: UpbitDataCollector
    ) -> None:
        """Test collect_multiple with exception during collect."""
        # Mock collect to raise an exception
        with patch.object(collector, "collect", side_effect=ConnectionError("Collection error")):
            results = collector.collect_multiple(["KRW-BTC"], ["day"])

        # collect_multiple catches exceptions and sets -1 for failed items
        assert "KRW-BTC_day" in results
        assert results["KRW-BTC_day"] == -1
