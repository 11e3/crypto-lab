"""Tests for data.binance_source module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.binance_source import BinanceDataSource
from src.exceptions.data import DataSourceConnectionError, DataSourceError


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory."""
    data_dir = tmp_path / "data" / "binance"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def source(temp_data_dir: Path) -> BinanceDataSource:
    """Create BinanceDataSource instance for testing."""
    return BinanceDataSource(data_dir=temp_data_dir)


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


@pytest.fixture
def sample_ccxt_ohlcv() -> list[list[float]]:
    """Create sample ccxt OHLCV data."""
    return [
        [1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 100.5],
        [1704153600000, 42200.0, 42800.0, 42000.0, 42600.0, 120.3],
        [1704240000000, 42600.0, 43000.0, 42400.0, 42900.0, 95.7],
    ]


class TestBinanceDataSourceInit:
    """Test BinanceDataSource initialization."""

    def test_initialization(self, temp_data_dir: Path) -> None:
        """Test initialization with custom data dir."""
        source = BinanceDataSource(data_dir=temp_data_dir)
        assert source.data_dir == temp_data_dir

    def test_initialization_creates_directory(self, tmp_path: Path) -> None:
        """Test initialization creates directory."""
        data_dir = tmp_path / "new_binance"
        BinanceDataSource(data_dir=data_dir)
        assert data_dir.exists()


class TestBinanceGetOhlcv:
    """Test get_ohlcv method."""

    @patch.object(BinanceDataSource, "_get_exchange")
    def test_get_ohlcv_success(
        self,
        mock_get_exchange: MagicMock,
        source: BinanceDataSource,
        sample_ccxt_ohlcv: list[list[float]],
    ) -> None:
        """Test successful OHLCV fetch."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ccxt_ohlcv
        mock_get_exchange.return_value = mock_exchange

        result = source.get_ohlcv("BTCUSDT", "1d", count=200)
        assert result is not None
        assert len(result) == 3
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    @patch.object(BinanceDataSource, "_get_exchange")
    def test_get_ohlcv_empty(
        self,
        mock_get_exchange: MagicMock,
        source: BinanceDataSource,
    ) -> None:
        """Test OHLCV returns None for empty data."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []
        mock_get_exchange.return_value = mock_exchange

        result = source.get_ohlcv("BTCUSDT", "1d")
        assert result is None

    @patch.object(BinanceDataSource, "_get_exchange")
    def test_get_ohlcv_error(
        self,
        mock_get_exchange: MagicMock,
        source: BinanceDataSource,
    ) -> None:
        """Test OHLCV raises DataSourceConnectionError on failure."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = ConnectionError("API error")
        mock_get_exchange.return_value = mock_exchange

        with pytest.raises(DataSourceConnectionError):
            source.get_ohlcv("BTCUSDT", "1d")


class TestBinanceGetCurrentPrice:
    """Test get_current_price method."""

    @patch.object(BinanceDataSource, "_get_exchange")
    def test_get_current_price_success(
        self,
        mock_get_exchange: MagicMock,
        source: BinanceDataSource,
    ) -> None:
        """Test successful price fetch."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {"last": 42500.0}
        mock_get_exchange.return_value = mock_exchange

        price = source.get_current_price("BTCUSDT")
        assert price == 42500.0

    @patch.object(BinanceDataSource, "_get_exchange")
    def test_get_current_price_error(
        self,
        mock_get_exchange: MagicMock,
        source: BinanceDataSource,
    ) -> None:
        """Test price fetch raises DataSourceError on failure."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.side_effect = ConnectionError("API error")
        mock_get_exchange.return_value = mock_exchange

        with pytest.raises(DataSourceError):
            source.get_current_price("BTCUSDT")


class TestBinanceSaveLoadOhlcv:
    """Test save_ohlcv and load_ohlcv methods."""

    def test_save_and_load_roundtrip(
        self, source: BinanceDataSource, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test save and load produces equivalent data."""
        success = source.save_ohlcv("BTCUSDT", "1d", sample_ohlcv_data)
        assert success is True

        loaded = source.load_ohlcv("BTCUSDT", "1d")
        assert loaded is not None
        assert len(loaded) == 5

    def test_load_nonexistent(self, source: BinanceDataSource) -> None:
        """Test loading non-existent file returns None."""
        result = source.load_ohlcv("BTCUSDT", "1d")
        assert result is None

    def test_save_with_custom_filepath(
        self, source: BinanceDataSource, sample_ohlcv_data: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test saving with custom filepath."""
        filepath = tmp_path / "custom.parquet"
        success = source.save_ohlcv("BTCUSDT", "1d", sample_ohlcv_data, filepath=filepath)
        assert success is True
        assert filepath.exists()


class TestBinanceUpdateOhlcv:
    """Test update_ohlcv method."""

    @patch.object(BinanceDataSource, "get_ohlcv")
    def test_update_no_existing_data(
        self,
        mock_get_ohlcv: MagicMock,
        source: BinanceDataSource,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test update with no existing data fetches full dataset."""
        mock_get_ohlcv.return_value = sample_ohlcv_data

        result = source.update_ohlcv("BTCUSDT", "1d")
        assert result is not None
        assert len(result) == 5

    @patch.object(BinanceDataSource, "get_ohlcv")
    def test_update_incremental(
        self,
        mock_get_ohlcv: MagicMock,
        source: BinanceDataSource,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test incremental update merges new data."""
        # Save existing data (first 3 rows)
        existing = sample_ohlcv_data.iloc[:3].copy()
        source.save_ohlcv("BTCUSDT", "1d", existing)

        # Return new data including overlap
        mock_get_ohlcv.return_value = sample_ohlcv_data.iloc[2:]

        result = source.update_ohlcv("BTCUSDT", "1d")
        assert result is not None
        assert len(result) >= 5

    @patch.object(BinanceDataSource, "get_ohlcv")
    def test_update_no_new_data(
        self,
        mock_get_ohlcv: MagicMock,
        source: BinanceDataSource,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test update when no new data available."""
        source.save_ohlcv("BTCUSDT", "1d", sample_ohlcv_data)
        mock_get_ohlcv.return_value = None

        result = source.update_ohlcv("BTCUSDT", "1d")
        assert result is not None
        assert len(result) == 5
