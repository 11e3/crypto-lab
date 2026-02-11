"""Tests for data.binance_fetch module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.binance_fetch import (
    _ohlcv_to_dataframe,
    fetch_all_binance_candles,
    fetch_binance_candles,
)


@pytest.fixture
def sample_ccxt_ohlcv() -> list[list[float]]:
    """Create sample ccxt OHLCV data."""
    return [
        [1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 100.5],
        [1704153600000, 42200.0, 42800.0, 42000.0, 42600.0, 120.3],
        [1704240000000, 42600.0, 43000.0, 42400.0, 42900.0, 95.7],
    ]


class TestOhlcvToDataframe:
    """Test _ohlcv_to_dataframe conversion."""

    def test_converts_to_dataframe(self, sample_ccxt_ohlcv: list[list[float]]) -> None:
        """Test conversion to DataFrame with correct columns."""
        df = _ohlcv_to_dataframe(sample_ccxt_ohlcv)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "datetime"

    def test_correct_values(self, sample_ccxt_ohlcv: list[list[float]]) -> None:
        """Test that values are correctly placed."""
        df = _ohlcv_to_dataframe(sample_ccxt_ohlcv)
        assert df.iloc[0]["open"] == 42000.0
        assert df.iloc[0]["close"] == 42200.0
        assert len(df) == 3

    def test_timezone_naive_index(self, sample_ccxt_ohlcv: list[list[float]]) -> None:
        """Test that index is timezone-naive."""
        df = _ohlcv_to_dataframe(sample_ccxt_ohlcv)
        assert df.index.tz is None


class TestFetchBinanceCandles:
    """Test fetch_binance_candles function."""

    @patch("src.data.binance_fetch._get_exchange")
    def test_successful_fetch(
        self, mock_get_exchange: MagicMock, sample_ccxt_ohlcv: list[list[float]]
    ) -> None:
        """Test successful candle fetch."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ccxt_ohlcv
        mock_get_exchange.return_value = mock_exchange

        result = fetch_binance_candles("BTCUSDT", "1d")
        assert result is not None
        assert len(result) == 3
        mock_exchange.fetch_ohlcv.assert_called_once()

    @patch("src.data.binance_fetch._get_exchange")
    def test_empty_response(self, mock_get_exchange: MagicMock) -> None:
        """Test empty response returns None."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []
        mock_get_exchange.return_value = mock_exchange

        result = fetch_binance_candles("BTCUSDT", "1d")
        assert result is None

    @patch("src.data.binance_fetch.time.sleep")
    @patch("src.data.binance_fetch._get_exchange")
    def test_retry_on_error(
        self,
        mock_get_exchange: MagicMock,
        mock_sleep: MagicMock,
        sample_ccxt_ohlcv: list[list[float]],
    ) -> None:
        """Test retry logic on API error."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = [
            ConnectionError("API error"),
            sample_ccxt_ohlcv,
        ]
        mock_get_exchange.return_value = mock_exchange

        result = fetch_binance_candles("BTCUSDT", "1d")
        assert result is not None
        assert len(result) == 3
        assert mock_exchange.fetch_ohlcv.call_count == 2

    @patch("src.data.binance_fetch.time.sleep")
    @patch("src.data.binance_fetch._get_exchange")
    def test_all_retries_fail(self, mock_get_exchange: MagicMock, mock_sleep: MagicMock) -> None:
        """Test returns None when all retries fail."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = ConnectionError("API error")
        mock_get_exchange.return_value = mock_exchange

        result = fetch_binance_candles("BTCUSDT", "1d")
        assert result is None
        assert mock_exchange.fetch_ohlcv.call_count == 3

    @patch("src.data.binance_fetch._get_exchange")
    def test_with_since_parameter(
        self, mock_get_exchange: MagicMock, sample_ccxt_ohlcv: list[list[float]]
    ) -> None:
        """Test fetch with since parameter."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ccxt_ohlcv
        mock_get_exchange.return_value = mock_exchange

        result = fetch_binance_candles("BTCUSDT", "1d", since=1704067200000)
        assert result is not None
        mock_exchange.fetch_ohlcv.assert_called_once_with(
            symbol="BTCUSDT",
            timeframe="1d",
            since=1704067200000,
            limit=1000,
        )


class TestFetchAllBinanceCandles:
    """Test fetch_all_binance_candles function."""

    @patch("src.data.binance_fetch.time.sleep")
    @patch("src.data.binance_fetch.fetch_binance_candles")
    def test_single_page(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        """Test fetching when all data fits in one page."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [95.0] * 5,
                "close": [102.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        df.index.name = "datetime"
        mock_fetch.return_value = df

        result = fetch_all_binance_candles("BTCUSDT", "1d")
        assert result is not None
        assert len(result) == 5

    @patch("src.data.binance_fetch.time.sleep")
    @patch("src.data.binance_fetch.fetch_binance_candles")
    def test_pagination(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        """Test pagination across multiple pages."""
        # First page: 1000 candles (triggers next page)
        dates1 = pd.date_range("2024-01-01", periods=1000, freq="h")
        df1 = pd.DataFrame(
            {
                "open": [100.0] * 1000,
                "high": [105.0] * 1000,
                "low": [95.0] * 1000,
                "close": [102.0] * 1000,
                "volume": [1000.0] * 1000,
            },
            index=dates1,
        )
        df1.index.name = "datetime"

        # Second page: fewer than 1000 (stops pagination)
        dates2 = pd.date_range("2024-02-12", periods=50, freq="h")
        df2 = pd.DataFrame(
            {
                "open": [110.0] * 50,
                "high": [115.0] * 50,
                "low": [105.0] * 50,
                "close": [112.0] * 50,
                "volume": [500.0] * 50,
            },
            index=dates2,
        )
        df2.index.name = "datetime"

        mock_fetch.side_effect = [df1, df2]

        result = fetch_all_binance_candles("BTCUSDT", "1h")
        assert result is not None
        assert len(result) == 1050
        assert mock_fetch.call_count == 2

    @patch("src.data.binance_fetch.fetch_binance_candles")
    def test_no_data(self, mock_fetch: MagicMock) -> None:
        """Test returns None when no data available."""
        mock_fetch.return_value = None

        result = fetch_all_binance_candles("BTCUSDT", "1d")
        assert result is None

    @patch("src.data.binance_fetch.time.sleep")
    @patch("src.data.binance_fetch.fetch_binance_candles")
    def test_with_since_parameter(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        """Test fetching with since datetime."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [95.0] * 5,
                "close": [102.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        df.index.name = "datetime"
        mock_fetch.return_value = df

        since = datetime(2024, 1, 1)
        result = fetch_all_binance_candles("BTCUSDT", "1d", since=since)
        assert result is not None

        # Verify since was passed as milliseconds
        call_args = mock_fetch.call_args
        assert call_args is not None
        assert call_args[1]["since"] is not None
