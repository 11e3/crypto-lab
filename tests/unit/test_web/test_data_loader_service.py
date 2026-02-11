"""Tests for web data loader service."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.web.services.data_loader import (
    get_data_dir,
    get_data_files,
    load_multiple_tickers_parallel,
    load_ticker_data,
    validate_data_availability,
)

# =========================================================================
# get_data_dir
# =========================================================================


class TestGetDataDir:
    """Tests for get_data_dir."""

    @patch("src.web.services.data_loader.BINANCE_DATA_DIR", Path("/data/binance"))
    @patch("src.web.services.data_loader.RAW_DATA_DIR", Path("/data/upbit"))
    def test_upbit_default(self) -> None:
        assert get_data_dir() == Path("/data/upbit")

    @patch("src.web.services.data_loader.BINANCE_DATA_DIR", Path("/data/binance"))
    @patch("src.web.services.data_loader.RAW_DATA_DIR", Path("/data/upbit"))
    def test_binance(self) -> None:
        assert get_data_dir("binance") == Path("/data/binance")

    @patch("src.web.services.data_loader.RAW_DATA_DIR", Path("/data/upbit"))
    def test_unknown_exchange_defaults_to_upbit(self) -> None:
        assert get_data_dir("unknown") == Path("/data/upbit")


# =========================================================================
# get_data_files
# =========================================================================


class TestGetDataFiles:
    """Tests for get_data_files."""

    @patch("src.web.services.data_loader.get_data_dir")
    @patch("src.web.services.data_loader.parquet_filename")
    def test_existing_files(
        self,
        mock_pf: MagicMock,
        mock_dir: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Create temp files
        mock_dir.return_value = tmp_path
        (tmp_path / "KRW-BTC_day.parquet").touch()
        (tmp_path / "KRW-ETH_day.parquet").touch()

        mock_pf.side_effect = lambda t, i: f"{t}_{i}.parquet"

        result = get_data_files(["KRW-BTC", "KRW-ETH"], "day")
        assert "KRW-BTC" in result
        assert "KRW-ETH" in result
        assert len(result) == 2

    @patch("src.web.services.data_loader.get_data_dir")
    @patch("src.web.services.data_loader.parquet_filename")
    def test_missing_files_excluded(
        self,
        mock_pf: MagicMock,
        mock_dir: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_dir.return_value = tmp_path
        (tmp_path / "KRW-BTC_day.parquet").touch()
        mock_pf.side_effect = lambda t, i: f"{t}_{i}.parquet"

        result = get_data_files(["KRW-BTC", "KRW-XRP"], "day")
        assert "KRW-BTC" in result
        assert "KRW-XRP" not in result

    @patch("src.web.services.data_loader.get_data_dir")
    @patch("src.web.services.data_loader.parquet_filename")
    def test_empty_tickers(
        self,
        mock_pf: MagicMock,
        mock_dir: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_dir.return_value = tmp_path
        result = get_data_files([], "day")
        assert result == {}


# =========================================================================
# validate_data_availability
# =========================================================================


class TestValidateDataAvailability:
    """Tests for validate_data_availability."""

    @patch("src.web.services.data_loader.get_data_dir")
    @patch("src.web.services.data_loader.parquet_filename")
    def test_mixed_availability(
        self,
        mock_pf: MagicMock,
        mock_dir: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_dir.return_value = tmp_path
        (tmp_path / "KRW-BTC_day.parquet").touch()
        mock_pf.side_effect = lambda t, i: f"{t}_{i}.parquet"

        available, missing = validate_data_availability(["KRW-BTC", "KRW-ETH"], "day")
        assert available == ["KRW-BTC"]
        assert missing == ["KRW-ETH"]

    @patch("src.web.services.data_loader.get_data_dir")
    @patch("src.web.services.data_loader.parquet_filename")
    def test_all_available(
        self,
        mock_pf: MagicMock,
        mock_dir: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_dir.return_value = tmp_path
        (tmp_path / "KRW-BTC_day.parquet").touch()
        (tmp_path / "KRW-ETH_day.parquet").touch()
        mock_pf.side_effect = lambda t, i: f"{t}_{i}.parquet"

        available, missing = validate_data_availability(["KRW-BTC", "KRW-ETH"], "day")
        assert len(available) == 2
        assert len(missing) == 0


# =========================================================================
# load_ticker_data
# =========================================================================


class TestLoadTickerData:
    """Tests for load_ticker_data."""

    @patch("src.web.services.data_loader.get_data_dir")
    @patch("src.web.services.data_loader.parquet_filename")
    def test_file_not_found_returns_none(
        self,
        mock_pf: MagicMock,
        mock_dir: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_dir.return_value = tmp_path
        mock_pf.return_value = "nonexistent.parquet"

        # load_ticker_data is decorated with @st.cache_data; call underlying
        result = load_ticker_data.__wrapped__(  # type: ignore[attr-defined]
            "KRW-BTC", "day"
        )
        assert result is None

    @patch("src.web.services.data_loader.get_data_dir")
    @patch("src.web.services.data_loader.parquet_filename")
    def test_loads_parquet_data(
        self,
        mock_pf: MagicMock,
        mock_dir: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Create a real parquet file
        df = pd.DataFrame(
            {"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]},
            index=pd.DatetimeIndex([pd.Timestamp("2023-01-01")]),
        )
        fpath = tmp_path / "KRW-BTC_day.parquet"
        df.to_parquet(fpath)

        mock_dir.return_value = tmp_path
        mock_pf.return_value = "KRW-BTC_day.parquet"

        result = load_ticker_data.__wrapped__(  # type: ignore[attr-defined]
            "KRW-BTC", "day"
        )
        assert result is not None
        assert len(result) == 1

    @patch("src.web.services.data_loader.get_data_dir")
    @patch("src.web.services.data_loader.parquet_filename")
    def test_date_filtering(
        self,
        mock_pf: MagicMock,
        mock_dir: MagicMock,
        tmp_path: Path,
    ) -> None:
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {"close": range(30)},
            index=dates,
        )
        fpath = tmp_path / "KRW-BTC_day.parquet"
        df.to_parquet(fpath)

        mock_dir.return_value = tmp_path
        mock_pf.return_value = "KRW-BTC_day.parquet"

        result = load_ticker_data.__wrapped__(  # type: ignore[attr-defined]
            "KRW-BTC",
            "day",
            start_date=date(2023, 1, 10),
            end_date=date(2023, 1, 20),
        )
        assert result is not None
        assert len(result) == 11  # Jan 10 to Jan 20 inclusive

    @patch("src.web.services.data_loader.get_data_dir")
    @patch("src.web.services.data_loader.parquet_filename")
    @patch("src.web.services.data_loader.pd.read_parquet")
    def test_exception_returns_none(
        self,
        mock_read: MagicMock,
        mock_pf: MagicMock,
        mock_dir: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_dir.return_value = tmp_path
        mock_pf.return_value = "KRW-BTC_day.parquet"
        (tmp_path / "KRW-BTC_day.parquet").touch()
        mock_read.side_effect = Exception("corrupt file")

        result = load_ticker_data.__wrapped__(  # type: ignore[attr-defined]
            "KRW-BTC", "day"
        )
        assert result is None


# =========================================================================
# load_multiple_tickers_parallel
# =========================================================================


class TestLoadMultipleTickersParallel:
    """Tests for load_multiple_tickers_parallel."""

    @patch("src.web.services.data_loader.load_ticker_data")
    def test_parallel_loading(self, mock_load: MagicMock) -> None:
        mock_df = pd.DataFrame({"close": [100, 110]})
        mock_load.return_value = mock_df

        result = load_multiple_tickers_parallel(["KRW-BTC", "KRW-ETH"], "day", max_workers=2)
        assert len(result) == 2
        assert "KRW-BTC" in result
        assert "KRW-ETH" in result

    @patch("src.web.services.data_loader.load_ticker_data")
    def test_skips_none_results(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = [
            pd.DataFrame({"close": [100]}),
            None,
        ]

        result = load_multiple_tickers_parallel(["KRW-BTC", "KRW-ETH"], "day", max_workers=2)
        # Only 1 successful load
        assert len(result) == 1

    @patch("src.web.services.data_loader.load_ticker_data")
    def test_empty_tickers(self, mock_load: MagicMock) -> None:
        result = load_multiple_tickers_parallel([], "day")
        assert result == {}
