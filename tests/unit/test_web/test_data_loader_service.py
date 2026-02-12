"""Tests for web data loader service."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.web.services.data_loader import (
    get_data_date_range,
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


# =========================================================================
# load_multiple_tickers_parallel — exception in inner function
# =========================================================================


class TestLoadMultipleTickersParallelExceptionHandling:
    """Tests for exception handling in the inner load_single_ticker function."""

    @patch("src.web.services.data_loader.load_ticker_data")
    def test_exception_in_load_returns_none(self, mock_load: MagicMock) -> None:
        """When load_ticker_data raises, the ticker is excluded from results."""
        mock_load.side_effect = RuntimeError("network error")
        result = load_multiple_tickers_parallel(["KRW-BTC"], "day", max_workers=1)
        assert result == {}

    @patch("src.web.services.data_loader.load_ticker_data")
    def test_exception_mixed_with_success(self, mock_load: MagicMock) -> None:
        """One ticker raises, another succeeds — only the success is kept."""
        mock_df = pd.DataFrame({"close": [100]})

        def _side_effect(ticker: str, *args: object, **kwargs: object) -> pd.DataFrame:
            if ticker == "KRW-BTC":
                raise RuntimeError("corrupt")
            return mock_df

        mock_load.side_effect = _side_effect
        result = load_multiple_tickers_parallel(["KRW-BTC", "KRW-ETH"], "day", max_workers=1)
        assert "KRW-BTC" not in result
        assert "KRW-ETH" in result
        assert len(result) == 1


# =========================================================================
# get_data_date_range
# =========================================================================


class TestGetDataDateRange:
    """Tests for get_data_date_range."""

    @patch("src.web.services.data_loader.get_data_dir")
    def test_no_files_returns_none_none(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        """Empty directory returns (None, None)."""
        mock_dir.return_value = tmp_path
        result = get_data_date_range.__wrapped__("day")  # type: ignore[attr-defined]
        assert result == (None, None)

    @patch("src.web.services.data_loader.get_data_dir")
    def test_datetime_index(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        """File with DatetimeIndex returns correct min/max dates."""
        mock_dir.return_value = tmp_path
        dates = pd.date_range("2023-06-01", periods=10, freq="D")
        df = pd.DataFrame({"close": range(10)}, index=dates)
        df.to_parquet(tmp_path / "KRW-BTC_day.parquet")

        min_date, max_date = get_data_date_range.__wrapped__("day")  # type: ignore[attr-defined]
        assert min_date == date(2023, 6, 1)
        assert max_date == date(2023, 6, 10)

    @patch("src.web.services.data_loader.get_data_dir")
    def test_datetime_column(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        """File with 'datetime' column (not index) returns correct dates."""
        mock_dir.return_value = tmp_path
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-15", periods=5, freq="D"),
                "close": range(5),
            }
        )
        df.to_parquet(tmp_path / "KRW-ETH_day.parquet")

        min_date, max_date = get_data_date_range.__wrapped__("day")  # type: ignore[attr-defined]
        assert min_date == date(2024, 1, 15)
        assert max_date == date(2024, 1, 19)

    @patch("src.web.services.data_loader.get_data_dir")
    def test_no_recognized_date_format_skipped(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        """File without DatetimeIndex or 'datetime' column is skipped."""
        mock_dir.return_value = tmp_path

        # File with recognized dates
        good_df = pd.DataFrame(
            {"close": range(3)},
            index=pd.date_range("2023-03-01", periods=3, freq="D"),
        )
        good_df.to_parquet(tmp_path / "KRW-BTC_day.parquet")

        # File without recognized date format (integer index, no 'datetime' column)
        bad_df = pd.DataFrame({"close": [1, 2, 3], "value": [10, 20, 30]})
        bad_df.to_parquet(tmp_path / "KRW-XRP_day.parquet")

        min_date, max_date = get_data_date_range.__wrapped__("day")  # type: ignore[attr-defined]
        # Should only reflect the good file
        assert min_date == date(2023, 3, 1)
        assert max_date == date(2023, 3, 3)

    @patch("src.web.services.data_loader.get_data_dir")
    def test_corrupted_file_handled(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        """Corrupted parquet file triggers exception handler, continues."""
        mock_dir.return_value = tmp_path

        # Write garbage bytes as a .parquet file
        (tmp_path / "KRW-BAD_day.parquet").write_bytes(b"not a parquet file")

        # Valid file alongside the corrupted one
        df = pd.DataFrame(
            {"close": range(5)},
            index=pd.date_range("2024-07-01", periods=5, freq="D"),
        )
        df.to_parquet(tmp_path / "KRW-BTC_day.parquet")

        min_date, max_date = get_data_date_range.__wrapped__("day")  # type: ignore[attr-defined]
        assert min_date == date(2024, 7, 1)
        assert max_date == date(2024, 7, 5)

    @patch("src.web.services.data_loader.get_data_dir")
    def test_multiple_files_overall_min_max(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        """Multiple files returns the overall min and max across all."""
        mock_dir.return_value = tmp_path

        df1 = pd.DataFrame(
            {"close": range(5)},
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )
        df1.to_parquet(tmp_path / "KRW-BTC_day.parquet")

        df2 = pd.DataFrame(
            {"close": range(3)},
            index=pd.date_range("2022-06-15", periods=3, freq="D"),
        )
        df2.to_parquet(tmp_path / "KRW-ETH_day.parquet")

        df3 = pd.DataFrame(
            {"close": range(2)},
            index=pd.date_range("2024-12-30", periods=2, freq="D"),
        )
        df3.to_parquet(tmp_path / "KRW-XRP_day.parquet")

        min_date, max_date = get_data_date_range.__wrapped__("day")  # type: ignore[attr-defined]
        assert min_date == date(2022, 6, 15)
        assert max_date == date(2024, 12, 31)

    @patch("src.web.services.data_loader.get_data_dir")
    def test_only_corrupted_files_returns_none_none(
        self, mock_dir: MagicMock, tmp_path: Path
    ) -> None:
        """When all files are corrupted, returns (None, None)."""
        mock_dir.return_value = tmp_path
        (tmp_path / "KRW-BAD_day.parquet").write_bytes(b"garbage")

        min_date, max_date = get_data_date_range.__wrapped__("day")  # type: ignore[attr-defined]
        assert min_date is None
        assert max_date is None
