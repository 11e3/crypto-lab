"""Tests for src/backtester/engine/data_loader_base.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.backtester.engine.data_loader_base import (
    apply_strategy_signals,
    load_parquet_data,
    validate_required_columns,
)

# ---------------------------------------------------------------------------
# load_parquet_data
# ---------------------------------------------------------------------------


class TestLoadParquetData:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.parquet"
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_parquet_data(missing)

    def test_loads_and_normalises_columns(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"Open": [1.0], "Close": [2.0]})
        path = tmp_path / "data.parquet"
        df.to_parquet(path)

        result = load_parquet_data(path)

        assert "open" in result.columns
        assert "close" in result.columns
        assert "Open" not in result.columns

    def test_invalid_parquet_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.parquet"
        bad.write_bytes(b"not parquet")
        with pytest.raises(ValueError, match="Error loading data"):
            load_parquet_data(bad)

    def test_index_is_datetimeindex(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2024-01-01"]))
        path = tmp_path / "data.parquet"
        df.to_parquet(path)

        result = load_parquet_data(path)
        assert isinstance(result.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# apply_strategy_signals
# ---------------------------------------------------------------------------


class TestApplyStrategySignals:
    def test_calls_calculate_indicators_and_generate_signals(self) -> None:
        mock_strategy = MagicMock()
        df = pd.DataFrame({"close": [1.0]})

        mock_strategy.calculate_indicators.return_value = df
        mock_strategy.generate_signals.return_value = df

        result = apply_strategy_signals(df, mock_strategy)

        mock_strategy.calculate_indicators.assert_called_once_with(df)
        mock_strategy.generate_signals.assert_called_once()
        assert result is df


# ---------------------------------------------------------------------------
# validate_required_columns
# ---------------------------------------------------------------------------


class TestValidateRequiredColumns:
    def test_returns_true_when_all_columns_present(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert validate_required_columns(df, ["a", "b"]) is True

    def test_returns_false_when_column_missing(self) -> None:
        df = pd.DataFrame({"a": [1]})
        assert validate_required_columns(df, ["a", "b"]) is False

    def test_empty_required_list_returns_true(self) -> None:
        df = pd.DataFrame({"a": [1]})
        assert validate_required_columns(df, []) is True

    def test_logs_missing_columns(self, caplog: pytest.LogCaptureFixture) -> None:
        df = pd.DataFrame({"a": [1]})
        with caplog.at_level("ERROR"):
            validate_required_columns(df, ["b"], ticker="KRW-BTC")
        assert "KRW-BTC" in caplog.text
        assert "b" in caplog.text
