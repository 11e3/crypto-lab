"""Tests for data.collector_factory module."""

import pytest

from src.data.binance_collector import BinanceDataCollector
from src.data.collector import UpbitDataCollector
from src.data.collector_factory import DataCollectorFactory


class TestCollectorFactory:
    """Test collector factory."""

    def test_create_upbit(self) -> None:
        """Test DataCollectorFactory.create with upbit."""
        collector = DataCollectorFactory.create("upbit")
        assert isinstance(collector, UpbitDataCollector)

    def test_create_binance(self) -> None:
        """Test DataCollectorFactory.create with binance."""
        collector = DataCollectorFactory.create("binance")
        assert isinstance(collector, BinanceDataCollector)

    def test_create_binance_case_insensitive(self) -> None:
        """Test DataCollectorFactory.create is case-insensitive for binance."""
        collector = DataCollectorFactory.create("BINANCE")
        assert isinstance(collector, BinanceDataCollector)

    def test_create_binance_custom_data_dir(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test DataCollectorFactory.create with custom data_dir for binance."""
        collector = DataCollectorFactory.create("binance", data_dir=tmp_path)  # type: ignore[arg-type]
        assert isinstance(collector, BinanceDataCollector)

    def test_create_invalid_type(self) -> None:
        """Test DataCollectorFactory.create with invalid type."""
        with pytest.raises(ValueError, match="Unsupported exchange"):
            DataCollectorFactory.create("invalid_exchange")

    def test_create_default(self) -> None:
        """Test DataCollectorFactory.create with default (None)."""
        collector = DataCollectorFactory.create(None)
        assert collector is not None

    def test_create_upbit_case_insensitive(self) -> None:
        """Test DataCollectorFactory.create is case-insensitive."""
        collector = DataCollectorFactory.create("UPBIT")
        assert isinstance(collector, UpbitDataCollector)
