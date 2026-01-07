"""Tests for data.collector_factory module."""

import pytest

from src.data.collector import UpbitDataCollector
from src.data.collector_factory import DataCollectorFactory


class TestCollectorFactory:
    """Test collector factory."""

    def test_create_upbit(self) -> None:
        """Test DataCollectorFactory.create with upbit."""
        collector = DataCollectorFactory.create("upbit")
        assert isinstance(collector, UpbitDataCollector)

    def test_create_invalid_type(self) -> None:
        """Test DataCollectorFactory.create with invalid type."""
        with pytest.raises(ValueError):
            DataCollectorFactory.create("invalid_exchange")

    def test_create_default(self) -> None:
        """Test DataCollectorFactory.create with default (None)."""
        collector = DataCollectorFactory.create(None)
        assert collector is not None

    def test_create_upbit_case_insensitive(self) -> None:
        """Test DataCollectorFactory.create is case-insensitive."""
        collector = DataCollectorFactory.create("UPBIT")
        assert isinstance(collector, UpbitDataCollector)
