"""Tests for data.exchange_mapping module."""

import pytest

from src.data.exchange_mapping import (
    BINANCE_TO_CANONICAL,
    CANONICAL_TO_BINANCE,
    CANONICAL_TO_UPBIT,
    UPBIT_TO_CANONICAL,
    get_exchange_intervals,
)


class TestIntervalMapping:
    """Test interval mapping between exchanges."""

    def test_upbit_to_canonical_day(self) -> None:
        """Test Upbit day interval maps to canonical."""
        assert UPBIT_TO_CANONICAL["day"] == "1d"

    def test_upbit_to_canonical_minute240(self) -> None:
        """Test Upbit minute240 maps to 4h."""
        assert UPBIT_TO_CANONICAL["minute240"] == "4h"

    def test_upbit_to_canonical_week(self) -> None:
        """Test Upbit week maps to 1w."""
        assert UPBIT_TO_CANONICAL["week"] == "1w"

    def test_binance_to_canonical_identity(self) -> None:
        """Test Binance intervals are identity mapping."""
        assert BINANCE_TO_CANONICAL["1d"] == "1d"
        assert BINANCE_TO_CANONICAL["4h"] == "4h"
        assert BINANCE_TO_CANONICAL["1m"] == "1m"

    def test_canonical_to_upbit_reverse(self) -> None:
        """Test reverse mapping from canonical to Upbit."""
        assert CANONICAL_TO_UPBIT["1d"] == "day"
        assert CANONICAL_TO_UPBIT["4h"] == "minute240"
        assert CANONICAL_TO_UPBIT["1m"] == "minute1"

    def test_canonical_to_binance_reverse(self) -> None:
        """Test reverse mapping from canonical to Binance."""
        assert CANONICAL_TO_BINANCE["1d"] == "1d"
        assert CANONICAL_TO_BINANCE["4h"] == "4h"

    def test_all_upbit_intervals_mapped(self) -> None:
        """Test that all Upbit intervals have a canonical mapping."""
        upbit_intervals = [
            "minute1",
            "minute3",
            "minute5",
            "minute10",
            "minute15",
            "minute30",
            "minute60",
            "minute240",
            "day",
            "week",
            "month",
        ]
        for interval in upbit_intervals:
            assert interval in UPBIT_TO_CANONICAL, f"{interval} not in UPBIT_TO_CANONICAL"

    def test_all_binance_intervals_mapped(self) -> None:
        """Test that all Binance intervals have a canonical mapping."""
        binance_intervals = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ]
        for interval in binance_intervals:
            assert interval in BINANCE_TO_CANONICAL, f"{interval} not in BINANCE_TO_CANONICAL"


class TestGetExchangeIntervals:
    """Test get_exchange_intervals function."""

    def test_upbit_intervals(self) -> None:
        """Test Upbit intervals are returned correctly."""
        intervals = get_exchange_intervals("upbit")
        codes = [code for code, _ in intervals]
        assert "day" in codes
        assert "minute240" in codes
        assert "week" in codes

    def test_binance_intervals(self) -> None:
        """Test Binance intervals are returned correctly."""
        intervals = get_exchange_intervals("binance")
        codes = [code for code, _ in intervals]
        assert "1d" in codes
        assert "4h" in codes
        assert "1w" in codes

    def test_unsupported_exchange(self) -> None:
        """Test unsupported exchange raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported exchange"):
            get_exchange_intervals("invalid")  # type: ignore[arg-type]

    def test_intervals_have_display_names(self) -> None:
        """Test all intervals have display names."""
        for exchange in ("upbit", "binance"):
            intervals = get_exchange_intervals(exchange)  # type: ignore[arg-type]
            for code, name in intervals:
                assert code, "Interval code should not be empty"
                assert name, "Interval name should not be empty"
