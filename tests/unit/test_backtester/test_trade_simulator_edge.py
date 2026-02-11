"""Tests for trade simulator edge cases.

Tests NaN handling, equity correction, and price fallback logic.
"""

from __future__ import annotations

import numpy as np

from src.backtester.engine.trade_simulator import _get_current_price, _get_final_price


class TestGetCurrentPrice:
    """Test _get_current_price with missing/NaN data."""

    def test_valid_current_price(self) -> None:
        """Returns current close when valid."""

        class MockState:
            position_entry_prices = np.array([50000.0])

        closes = np.array([[50000.0, 51000.0]])
        valid_data = np.array([True])

        price = _get_current_price(
            MockState(), t_idx=0, d_idx=1, closes=closes, valid_data=valid_data
        )

        assert price == 51000.0

    def test_nan_current_uses_previous(self) -> None:
        """Falls back to previous day price when current is NaN."""

        class MockState:
            position_entry_prices = np.array([50000.0])

        closes = np.array([[50000.0, np.nan]])
        valid_data = np.array([True])

        price = _get_current_price(
            MockState(), t_idx=0, d_idx=1, closes=closes, valid_data=valid_data
        )

        assert price == 50000.0

    def test_nan_current_and_previous_uses_entry(self) -> None:
        """Falls back to entry price when current and previous are NaN."""

        class MockState:
            position_entry_prices = np.array([48000.0])

        closes = np.array([[np.nan, np.nan]])
        valid_data = np.array([True])

        price = _get_current_price(
            MockState(), t_idx=0, d_idx=1, closes=closes, valid_data=valid_data
        )

        assert price == 48000.0

    def test_invalid_data_flag_uses_previous_close(self) -> None:
        """When valid_data is False, falls back to previous day's close (not entry)."""

        class MockState:
            position_entry_prices = np.array([45000.0])

        closes = np.array([[50000.0, 51000.0]])
        valid_data = np.array([False])

        price = _get_current_price(
            MockState(), t_idx=0, d_idx=1, closes=closes, valid_data=valid_data
        )

        # valid_data=False skips current bar, but previous close (50000) is used
        assert price == 50000.0

    def test_invalid_data_no_previous_uses_entry(self) -> None:
        """When valid_data is False and no previous close, falls back to entry price."""

        class MockState:
            position_entry_prices = np.array([45000.0])

        closes = np.array([[np.nan, np.nan]])
        valid_data = np.array([False])

        price = _get_current_price(
            MockState(), t_idx=0, d_idx=1, closes=closes, valid_data=valid_data
        )

        assert price == 45000.0

    def test_d_idx_zero_with_nan(self) -> None:
        """At d_idx=0, no previous day available, uses entry price."""

        class MockState:
            position_entry_prices = np.array([42000.0])

        closes = np.array([[np.nan]])
        valid_data = np.array([True])

        price = _get_current_price(
            MockState(), t_idx=0, d_idx=0, closes=closes, valid_data=valid_data
        )

        assert price == 42000.0


class TestGetFinalPrice:
    """Test _get_final_price with missing/NaN data."""

    def test_valid_last_close(self) -> None:
        """Returns last valid close price."""
        closes = np.array([[50000.0, 51000.0, 52000.0]])

        price = _get_final_price(closes, t_idx=0, fallback_price=48000.0)

        assert price == 52000.0

    def test_nan_last_uses_previous(self) -> None:
        """Falls back to previous valid close when last is NaN."""
        closes = np.array([[50000.0, 51000.0, np.nan]])

        price = _get_final_price(closes, t_idx=0, fallback_price=48000.0)

        assert price == 51000.0

    def test_all_nan_uses_fallback(self) -> None:
        """Uses fallback price when all closes are NaN (delisted ticker)."""
        closes = np.array([[np.nan, np.nan, np.nan]])

        price = _get_final_price(closes, t_idx=0, fallback_price=48000.0)

        assert price == 48000.0

    def test_single_valid_price(self) -> None:
        """Returns the only valid price."""
        closes = np.array([[np.nan, 50000.0, np.nan]])

        price = _get_final_price(closes, t_idx=0, fallback_price=48000.0)

        assert price == 50000.0
