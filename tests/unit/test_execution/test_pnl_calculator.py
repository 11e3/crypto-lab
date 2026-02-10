"""Tests for PnL calculator edge cases.

Tests missing price data, zero/negative prices, and total PnL calculation.
"""

from __future__ import annotations

import pytest

from src.execution.pnl_calculator import PnLCalculator
from src.execution.position import Position


class TestCalculatePnl:
    """Test individual position PnL calculation."""

    def test_normal_profit(self) -> None:
        """Positive PnL from price increase."""
        pos = Position(ticker="KRW-BTC", entry_price=50000.0, amount=1.0)
        pnl = PnLCalculator.calculate_pnl(pos, 55000.0)
        assert pnl == pytest.approx(5000.0)

    def test_normal_loss(self) -> None:
        """Negative PnL from price decrease."""
        pos = Position(ticker="KRW-BTC", entry_price=50000.0, amount=1.0)
        pnl = PnLCalculator.calculate_pnl(pos, 45000.0)
        assert pnl == pytest.approx(-5000.0)

    def test_zero_current_price_returns_zero(self) -> None:
        """Zero current price returns 0 PnL (not negative entry_price * amount)."""
        pos = Position(ticker="KRW-BTC", entry_price=50000.0, amount=1.0)
        pnl = PnLCalculator.calculate_pnl(pos, 0.0)
        assert pnl == 0.0

    def test_negative_current_price_returns_zero(self) -> None:
        """Negative current price returns 0 PnL."""
        pos = Position(ticker="KRW-BTC", entry_price=50000.0, amount=1.0)
        pnl = PnLCalculator.calculate_pnl(pos, -1000.0)
        assert pnl == 0.0


class TestCalculatePnlPct:
    """Test PnL percentage calculation."""

    def test_normal_percentage(self) -> None:
        """10% gain calculated correctly."""
        pos = Position(ticker="KRW-BTC", entry_price=50000.0, amount=1.0)
        pct = PnLCalculator.calculate_pnl_pct(pos, 55000.0)
        assert pct == pytest.approx(10.0)

    def test_zero_current_price_returns_zero(self) -> None:
        """Zero current price returns 0%."""
        pos = Position(ticker="KRW-BTC", entry_price=50000.0, amount=1.0)
        pct = PnLCalculator.calculate_pnl_pct(pos, 0.0)
        assert pct == 0.0

    def test_zero_entry_price_returns_zero(self) -> None:
        """Zero entry price returns 0% (avoids division by zero)."""
        pos = Position(ticker="KRW-BTC", entry_price=0.0, amount=1.0)
        pct = PnLCalculator.calculate_pnl_pct(pos, 55000.0)
        assert pct == 0.0


class TestCalculateTotalPnl:
    """Test total PnL across multiple positions."""

    def test_normal_total(self) -> None:
        """Total PnL across two positions."""
        positions = {
            "KRW-BTC": Position(ticker="KRW-BTC", entry_price=50000.0, amount=1.0),
            "KRW-ETH": Position(ticker="KRW-ETH", entry_price=3000.0, amount=10.0),
        }
        prices = {"KRW-BTC": 55000.0, "KRW-ETH": 3200.0}

        total = PnLCalculator.calculate_total_pnl(positions, prices)

        # BTC: (55000-50000)*1 = 5000, ETH: (3200-3000)*10 = 2000
        assert total == pytest.approx(7000.0)

    def test_missing_ticker_in_prices(self) -> None:
        """Missing ticker uses default price of 0.0 (returns 0 PnL for that position)."""
        positions = {
            "KRW-BTC": Position(ticker="KRW-BTC", entry_price=50000.0, amount=1.0),
            "KRW-DELISTED": Position(ticker="KRW-DELISTED", entry_price=1000.0, amount=100.0),
        }
        prices = {"KRW-BTC": 55000.0}  # Missing KRW-DELISTED

        total = PnLCalculator.calculate_total_pnl(positions, prices)

        # BTC: 5000, DELISTED: price=0.0 → PnL=0.0 (price <= 0 guard)
        assert total == pytest.approx(5000.0)

    def test_empty_positions(self) -> None:
        """No positions returns 0 total PnL."""
        total = PnLCalculator.calculate_total_pnl({}, {"KRW-BTC": 55000.0})
        assert total == 0.0

    def test_empty_prices(self) -> None:
        """No prices, all positions use default 0.0 price."""
        positions = {
            "KRW-BTC": Position(ticker="KRW-BTC", entry_price=50000.0, amount=1.0),
        }
        total = PnLCalculator.calculate_total_pnl(positions, {})
        # Price defaults to 0.0, which triggers the <= 0 guard → PnL = 0.0
        assert total == 0.0
