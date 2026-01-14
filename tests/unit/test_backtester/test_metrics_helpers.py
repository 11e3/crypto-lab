"""Tests for src/backtester/metrics_helpers.py - metrics calculation helpers."""

from datetime import date, timedelta

import numpy as np
import pytest

from src.backtester.metrics_helpers import (
    calculate_return_metrics,
    calculate_risk_metrics_from_equity,
    calculate_trade_stats,
)
from src.backtester.models import Trade


class TestCalculateReturnMetrics:
    """Tests for calculate_return_metrics function."""

    def test_basic_return_metrics(self) -> None:
        """Test basic return metrics calculation."""
        equity_curve = np.array([10000.0, 10500.0, 11000.0, 11500.0, 12000.0])
        base_date = date(2024, 1, 1)
        dates = np.array([base_date + timedelta(days=i) for i in range(5)])

        total_return, cagr = calculate_return_metrics(equity_curve, dates, 10000.0)

        assert total_return == pytest.approx(20.0)  # 20% return
        assert cagr > 0

    def test_negative_return(self) -> None:
        """Test with negative return."""
        equity_curve = np.array([10000.0, 9500.0, 9000.0, 8500.0, 8000.0])
        base_date = date(2024, 1, 1)
        dates = np.array([base_date + timedelta(days=i) for i in range(5)])

        total_return, cagr = calculate_return_metrics(equity_curve, dates, 10000.0)

        assert total_return == pytest.approx(-20.0)  # -20% return
        assert cagr < 0

    def test_short_equity_curve(self) -> None:
        """Test with equity curve shorter than 2 elements."""
        equity_curve = np.array([10000.0])
        dates = np.array([date(2024, 1, 1)])

        total_return, cagr = calculate_return_metrics(equity_curve, dates, 10000.0)

        assert total_return == 0.0
        assert cagr == 0.0


class TestCalculateRiskMetricsFromEquity:
    """Tests for calculate_risk_metrics_from_equity function."""

    def test_basic_risk_metrics(self) -> None:
        """Test basic risk metrics calculation."""
        equity_curve = np.array([10000.0, 12000.0, 9000.0, 11000.0, 13000.0])

        mdd, calmar, sharpe = calculate_risk_metrics_from_equity(equity_curve, 30.0)

        assert mdd > 0  # Should have drawdown
        assert isinstance(calmar, float)
        assert isinstance(sharpe, float)

    def test_no_drawdown(self) -> None:
        """Test with monotonically increasing equity."""
        equity_curve = np.array([10000.0, 11000.0, 12000.0, 13000.0])

        mdd, calmar, sharpe = calculate_risk_metrics_from_equity(equity_curve, 30.0)

        assert mdd == pytest.approx(0.0)

    def test_short_equity_curve(self) -> None:
        """Test with short equity curve."""
        equity_curve = np.array([10000.0])

        mdd, calmar, sharpe = calculate_risk_metrics_from_equity(equity_curve, 0.0)

        assert mdd == 0.0
        assert calmar == 0.0
        assert sharpe == 0.0


class TestCalculateTradeStats:
    """Tests for calculate_trade_stats function."""

    def test_empty_trades(self) -> None:
        """Test with no trades."""
        trades: list[Trade] = []
        result = calculate_trade_stats(trades)

        assert result["total_trades"] == 0
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0
        assert result["avg_trade_return"] == 0.0

    def test_all_winning_trades(self) -> None:
        """Test with all winning trades."""
        trades = [
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                exit_date=date(2024, 1, 2),
                exit_price=110.0,
                amount=1.0,
                pnl=10.0,
                pnl_pct=10.0,
                is_whipsaw=False,
            ),
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 3),
                entry_price=110.0,
                exit_date=date(2024, 1, 4),
                exit_price=120.0,
                amount=1.0,
                pnl=10.0,
                pnl_pct=9.1,
                is_whipsaw=False,
            ),
        ]
        result = calculate_trade_stats(trades)

        assert result["total_trades"] == 2
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 100.0
        assert result["profit_factor"] == 0.0  # No losses

    def test_mixed_trades(self) -> None:
        """Test with mixed winning and losing trades."""
        trades = [
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                exit_date=date(2024, 1, 2),
                exit_price=110.0,
                amount=1.0,
                pnl=100.0,
                pnl_pct=10.0,
                is_whipsaw=False,
            ),
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 3),
                entry_price=110.0,
                exit_date=date(2024, 1, 4),
                exit_price=100.0,
                amount=1.0,
                pnl=-50.0,
                pnl_pct=-9.1,
                is_whipsaw=False,
            ),
        ]
        result = calculate_trade_stats(trades)

        assert result["total_trades"] == 2
        assert result["winning_trades"] == 1
        assert result["losing_trades"] == 1
        assert result["win_rate"] == 50.0
        assert result["profit_factor"] == pytest.approx(2.0)  # 100/50

    def test_all_losing_trades(self) -> None:
        """Test with all losing trades."""
        trades = [
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                exit_date=date(2024, 1, 2),
                exit_price=90.0,
                amount=1.0,
                pnl=-10.0,
                pnl_pct=-10.0,
                is_whipsaw=False,
            ),
        ]
        result = calculate_trade_stats(trades)

        assert result["total_trades"] == 1
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 1
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0  # No winning trades
