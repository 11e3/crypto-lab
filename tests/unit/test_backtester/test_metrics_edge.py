"""Tests for metrics calculation edge cases.

Tests portfolio risk metrics, trade metrics, and metrics_helpers edge cases.
"""

from __future__ import annotations

import datetime
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtester.metrics import calculate_metrics, calculate_trade_metrics
from src.backtester.metrics_helpers import (
    calculate_return_metrics,
    calculate_risk_metrics_from_equity,
    calculate_trade_stats,
)
from src.backtester.models import BacktestConfig, Trade

# ============================================================
# metrics.py — calculate_metrics portfolio risk branch
# ============================================================


class TestCalculateMetricsPortfolioRisk:
    """Test portfolio risk metrics calculation in calculate_metrics."""

    def test_portfolio_risk_metrics_populated(self) -> None:
        """Portfolio risk metrics should be calculated when asset_returns provided."""
        equity = np.linspace(10_000_000, 11_000_000, 100)
        dates = np.array([date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(100)])
        trades = [
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 1),
                entry_price=50000.0,
                exit_date=date(2024, 3, 1),
                exit_price=55000.0,
                amount=1.0,
                pnl=5000.0,
                pnl_pct=10.0,
            ),
        ]
        config = BacktestConfig(initial_capital=10_000_000)
        asset_returns = {
            "BTC": np.random.normal(0.001, 0.02, 100).tolist(),
            "ETH": np.random.normal(0.0005, 0.03, 100).tolist(),
        }

        result = calculate_metrics(
            equity_curve=equity,
            dates=dates,
            trades=trades,
            config=config,
            strategy_name="test",
            asset_returns=asset_returns,
        )

        assert result.risk_metrics is not None

    def test_portfolio_risk_empty_returns_filtered(self) -> None:
        """Empty returns list for a ticker should be filtered out."""
        equity = np.linspace(10_000_000, 11_000_000, 100)
        dates = np.array([date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(100)])
        trades = [
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 1),
                entry_price=50000.0,
                exit_date=date(2024, 3, 1),
                exit_price=55000.0,
                amount=1.0,
                pnl=5000.0,
                pnl_pct=10.0,
            ),
        ]
        config = BacktestConfig(initial_capital=10_000_000)
        asset_returns = {
            "BTC": np.random.normal(0.001, 0.02, 100).tolist(),
            "ETH": [],  # Empty — should be filtered
        }

        # Should not crash
        result = calculate_metrics(
            equity_curve=equity,
            dates=dates,
            trades=trades,
            config=config,
            asset_returns=asset_returns,
        )

        assert result is not None

    def test_portfolio_risk_exception_caught(self) -> None:
        """Exception in portfolio risk calculation should be caught."""
        equity = np.linspace(10_000_000, 11_000_000, 100)
        dates = np.array([date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(100)])
        trades = [
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 1),
                entry_price=50000.0,
                exit_date=date(2024, 3, 1),
                exit_price=55000.0,
                amount=1.0,
                pnl=5000.0,
                pnl_pct=10.0,
            ),
        ]
        config = BacktestConfig(initial_capital=10_000_000)
        asset_returns = {
            "BTC": [0.01] * 100,
            "ETH": [0.02] * 100,
        }

        with patch("src.backtester.metrics.calculate_portfolio_risk_metrics") as mock_calc:
            mock_calc.side_effect = ValueError("Numerical instability")
            result = calculate_metrics(
                equity_curve=equity,
                dates=dates,
                trades=trades,
                config=config,
                asset_returns=asset_returns,
            )

        # Should not crash, risk_metrics stays None
        assert result is not None
        assert result.risk_metrics is None

    def test_short_equity_curve_returns_defaults(self) -> None:
        """Equity curve with < 2 points returns default result."""
        equity = np.array([10_000_000.0])
        dates = np.array([date(2024, 1, 1)])
        config = BacktestConfig(initial_capital=10_000_000)

        result = calculate_metrics(
            equity_curve=equity,
            dates=dates,
            trades=[],
            config=config,
        )

        assert result.total_return == 0.0
        assert result.cagr == 0.0
        assert result.mdd == 0.0


# ============================================================
# metrics.py — calculate_trade_metrics
# ============================================================


class TestCalculateTradeMetrics:
    """Test calculate_trade_metrics function."""

    def test_empty_dataframe_returns_empty_dict(self) -> None:
        """Empty trades DataFrame returns empty dict."""
        df = pd.DataFrame()
        result = calculate_trade_metrics(df)
        assert result == {}

    def test_all_winning_trades(self) -> None:
        """All winning trades should not crash profit_factor calculation."""
        df = pd.DataFrame({
            "pnl": [100.0, 200.0, 50.0],
            "pnl_pct": [5.0, 10.0, 2.5],
        })

        result = calculate_trade_metrics(df)

        assert result["total_trades"] == 3.0
        assert result["winning_trades"] == 3.0
        assert result["losing_trades"] == 0.0
        assert result["win_rate"] == 100.0
        # profit_factor not set when no losing trades (only when both exist)
        assert "profit_factor" not in result or result.get("profit_factor", 0) >= 0

    def test_all_losing_trades(self) -> None:
        """All losing trades."""
        df = pd.DataFrame({
            "pnl": [-100.0, -200.0, -50.0],
            "pnl_pct": [-5.0, -10.0, -2.5],
        })

        result = calculate_trade_metrics(df)

        assert result["total_trades"] == 3.0
        assert result["winning_trades"] == 0.0
        assert result["losing_trades"] == 3.0
        assert result["win_rate"] == 0.0

    def test_mixed_trades(self) -> None:
        """Mixed winning and losing trades."""
        df = pd.DataFrame({
            "pnl": [100.0, -50.0, 200.0, -30.0],
            "pnl_pct": [5.0, -2.5, 10.0, -1.5],
        })

        result = calculate_trade_metrics(df)

        assert result["total_trades"] == 4.0
        assert result["winning_trades"] == 2.0
        assert result["losing_trades"] == 2.0
        assert result["profit_factor"] == pytest.approx(300.0 / 80.0)

    def test_zero_pnl_trades_counted_correctly(self) -> None:
        """Zero PnL trades are neither winning nor losing."""
        df = pd.DataFrame({
            "pnl": [0.0, 100.0, -50.0],
            "pnl_pct": [0.0, 5.0, -2.5],
        })

        result = calculate_trade_metrics(df)

        assert result["winning_trades"] == 1.0
        assert result["losing_trades"] == 1.0


# ============================================================
# metrics_helpers.py — edge cases
# ============================================================


class TestReturnMetricsEdgeCases:
    """Test calculate_return_metrics edge cases."""

    def test_single_point_equity_curve(self) -> None:
        """Single point equity curve returns zeros."""
        equity = np.array([10_000_000.0])
        dates = np.array([date(2024, 1, 1)])

        total_return, cagr = calculate_return_metrics(equity, dates, 10_000_000.0)

        assert total_return == 0.0
        assert cagr == 0.0

    def test_empty_equity_curve(self) -> None:
        """Empty equity curve returns zeros."""
        equity = np.array([])
        dates = np.array([])

        total_return, cagr = calculate_return_metrics(equity, dates, 10_000_000.0)

        assert total_return == 0.0
        assert cagr == 0.0


class TestRiskMetricsEdgeCases:
    """Test calculate_risk_metrics_from_equity edge cases."""

    def test_single_point_equity_curve(self) -> None:
        """Single point equity curve returns zeros."""
        equity = np.array([10_000_000.0])

        mdd, calmar, sharpe = calculate_risk_metrics_from_equity(equity, 10.0)

        assert mdd == 0.0
        assert calmar == 0.0
        assert sharpe == 0.0

    def test_empty_equity_curve(self) -> None:
        """Empty equity curve returns zeros."""
        equity = np.array([])

        mdd, calmar, sharpe = calculate_risk_metrics_from_equity(equity, 10.0)

        assert mdd == 0.0
        assert calmar == 0.0
        assert sharpe == 0.0


class TestTradeStatsEdgeCases:
    """Test calculate_trade_stats edge cases."""

    def test_empty_trades_returns_defaults(self) -> None:
        """Empty trade list returns default stats."""
        result = calculate_trade_stats([])

        assert result["total_trades"] == 0
        assert result["profit_factor"] == 0.0
        assert result["win_rate"] == 0.0

    def test_all_winning_trades_profit_factor(self) -> None:
        """All winning trades get sentinel profit factor."""
        trades = [
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 1),
                entry_price=50000.0,
                pnl=1000.0,
                pnl_pct=2.0,
            ),
            Trade(
                ticker="ETH",
                entry_date=date(2024, 1, 1),
                entry_price=3000.0,
                pnl=500.0,
                pnl_pct=3.0,
            ),
        ]

        result = calculate_trade_stats(trades)

        assert result["profit_factor"] == 999.99
        assert result["win_rate"] == 100.0

    def test_all_losing_trades_profit_factor(self) -> None:
        """All losing trades get zero profit factor."""
        trades = [
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 1),
                entry_price=50000.0,
                pnl=-1000.0,
                pnl_pct=-2.0,
            ),
            Trade(
                ticker="ETH",
                entry_date=date(2024, 1, 1),
                entry_price=3000.0,
                pnl=-500.0,
                pnl_pct=-3.0,
            ),
        ]

        result = calculate_trade_stats(trades)

        assert result["profit_factor"] == 0.0
        assert result["win_rate"] == 0.0

    def test_mixed_trades_profit_factor(self) -> None:
        """Mixed trades calculate correct profit factor."""
        trades = [
            Trade(ticker="BTC", entry_date=date(2024, 1, 1), entry_price=50000.0, pnl=300.0, pnl_pct=2.0),
            Trade(ticker="ETH", entry_date=date(2024, 1, 1), entry_price=3000.0, pnl=-100.0, pnl_pct=-1.0),
        ]

        result = calculate_trade_stats(trades)

        assert result["profit_factor"] == pytest.approx(3.0)
        assert result["win_rate"] == pytest.approx(50.0)
