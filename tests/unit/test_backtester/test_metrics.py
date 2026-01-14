"""Tests for src/backtester/metrics.py - backtest metrics calculation."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtester.metrics import calculate_metrics, calculate_trade_metrics
from src.backtester.models import BacktestConfig, Trade


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_basic_metrics(self) -> None:
        """Test basic metrics calculation."""
        equity_curve = np.array([10000.0, 10500.0, 11000.0, 10800.0, 11500.0])
        dates = np.array([date(2024, 1, i + 1) for i in range(5)])
        trades: list[Trade] = []
        config = BacktestConfig(initial_capital=10000.0)

        result = calculate_metrics(equity_curve, dates, trades, config)

        assert result.equity_curve is not None
        assert result.total_return > 0
        assert result.mdd >= 0

    def test_with_trades(self) -> None:
        """Test metrics with trades."""
        equity_curve = np.array([10000.0, 10500.0, 11000.0, 10800.0, 11500.0])
        dates = np.array([date(2024, 1, i + 1) for i in range(5)])
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
                exit_price=105.0,
                amount=1.0,
                pnl=-5.0,
                pnl_pct=-4.5,
                is_whipsaw=False,
            ),
        ]
        config = BacktestConfig(initial_capital=10000.0)

        result = calculate_metrics(equity_curve, dates, trades, config)

        assert result.total_trades == 2
        assert result.winning_trades == 1
        assert result.losing_trades == 1
        assert result.win_rate == pytest.approx(50.0)

    def test_short_equity_curve(self) -> None:
        """Test with equity curve shorter than 2 elements."""
        equity_curve = np.array([10000.0])
        dates = np.array([date(2024, 1, 1)])
        trades: list[Trade] = []
        config = BacktestConfig(initial_capital=10000.0)

        result = calculate_metrics(equity_curve, dates, trades, config)

        # Should return result without metrics calculated
        assert result.equity_curve is not None

    def test_with_strategy_name(self) -> None:
        """Test with strategy name."""
        equity_curve = np.array([10000.0, 11000.0])
        dates = np.array([date(2024, 1, 1), date(2024, 1, 2)])
        trades: list[Trade] = []
        config = BacktestConfig(initial_capital=10000.0)

        result = calculate_metrics(
            equity_curve, dates, trades, config, strategy_name="TestStrategy"
        )

        assert result.strategy_name == "TestStrategy"

    def test_with_asset_returns(self) -> None:
        """Test with asset returns for portfolio metrics."""
        equity_curve = np.array([10000.0, 10500.0, 11000.0, 10800.0, 11500.0])
        dates = np.array([date(2024, 1, i + 1) for i in range(5)])
        trades = [
            Trade(
                ticker="BTC",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                exit_date=date(2024, 1, 5),
                exit_price=115.0,
                amount=1.0,
                pnl=15.0,
                pnl_pct=15.0,
                is_whipsaw=False,
            ),
        ]
        config = BacktestConfig(initial_capital=10000.0)
        asset_returns = {"BTC": [0.05, 0.048, -0.018, 0.065]}

        result = calculate_metrics(
            equity_curve, dates, trades, config, asset_returns=asset_returns
        )

        assert result.total_trades == 1


class TestCalculateTradeMetrics:
    """Tests for calculate_trade_metrics function."""

    def test_empty_dataframe(self) -> None:
        """Test with empty trades DataFrame."""
        trades_df = pd.DataFrame()
        result = calculate_trade_metrics(trades_df)

        assert result == {}

    def test_all_winning_trades(self) -> None:
        """Test with all winning trades."""
        trades_df = pd.DataFrame(
            {
                "pnl": [100.0, 200.0, 150.0],
                "pnl_pct": [10.0, 20.0, 15.0],
            }
        )
        result = calculate_trade_metrics(trades_df)

        assert result["total_trades"] == 3
        assert result["winning_trades"] == 3
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 100.0
        assert result["avg_pnl"] == pytest.approx(150.0)

    def test_all_losing_trades(self) -> None:
        """Test with all losing trades."""
        trades_df = pd.DataFrame(
            {
                "pnl": [-100.0, -200.0, -150.0],
                "pnl_pct": [-10.0, -20.0, -15.0],
            }
        )
        result = calculate_trade_metrics(trades_df)

        assert result["total_trades"] == 3
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 3
        assert result["win_rate"] == 0.0

    def test_mixed_trades(self) -> None:
        """Test with mixed winning and losing trades."""
        trades_df = pd.DataFrame(
            {
                "pnl": [100.0, -50.0, 200.0, -30.0],
                "pnl_pct": [10.0, -5.0, 20.0, -3.0],
            }
        )
        result = calculate_trade_metrics(trades_df)

        assert result["total_trades"] == 4
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 2
        assert result["win_rate"] == 50.0
        assert "profit_factor" in result
        assert result["profit_factor"] == pytest.approx(300.0 / 80.0)

    def test_max_metrics(self) -> None:
        """Test max win and loss metrics."""
        trades_df = pd.DataFrame(
            {
                "pnl": [100.0, -50.0, 200.0, -30.0],
                "pnl_pct": [10.0, -5.0, 20.0, -3.0],
            }
        )
        result = calculate_trade_metrics(trades_df)

        assert result["max_win"] == 200.0
        assert result["max_loss"] == -50.0
        assert result["max_win_pct"] == 20.0
        assert result["max_loss_pct"] == -5.0

    def test_average_metrics(self) -> None:
        """Test average win and loss metrics."""
        trades_df = pd.DataFrame(
            {
                "pnl": [100.0, -50.0, 200.0, -30.0],
                "pnl_pct": [10.0, -5.0, 20.0, -3.0],
            }
        )
        result = calculate_trade_metrics(trades_df)

        assert result["avg_win"] == pytest.approx(150.0)
        assert result["avg_loss"] == pytest.approx(-40.0)
        assert result["avg_win_pct"] == pytest.approx(15.0)
        assert result["avg_loss_pct"] == pytest.approx(-4.0)
