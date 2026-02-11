"""Tests for WFA simple backtest runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtester.models import BacktestResult
from src.backtester.wfa.wfa_backtest import (
    _calculate_metrics,
    _calculate_win_rate,
    _create_empty_result,
    _simulate_positions,
    simple_backtest,
)


# =========================================================================
# _calculate_win_rate
# =========================================================================


class TestCalculateWinRate:
    """Tests for _calculate_win_rate."""

    def test_empty_trades(self) -> None:
        wins, rate = _calculate_win_rate([])
        assert wins == 0
        assert rate == 0.0

    def test_all_winning(self) -> None:
        wins, rate = _calculate_win_rate([0.1, 0.05, 0.2])
        assert wins == 3
        assert rate == 1.0

    def test_mixed_trades(self) -> None:
        wins, rate = _calculate_win_rate([0.1, -0.05, 0.2, -0.1])
        assert wins == 2
        assert rate == 0.5

    def test_all_losing(self) -> None:
        wins, rate = _calculate_win_rate([-0.1, -0.05])
        assert wins == 0
        assert rate == 0.0


# =========================================================================
# _create_empty_result
# =========================================================================


class TestCreateEmptyResult:
    """Tests for _create_empty_result."""

    def test_returns_backtest_result(self) -> None:
        result = _create_empty_result()
        assert isinstance(result, BacktestResult)

    def test_all_zeros(self) -> None:
        result = _create_empty_result()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.mdd == 0.0
        assert result.total_trades == 0
        assert result.winning_trades == 0
        assert result.win_rate == 0.0


# =========================================================================
# _simulate_positions
# =========================================================================


class TestSimulatePositions:
    """Tests for _simulate_positions."""

    def test_no_signals(self) -> None:
        df = pd.DataFrame({"close": [100, 101, 102], "signal": [0, 0, 0]})
        trades, equity = _simulate_positions(df, 10000.0)
        assert trades == []
        assert equity[0] == 10000.0

    def test_single_long_trade(self) -> None:
        df = pd.DataFrame({
            "close": [100.0, 105.0, 110.0],
            "signal": [1, 0, -1],
        })
        trades, equity = _simulate_positions(df, 10000.0)
        # Entry at 100, exit at 110 with reversed signal
        assert len(trades) >= 1

    def test_open_position_closed_at_end(self) -> None:
        df = pd.DataFrame({
            "close": [100.0, 105.0, 110.0],
            "signal": [1, 0, 0],
        })
        trades, equity = _simulate_positions(df, 10000.0)
        # Position should be closed at last bar
        assert len(trades) == 1
        assert trades[0] == pytest.approx(0.1, abs=0.001)  # (110-100)/100


# =========================================================================
# _calculate_metrics
# =========================================================================


class TestCalculateMetrics:
    """Tests for _calculate_metrics."""

    def test_basic_metrics(self) -> None:
        trades = [0.1, -0.05, 0.08]
        equity = [10000, 11000, 10450, 11286]
        result = _calculate_metrics(trades, equity, 10000.0)

        assert isinstance(result, BacktestResult)
        assert result.total_trades == 3
        assert result.winning_trades == 2
        assert result.total_return == pytest.approx(0.1286, abs=0.001)

    def test_no_trades(self) -> None:
        result = _calculate_metrics([], [10000], 10000.0)
        assert result.total_trades == 0
        assert result.total_return == 0.0


# =========================================================================
# simple_backtest (integration)
# =========================================================================


class TestSimpleBacktest:
    """Tests for simple_backtest function."""

    def test_with_mock_strategy(self) -> None:
        """Test simple_backtest with a mock strategy."""
        data = pd.DataFrame({
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [105.0, 106.0, 107.0, 108.0],
            "low": [95.0, 96.0, 97.0, 98.0],
            "close": [101.0, 102.0, 103.0, 104.0],
            "volume": [1000, 1100, 1200, 1300],
        })

        # Create mock strategy that adds signals
        strategy = MagicMock()
        strategy.calculate_indicators.return_value = data.copy()

        signaled = data.copy()
        signaled["signal"] = [1, 0, 0, -1]
        strategy.generate_signals.return_value = signaled

        result = simple_backtest(data, strategy)
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 1

    def test_no_signal_column(self) -> None:
        """Returns empty result when strategy produces no signals."""
        data = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [101.0],
            "volume": [1000],
        })

        strategy = MagicMock()
        strategy.calculate_indicators.return_value = data.copy()
        strategy.generate_signals.return_value = data.copy()  # No 'signal' column

        result = simple_backtest(data, strategy)
        assert result.total_trades == 0

    def test_strategy_error_returns_empty(self) -> None:
        """Returns empty result when strategy raises error."""
        data = pd.DataFrame({"close": [100]})
        strategy = MagicMock()
        strategy.calculate_indicators.side_effect = ValueError("bad data")

        result = simple_backtest(data, strategy)
        assert result.total_trades == 0
        assert result.total_return == 0.0
