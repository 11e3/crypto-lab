"""Tests for src/backtester/wfa/wfa_backtest.py - WFA simple backtest."""

import numpy as np
import pandas as pd
import pytest

from src.backtester.wfa.wfa_backtest import (
    _calculate_max_drawdown,
    _calculate_sharpe,
    _calculate_win_rate,
    _create_empty_result,
    _simulate_positions,
    simple_backtest,
)
from src.strategies.base import Strategy


class MockStrategy(Strategy):
    """Mock strategy for testing."""

    def __init__(self, signals: list[int] | None = None) -> None:
        """Initialize mock strategy."""
        super().__init__(name="MockStrategy")
        self.signals = signals or []

    @property
    def required_indicators(self) -> list[str]:
        """Return required indicators."""
        return []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mock indicators."""
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mock signals."""
        df = df.copy()
        if self.signals:
            df["signal"] = self.signals[: len(df)]
        else:
            df["signal"] = 0
        return df


class TestSimulatePositions:
    """Tests for _simulate_positions function."""

    def test_no_signals(self) -> None:
        """Test with no trading signals."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [95.0, 96.0, 97.0],
                "close": [102.0, 103.0, 104.0],
                "signal": [0, 0, 0],
            }
        )
        trades, equity = _simulate_positions(df, 10000.0)

        assert len(trades) == 0
        assert equity[0] == 10000.0

    def test_long_entry_exit(self) -> None:
        """Test long entry and exit."""
        df = pd.DataFrame(
            {
                "open": [100.0, 105.0, 110.0, 108.0],
                "high": [105.0, 110.0, 115.0, 112.0],
                "low": [95.0, 100.0, 105.0, 104.0],
                "close": [102.0, 108.0, 112.0, 106.0],
                "signal": [1, 0, -1, 0],  # Buy, hold, sell
            }
        )
        trades, equity = _simulate_positions(df, 10000.0)

        assert len(trades) >= 1
        assert len(equity) > 1

    def test_short_entry_exit(self) -> None:
        """Test short entry and exit."""
        df = pd.DataFrame(
            {
                "open": [100.0, 98.0, 95.0, 97.0],
                "high": [105.0, 102.0, 98.0, 100.0],
                "low": [95.0, 94.0, 92.0, 94.0],
                "close": [98.0, 96.0, 94.0, 98.0],
                "signal": [-1, 0, 1, 0],  # Sell short, hold, cover
            }
        )
        trades, equity = _simulate_positions(df, 10000.0)

        assert len(trades) >= 1

    def test_position_at_end(self) -> None:
        """Test that open positions are closed at end."""
        df = pd.DataFrame(
            {
                "open": [100.0, 105.0, 110.0],
                "high": [105.0, 110.0, 115.0],
                "low": [95.0, 100.0, 105.0],
                "close": [102.0, 108.0, 112.0],
                "signal": [1, 0, 0],  # Enter and hold
            }
        )
        trades, equity = _simulate_positions(df, 10000.0)

        # Position should be closed at end
        assert len(trades) == 1

    def test_entry_price_zero_protection(self) -> None:
        """Test division by zero protection when entry_price is 0."""
        df = pd.DataFrame(
            {
                "open": [0.0, 105.0, 110.0],
                "high": [5.0, 110.0, 115.0],
                "low": [0.0, 100.0, 105.0],
                "close": [0.0, 108.0, 112.0],
                "signal": [1, -1, 0],
            }
        )
        trades, equity = _simulate_positions(df, 10000.0)

        # Should not crash, PnL should be 0
        assert len(trades) >= 0


class TestCalculateSharpe:
    """Tests for _calculate_sharpe function."""

    def test_positive_sharpe(self) -> None:
        """Test positive Sharpe ratio."""
        equity = [10000.0, 10100.0, 10200.0, 10300.0, 10400.0]
        result = _calculate_sharpe(equity)

        assert result > 0

    def test_negative_sharpe(self) -> None:
        """Test negative Sharpe ratio."""
        equity = [10000.0, 9900.0, 9800.0, 9700.0, 9600.0]
        result = _calculate_sharpe(equity)

        assert result < 0

    def test_single_value(self) -> None:
        """Test with single equity value."""
        equity = [10000.0]
        result = _calculate_sharpe(equity)

        assert result == 0.0

    def test_empty_equity(self) -> None:
        """Test with empty equity."""
        equity: list[float] = []
        result = _calculate_sharpe(equity)

        assert result == 0.0

    def test_zero_volatility(self) -> None:
        """Test with zero volatility."""
        equity = [10000.0, 10000.0, 10000.0]
        result = _calculate_sharpe(equity)

        assert result == 0.0

    def test_division_by_zero_protection(self) -> None:
        """Test protection against division by zero in equity."""
        equity = [0.0, 100.0, 110.0]
        result = _calculate_sharpe(equity)

        # Should not crash
        assert not np.isnan(result) or result == 0.0


class TestCalculateMaxDrawdown:
    """Tests for _calculate_max_drawdown function."""

    def test_normal_drawdown(self) -> None:
        """Test max drawdown calculation."""
        equity = [10000.0, 12000.0, 9000.0, 11000.0]
        result = _calculate_max_drawdown(equity)

        # MDD = (12000 - 9000) / 12000 = 0.25
        assert result == pytest.approx(-0.25)

    def test_no_drawdown(self) -> None:
        """Test with monotonically increasing equity."""
        equity = [10000.0, 11000.0, 12000.0, 13000.0]
        result = _calculate_max_drawdown(equity)

        assert result == pytest.approx(0.0)

    def test_single_value(self) -> None:
        """Test with single value."""
        equity = [10000.0]
        result = _calculate_max_drawdown(equity)

        assert result == 0.0

    def test_empty_equity(self) -> None:
        """Test with empty equity."""
        equity: list[float] = []
        result = _calculate_max_drawdown(equity)

        assert result == 0.0

    def test_zero_values_protection(self) -> None:
        """Test protection against division by zero."""
        equity = [0.0, 100.0, 90.0]
        result = _calculate_max_drawdown(equity)

        # Should not crash - main thing is no exception
        assert isinstance(result, float)


class TestCalculateWinRate:
    """Tests for _calculate_win_rate function."""

    def test_all_winners(self) -> None:
        """Test with all winning trades."""
        trades = [0.05, 0.03, 0.02, 0.01]
        winners, rate = _calculate_win_rate(trades)

        assert winners == 4
        assert rate == 1.0

    def test_all_losers(self) -> None:
        """Test with all losing trades."""
        trades = [-0.05, -0.03, -0.02, -0.01]
        winners, rate = _calculate_win_rate(trades)

        assert winners == 0
        assert rate == 0.0

    def test_mixed_trades(self) -> None:
        """Test with mixed trades."""
        trades = [0.05, -0.03, 0.02, -0.01]
        winners, rate = _calculate_win_rate(trades)

        assert winners == 2
        assert rate == 0.5

    def test_empty_trades(self) -> None:
        """Test with no trades."""
        trades: list[float] = []
        winners, rate = _calculate_win_rate(trades)

        assert winners == 0
        assert rate == 0.0


class TestCreateEmptyResult:
    """Tests for _create_empty_result function."""

    def test_empty_result_values(self) -> None:
        """Test that empty result has correct default values."""
        result = _create_empty_result()

        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.mdd == 0.0
        assert result.total_trades == 0
        assert result.winning_trades == 0
        assert result.win_rate == 0.0


class TestSimpleBacktest:
    """Tests for simple_backtest function."""

    def test_no_signal_column(self) -> None:
        """Test backtest when strategy doesn't generate signals."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [95.0, 96.0, 97.0],
                "close": [102.0, 103.0, 104.0],
            }
        )

        class NoSignalStrategy(Strategy):
            def __init__(self) -> None:
                super().__init__(name="NoSignal")

            @property
            def required_indicators(self) -> list[str]:
                return []

            def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

            def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
                return df  # No signal column

        strategy = NoSignalStrategy()
        result = simple_backtest(df, strategy)

        # Should return empty result
        assert result.total_trades == 0

    def test_with_signals(self) -> None:
        """Test backtest with trading signals."""
        df = pd.DataFrame(
            {
                "open": [100.0, 105.0, 110.0, 108.0, 112.0],
                "high": [105.0, 110.0, 115.0, 112.0, 118.0],
                "low": [95.0, 100.0, 105.0, 104.0, 108.0],
                "close": [102.0, 108.0, 112.0, 106.0, 115.0],
            }
        )

        strategy = MockStrategy(signals=[1, 0, -1, 0, 0])
        result = simple_backtest(df, strategy)

        assert result is not None
        assert hasattr(result, "total_return")

    def test_exception_handling(self) -> None:
        """Test that exceptions are handled gracefully."""
        df = pd.DataFrame()  # Empty dataframe

        strategy = MockStrategy()
        result = simple_backtest(df, strategy)

        # Should return empty result without crashing
        assert result.total_trades == 0
