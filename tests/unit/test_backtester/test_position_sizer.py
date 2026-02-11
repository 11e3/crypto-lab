"""Tests for position sizer edge cases.

Tests Kelly criterion, MPT optimization failures, and fallback sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtester.engine.position_sizer import (
    _calculate_fallback_sizes,
    _calculate_kelly_sizes,
    _calculate_mpt_sizes,
    calculate_position_sizes_for_entries,
)
from src.backtester.models import BacktestConfig


@dataclass
class MockSimulationState:
    """Minimal mock for SimulationState."""

    cash: float = 10_000_000.0
    asset_returns: dict[str, list[float]] = field(default_factory=dict)
    trades_list: list[dict] = field(default_factory=list)
    position_amounts: np.ndarray = field(default_factory=lambda: np.zeros(2))
    position_entry_prices: np.ndarray = field(default_factory=lambda: np.zeros(2))
    equity_curve: np.ndarray = field(default_factory=lambda: np.zeros(100))


class TestCalculateKellySizes:
    """Test Kelly criterion position sizing edge cases."""

    def test_fewer_than_10_trades_returns_empty(self) -> None:
        """Kelly requires at least 10 trades."""
        state = MockSimulationState(
            trades_list=[{"pnl_pct": 0.05}] * 9,
        )
        config = BacktestConfig(max_kelly=0.25)

        result = _calculate_kelly_sizes(state, config)

        assert result == {}

    def test_exactly_10_trades_proceeds(self) -> None:
        """Kelly proceeds with exactly 10 trades."""
        trades = [{"pnl_pct": 0.05, "ticker": "BTC"}] * 10
        state = MockSimulationState(trades_list=trades)
        config = BacktestConfig(max_kelly=0.25)

        with patch("src.risk.portfolio_optimization.PortfolioOptimizer") as mock_opt:
            mock_opt.return_value.optimize_kelly_portfolio.return_value = {"BTC": 2_500_000.0}
            result = _calculate_kelly_sizes(state, config)

        assert result == {"BTC": 2_500_000.0}

    def test_missing_pnl_pct_column_returns_empty(self) -> None:
        """Kelly returns empty dict when pnl_pct column is missing."""
        trades = [{"ticker": "BTC", "some_other_col": 0.05}] * 15
        state = MockSimulationState(trades_list=trades)
        config = BacktestConfig(max_kelly=0.25)

        result = _calculate_kelly_sizes(state, config)

        assert result == {}

    def test_kelly_optimization_exception_returns_empty(self) -> None:
        """Kelly catches exceptions and returns empty dict."""
        trades = [{"pnl_pct": -0.10, "ticker": "BTC"}] * 15
        state = MockSimulationState(trades_list=trades)
        config = BacktestConfig(max_kelly=0.25)

        with patch("src.risk.portfolio_optimization.PortfolioOptimizer") as mock_opt:
            mock_opt.return_value.optimize_kelly_portfolio.side_effect = ValueError(
                "All trades losing"
            )
            result = _calculate_kelly_sizes(state, config)

        assert result == {}


class TestCalculateMptSizes:
    """Test MPT/risk parity position sizing edge cases."""

    def test_fewer_than_2_tickers_with_data_returns_empty(self) -> None:
        """MPT requires at least 2 tickers with sufficient returns."""
        state = MockSimulationState(
            asset_returns={"BTC": [0.01] * 30},
        )
        config = BacktestConfig(position_sizing_lookback=20)

        result = _calculate_mpt_sizes(
            state, config, candidate_idx=np.array([0]), tickers=["BTC"], optimization_method="mpt"
        )

        assert result == {}

    def test_insufficient_lookback_returns_empty(self) -> None:
        """MPT returns empty when returns shorter than lookback."""
        state = MockSimulationState(
            asset_returns={"BTC": [0.01] * 5, "ETH": [0.02] * 5},
        )
        config = BacktestConfig(position_sizing_lookback=20)

        result = _calculate_mpt_sizes(
            state,
            config,
            candidate_idx=np.array([0, 1]),
            tickers=["BTC", "ETH"],
            optimization_method="mpt",
        )

        assert result == {}

    def test_optimization_exception_returns_empty(self) -> None:
        """MPT catches optimization exceptions gracefully."""
        state = MockSimulationState(
            asset_returns={"BTC": [0.01] * 30, "ETH": [0.02] * 30},
        )
        config = BacktestConfig(position_sizing_lookback=20)

        with patch("src.backtester.engine.position_sizer.optimize_portfolio") as mock_opt:
            mock_opt.side_effect = np.linalg.LinAlgError("Singular matrix")
            result = _calculate_mpt_sizes(
                state,
                config,
                candidate_idx=np.array([0, 1]),
                tickers=["BTC", "ETH"],
                optimization_method="mpt",
            )

        assert result == {}


class TestCalculateFallbackSizes:
    """Test fallback position sizing edge cases."""

    def test_d_idx_zero_uses_at_least_one_row(self) -> None:
        """At d_idx=0, uses iloc[:1] instead of empty slice."""
        state = MockSimulationState()
        config = BacktestConfig()

        hist_df = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        ticker_historical_data = {"BTC": hist_df}
        entry_prices = np.array([[100.0, 101.0]])

        with patch(
            "src.backtester.engine.position_sizer.calculate_multi_asset_position_sizes"
        ) as mock_calc:
            mock_calc.return_value = {"BTC": 5_000_000.0}
            _calculate_fallback_sizes(
                state,
                config,
                candidate_idx=np.array([0]),
                tickers=["BTC"],
                entry_prices=entry_prices,
                d_idx=0,
                ticker_historical_data=ticker_historical_data,
            )

        # Verify historical data was passed with at least 1 row
        call_args = mock_calc.call_args
        passed_hist = call_args.kwargs.get("historical_data") or call_args[1].get("historical_data")
        if passed_hist is None:
            # Check positional args
            passed_hist = call_args[0][4] if len(call_args[0]) > 4 else {}
        assert "BTC" in passed_hist
        assert len(passed_hist["BTC"]) >= 1


class TestCalculatePositionSizesForEntries:
    """Test the main entry point function."""

    def test_equal_sizing_returns_empty(self) -> None:
        """Equal sizing returns empty dict (handled by caller)."""
        state = MockSimulationState()
        config = BacktestConfig(position_sizing="equal")

        result = calculate_position_sizes_for_entries(
            state=state,
            config=config,
            candidate_idx=np.array([0, 1]),
            tickers=["BTC", "ETH"],
            entry_prices=np.array([[100.0], [200.0]]),
            d_idx=0,
            current_date=date(2024, 1, 1),
            ticker_historical_data={},
        )

        assert result == {}

    def test_single_candidate_returns_empty(self) -> None:
        """Single candidate returns empty (equal sizing used)."""
        state = MockSimulationState()
        config = BacktestConfig(position_sizing="mpt")

        result = calculate_position_sizes_for_entries(
            state=state,
            config=config,
            candidate_idx=np.array([0]),
            tickers=["BTC"],
            entry_prices=np.array([[100.0]]),
            d_idx=0,
            current_date=date(2024, 1, 1),
            ticker_historical_data={},
        )

        assert result == {}
