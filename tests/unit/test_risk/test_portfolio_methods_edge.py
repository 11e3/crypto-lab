"""Tests for portfolio methods edge cases.

Tests zero volatility, Kelly over-allocation, and all-winning scenarios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.portfolio_methods import (
    calculate_kelly_criterion,
    optimize_kelly_portfolio,
    optimize_mpt,
)


class TestOptimizeMptEdgeCases:
    """Test MPT optimizer edge cases."""

    def test_zero_volatility_assets(self) -> None:
        """All-zero returns (flat prices) should not crash."""
        returns_df = pd.DataFrame(
            {
                "BTC": [0.0] * 30,
                "ETH": [0.0] * 30,
            }
        )

        result = optimize_mpt(returns_df, risk_free_rate=0.0)

        # Should return valid weights (likely equal weight or handle inf)
        assert result is not None
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_perfectly_correlated_assets(self) -> None:
        """Perfectly correlated assets (ill-conditioned cov matrix)."""
        base = np.random.normal(0.001, 0.02, 50)
        returns_df = pd.DataFrame(
            {
                "BTC": base,
                "ETH": base,  # Perfect correlation
            }
        )

        result = optimize_mpt(returns_df, risk_free_rate=0.0)

        # Should fall back to equal weights due to ill-conditioned matrix
        assert result is not None
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_single_asset_returns_full_weight(self) -> None:
        """Single asset should get 100% weight."""
        returns_df = pd.DataFrame(
            {
                "BTC": np.random.normal(0.001, 0.02, 30),
            }
        )

        result = optimize_mpt(returns_df, risk_free_rate=0.0)

        assert result is not None
        assert result.weights["BTC"] == pytest.approx(1.0, abs=0.01)


class TestCalculateKellyCriterionEdgeCases:
    """Test Kelly criterion edge cases."""

    def test_zero_avg_win_returns_zero(self) -> None:
        """Zero average win returns 0 allocation."""
        result = calculate_kelly_criterion(win_rate=0.6, avg_win=0.0, avg_loss=0.05, max_kelly=0.25)
        assert result == 0.0

    def test_negative_kelly_returns_zero(self) -> None:
        """When Kelly formula is negative (poor strategy), returns 0."""
        # Very low win rate with poor payoff ratio → negative Kelly
        result = calculate_kelly_criterion(
            win_rate=0.2, avg_win=0.02, avg_loss=0.05, max_kelly=0.25
        )
        assert result == 0.0

    def test_kelly_capped_at_max(self) -> None:
        """Kelly is capped at max_kelly."""
        # Very high win rate → Kelly > max_kelly
        result = calculate_kelly_criterion(
            win_rate=0.9, avg_win=0.10, avg_loss=0.02, max_kelly=0.25
        )
        assert result <= 0.25

    def test_invalid_win_rate_raises(self) -> None:
        """Win rate outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Win rate"):
            calculate_kelly_criterion(win_rate=1.5, avg_win=0.05, avg_loss=0.03)

    def test_zero_avg_loss_raises(self) -> None:
        """Zero average loss raises ValueError."""
        with pytest.raises(ValueError, match="Average loss"):
            calculate_kelly_criterion(win_rate=0.6, avg_win=0.05, avg_loss=0.0)


class TestOptimizeKellyPortfolioEdgeCases:
    """Test Kelly portfolio optimization edge cases."""

    def test_all_winning_trades_skipped(self) -> None:
        """Tickers with all wins (no losses) are skipped."""
        df = pd.DataFrame(
            {
                "ticker": ["BTC"] * 5,
                "pnl_pct": [5.0, 3.0, 7.0, 2.0, 4.0],  # All positive
            }
        )

        result = optimize_kelly_portfolio(df, available_cash=10_000_000.0)

        # BTC has 0 losses → both wins and losses check fails → skipped
        assert result == {}

    def test_single_trade_per_ticker_skipped(self) -> None:
        """Tickers with fewer than 2 trades are skipped."""
        df = pd.DataFrame(
            {
                "ticker": ["BTC"],
                "pnl_pct": [5.0],
            }
        )

        result = optimize_kelly_portfolio(df, available_cash=10_000_000.0)

        assert result == {}

    def test_over_allocation_scaled_down(self) -> None:
        """When Kelly allocations exceed available cash, scale down proportionally."""
        # Create trades where Kelly formula gives aggressive allocation
        btc_returns = [
            10.0,
            10.0,
            -2.0,
            10.0,
            10.0,
            -2.0,
            10.0,
            10.0,
            -2.0,
            10.0,
        ]  # ~70% win, high payoff
        eth_returns = [8.0, 8.0, -3.0, 8.0, 8.0, -3.0, 8.0, 8.0, -3.0, 8.0]

        df = pd.DataFrame(
            {
                "ticker": ["BTC"] * len(btc_returns) + ["ETH"] * len(eth_returns),
                "pnl_pct": btc_returns + eth_returns,
            }
        )

        result = optimize_kelly_portfolio(df, available_cash=1_000_000.0)

        # Total allocation should not exceed available cash
        if result:
            total = sum(result.values())
            assert total <= 1_000_000.0 + 1.0  # Small tolerance for float

    def test_empty_dataframe_raises(self) -> None:
        """Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            optimize_kelly_portfolio(pd.DataFrame(), available_cash=10_000_000.0)

    def test_missing_ticker_column_raises(self) -> None:
        """Missing ticker column raises ValueError."""
        df = pd.DataFrame({"pnl_pct": [1.0, 2.0]})
        with pytest.raises(ValueError, match="ticker"):
            optimize_kelly_portfolio(df, available_cash=10_000_000.0)

    def test_missing_return_column_raises(self) -> None:
        """Missing return column raises ValueError."""
        df = pd.DataFrame({"ticker": ["BTC"], "some_col": [1.0]})
        with pytest.raises(ValueError, match="return column"):
            optimize_kelly_portfolio(df, available_cash=10_000_000.0)
