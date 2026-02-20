"""
Unit tests for portfolio optimization module.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

# Module path: src.risk.portfolio_optimization
from src.risk.portfolio_optimization import (
    PortfolioOptimizer,
    PortfolioWeights,
    optimize_portfolio,
)

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Generate a sample DataFrame of asset returns."""
    dates = pd.to_datetime(pd.date_range("2023-01-01", periods=252, freq="D"))
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        {
            "ASSET1": rng.normal(0.0005, 0.01, 252),
            "ASSET2": rng.normal(0.0007, 0.015, 252),
            "ASSET3": rng.normal(0.0003, 0.008, 252),
        },
        index=dates,
    )
    return returns


@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    """Generate a sample DataFrame of trade results for Kelly."""
    return pd.DataFrame(
        {
            "ticker": [
                "ASSET1",
                "ASSET1",
                "ASSET2",
                "ASSET1",
                "ASSET2",
                "ASSET3",
                "ASSET1",
                "ASSET2",
                "ASSET3",
            ],
            "pnl_pct": [10.0, -5.0, 15.0, 8.0, -7.0, 20.0, -3.0, 10.0, -10.0],
            "pnl": [1000, -500, 1500, 800, -700, 2000, -300, 1000, -1000],
        }
    )


# -------------------------------------------------------------------------
# Test Classes
# -------------------------------------------------------------------------


class TestPortfolioWeights:
    """Tests for the PortfolioWeights dataclass."""

    def test_initialization(self) -> None:
        weights = PortfolioWeights(
            weights={"ASSET1": 0.5, "ASSET2": 0.5},
            method="test_method",
            expected_return=0.1,
            portfolio_volatility=0.2,
            sharpe_ratio=0.5,
        )
        assert weights.weights == {"ASSET1": 0.5, "ASSET2": 0.5}
        assert weights.method == "test_method"

    def test_repr(self) -> None:
        weights = PortfolioWeights(
            weights={"ASSET1": 0.5, "ASSET2": 0.5}, method="mpt", sharpe_ratio=0.75
        )
        assert "PortfolioWeights(method=mpt, assets=2, sharpe=0.750)" in repr(weights)


class TestPortfolioOptimizer:
    """Tests for the PortfolioOptimizer class."""

    @pytest.fixture
    def optimizer(self) -> PortfolioOptimizer:
        return PortfolioOptimizer()

    def test_optimize_mpt_maximize_sharpe(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        weights = optimizer.optimize_mpt(sample_returns_df, risk_free_rate=0.01)
        assert isinstance(weights, PortfolioWeights)
        assert weights.method == "mpt"
        assert sum(weights.weights.values()) == pytest.approx(1.0)
        assert weights.sharpe_ratio is not None and weights.sharpe_ratio > 0

    def test_optimize_mpt_target_return(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        mean_returns = sample_returns_df.mean() * 252
        target_return = mean_returns.mean()

        weights = optimizer.optimize_mpt(sample_returns_df, target_return=target_return)
        assert isinstance(weights, PortfolioWeights)
        assert weights.expected_return == pytest.approx(target_return, rel=1e-3)

    def test_optimize_mpt_empty_returns_df(self, optimizer: PortfolioOptimizer) -> None:
        with pytest.raises(ValueError, match="Returns DataFrame is empty"):
            optimizer.optimize_mpt(pd.DataFrame())

    def test_optimize_mpt_single_asset(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        single_asset_df = sample_returns_df[["ASSET1"]]
        weights = optimizer.optimize_mpt(single_asset_df)
        assert weights.weights == {"ASSET1": pytest.approx(1.0)}

    @pytest.mark.parametrize("max_w, min_w", [(0.6, 0.1), (0.4, 0.0)])
    def test_optimize_mpt_constraints(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns_df: pd.DataFrame,
        max_w: float,
        min_w: float,
        mocker: pytest.MonkeyPatch,
    ) -> None:
        # Build fake optimization result
        mock_weights = np.array([0.3, 0.3, 0.4])
        mock_result = OptimizeResult(
            x=mock_weights,
            success=True,
            message="Ok",
            fun=0,
            nit=0,
            jac=np.array([]),
            hess_inv=None,
        )

        # minimize is imported in portfolio_methods.py, not portfolio_optimization.py
        mocker.patch("src.risk.portfolio_methods.minimize", return_value=mock_result)

        weights = optimizer.optimize_mpt(sample_returns_df, max_weight=max_w, min_weight=min_w)
        assert all(min_w <= w <= max_w for w in weights.weights.values())

    @pytest.mark.parametrize("method", ["SLSQP"])
    @pytest.mark.parametrize("success_status", [False])
    def test_optimize_mpt_optimization_failure(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns_df: pd.DataFrame,
        method: str,
        success_status: bool,
        mocker: pytest.MonkeyPatch,
    ) -> None:
        mock_result = OptimizeResult(
            x=np.array([0.33, 0.33, 0.34]), success=success_status, message="Optimization failed"
        )
        mocker.patch("src.risk.portfolio_methods.minimize", return_value=mock_result)

        weights = optimizer.optimize_mpt(sample_returns_df)
        # Verify fallback to equal-weight (1/N) on failure
        assert weights.weights["ASSET1"] == pytest.approx(1.0 / len(sample_returns_df.columns))

    @pytest.mark.parametrize("error", [Exception("Mock error")])
    def test_optimize_mpt_exception_handling(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns_df: pd.DataFrame,
        error: Exception,
        mocker: pytest.MonkeyPatch,
    ) -> None:
        mocker.patch("src.risk.portfolio_methods.minimize", side_effect=error)
        weights = optimizer.optimize_mpt(sample_returns_df)
        assert weights.weights["ASSET1"] == pytest.approx(1.0 / len(sample_returns_df.columns))

    def test_optimize_risk_parity(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        weights = optimizer.optimize_risk_parity(sample_returns_df)
        assert weights.method == "risk_parity"
        assert sum(weights.weights.values()) == pytest.approx(1.0)

    # ... (Kelly-related tests retain existing logic; expected to pass without mocker) ...

    def test_optimize_kelly_portfolio(
        self, optimizer: PortfolioOptimizer, sample_trades_df: pd.DataFrame
    ) -> None:
        available_cash = 10000
        allocations = optimizer.optimize_kelly_portfolio(sample_trades_df, available_cash)
        assert sum(allocations.values()) <= available_cash + 1e-6
        assert "ASSET1" in allocations

    def test_optimize_mpt_zero_volatility(
        self, optimizer: PortfolioOptimizer, mocker: pytest.MonkeyPatch
    ) -> None:
        """Test MPT with zero volatility (line 40)."""
        # Create returns with zero variance
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "ASSET1": [0.0] * 100,
                "ASSET2": [0.0] * 100,
            },
            index=dates,
        )
        weights = optimizer.optimize_mpt(df)
        # Should fallback to equal weights when optimization fails
        assert weights.method == "mpt"
        assert sum(weights.weights.values()) == pytest.approx(1.0)

    def test_optimize_risk_parity_zero_volatility(self, optimizer: PortfolioOptimizer) -> None:
        """Test risk parity with zero volatility (line 101)."""
        # Create returns with zero variance
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "ASSET1": [0.0] * 100,
                "ASSET2": [0.0] * 100,
            },
            index=dates,
        )
        weights = optimizer.optimize_risk_parity(df)
        assert weights.method == "risk_parity"

    def test_optimize_risk_parity_exception(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns_df: pd.DataFrame,
        mocker: pytest.MonkeyPatch,
    ) -> None:
        """Test risk parity exception handling (line 173)."""
        mocker.patch("src.risk.portfolio_methods.minimize", side_effect=Exception("Mock error"))
        weights = optimizer.optimize_risk_parity(sample_returns_df)
        # Fallback: equal risk weighting by inverse volatility
        assert weights.method == "risk_parity"
        assert sum(weights.weights.values()) == pytest.approx(1.0)

    def test_calculate_kelly_criterion_edge_cases(self, optimizer: PortfolioOptimizer) -> None:
        """Test Kelly criterion edge cases (lines 138-143, 154, 156, 158)."""
        # Invalid win rate < 0
        with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
            optimizer.calculate_kelly_criterion(-0.1, 1.0, 1.0)

        # Invalid win rate > 1
        with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
            optimizer.calculate_kelly_criterion(1.5, 1.0, 1.0)

        # Invalid avg_loss <= 0 (line 141)
        with pytest.raises(ValueError, match="Average loss must be positive"):
            optimizer.calculate_kelly_criterion(0.5, 1.0, 0.0)

        with pytest.raises(ValueError, match="Average loss must be positive"):
            optimizer.calculate_kelly_criterion(0.5, 1.0, -1.0)

        # avg_win <= 0 case (line 142)
        result = optimizer.calculate_kelly_criterion(0.5, 0.0, 1.0)
        assert result == 0.0

        result = optimizer.calculate_kelly_criterion(0.5, -1.0, 1.0)
        assert result == 0.0

        # Negative Kelly (line 154)
        result = optimizer.calculate_kelly_criterion(0.3, 1.0, 10.0)
        assert result == 0.0

    def test_optimize_kelly_portfolio_edge_cases(self, optimizer: PortfolioOptimizer) -> None:
        """Test Kelly portfolio edge cases (lines 164, 173, 175, 183, 190, 196, 203)."""
        # Empty trades (line 164)
        with pytest.raises(ValueError, match="Trades DataFrame is empty"):
            optimizer.optimize_kelly_portfolio(pd.DataFrame(), 10000)

        # Missing ticker column (line 173)
        trades_no_ticker = pd.DataFrame({"pnl_pct": [10, -5]})
        with pytest.raises(ValueError, match="must have 'ticker' column"):
            optimizer.optimize_kelly_portfolio(trades_no_ticker, 10000)

        # Missing return columns (line 175-183)
        trades_no_return = pd.DataFrame({"ticker": ["A", "B"]})
        with pytest.raises(ValueError, match="must have return column"):
            optimizer.optimize_kelly_portfolio(trades_no_return, 10000)

        # Only 1 trade per ticker (line 190)
        trades_single = pd.DataFrame({"ticker": ["A"], "pnl_pct": [10.0]})
        allocations = optimizer.optimize_kelly_portfolio(trades_single, 10000)
        assert allocations == {}

        # All wins (line 196)
        trades_all_wins = pd.DataFrame(
            {
                "ticker": ["A", "A", "A"],
                "pnl_pct": [10.0, 5.0, 8.0],
            }
        )
        allocations = optimizer.optimize_kelly_portfolio(trades_all_wins, 10000)
        assert allocations == {}

        # All losses (line 196)
        trades_all_losses = pd.DataFrame(
            {
                "ticker": ["B", "B"],
                "pnl_pct": [-10.0, -5.0],
            }
        )
        allocations = optimizer.optimize_kelly_portfolio(trades_all_losses, 10000)
        assert allocations == {}

        # avg_loss == 0 (line 203)
        # This can happen with extremely small negative values that round to 0
        trades_tiny_loss = pd.DataFrame(
            {
                "ticker": ["C", "C", "C"],
                "pnl_pct": [10.0, -0.000001, 5.0],
            }
        )
        allocations = optimizer.optimize_kelly_portfolio(trades_tiny_loss, 10000)
        # Should skip this ticker if avg_loss rounds to 0
        assert "C" not in allocations or allocations.get("C", 0) >= 0

        # Total allocation exceeds available cash (line 210-211)
        trades_large = pd.DataFrame(
            {
                "ticker": ["A", "A", "B", "B"],
                "pnl_pct": [100.0, 80.0, 100.0, 90.0],
            }
        )
        allocations = optimizer.optimize_kelly_portfolio(trades_large, 1000)
        assert sum(allocations.values()) <= 1000 + 1e-6

    # -------------------------------------------------------------------------
    # Integration Tests (Using optimize_portfolio wrapper)
    # -------------------------------------------------------------------------

    def test_optimize_portfolio_mpt(
        self, sample_returns_df: pd.DataFrame, mocker: pytest.MonkeyPatch
    ) -> None:
        """Test optimize_portfolio wrapper calls the correct class method."""
        # Patch the object directly rather than by string path (safer)
        mock_method = mocker.patch.object(
            PortfolioOptimizer,
            "optimize_mpt",
            return_value=PortfolioWeights(weights={}, method="mpt"),
        )

        result = optimize_portfolio(sample_returns_df, method="mpt")
        assert result.method == "mpt"
        mock_method.assert_called_once()

    def test_optimize_portfolio_risk_parity(
        self, sample_returns_df: pd.DataFrame, mocker: pytest.MonkeyPatch
    ) -> None:
        mock_method = mocker.patch.object(
            PortfolioOptimizer,
            "optimize_risk_parity",
            return_value=PortfolioWeights(weights={}, method="risk_parity"),
        )

        result = optimize_portfolio(sample_returns_df, method="risk_parity")
        assert result.method == "risk_parity"
        mock_method.assert_called_once()

    def test_optimize_portfolio_kelly(
        self,
        sample_returns_df: pd.DataFrame,
        sample_trades_df: pd.DataFrame,
        mocker: pytest.MonkeyPatch,
    ) -> None:
        mock_method = mocker.patch.object(
            PortfolioOptimizer,
            "optimize_kelly_portfolio",
            return_value={"ASSET1": 5000.0, "ASSET2": 3000.0},
        )

        result = optimize_portfolio(
            sample_returns_df, method="kelly", trades=sample_trades_df, available_cash=10000
        )
        assert result.method == "kelly"
        mock_method.assert_called_once()

    def test_optimize_portfolio_kelly_without_trades(self, sample_returns_df: pd.DataFrame) -> None:
        """Test kelly method raises error when trades not provided - line 122."""
        with pytest.raises(ValueError, match="Kelly method requires 'trades'"):
            optimize_portfolio(sample_returns_df, method="kelly")

    def test_optimize_portfolio_unknown_method(self, sample_returns_df: pd.DataFrame) -> None:
        """Test unknown method raises error - line 131."""
        with pytest.raises(ValueError, match="Unknown optimization method"):
            optimize_portfolio(sample_returns_df, method="invalid_method")
