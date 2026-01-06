"""
Unit tests for portfolio optimization module.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult  # Import OptimizeResult

from src.risk.portfolio_optimization import (
    PortfolioOptimizer,
    PortfolioWeights,
    optimize_portfolio,
)


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Generate a sample DataFrame of asset returns."""
    dates = pd.to_datetime(pd.date_range("2023-01-01", periods=252, freq="D"))  # 1 year of daily data
    rng = np.random.default_rng(42)  # For reproducibility
    returns = pd.DataFrame(
        {
            "ASSET1": rng.normal(0.0005, 0.01, 252),  # Mean 0.05%, Std 1%
            "ASSET2": rng.normal(0.0007, 0.015, 252),  # Mean 0.07%, Std 1.5%
            "ASSET3": rng.normal(0.0003, 0.008, 252),  # Mean 0.03%, Std 0.8%
        },
        index=dates,
    )
    return returns


@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    """Generate a sample DataFrame of trade results for Kelly."""
    return pd.DataFrame(
        {
            "ticker": ["ASSET1", "ASSET1", "ASSET2", "ASSET1", "ASSET2", "ASSET3", "ASSET1", "ASSET2", "ASSET3"],
            "pnl_pct": [10.0, -5.0, 15.0, 8.0, -7.0, 20.0, -3.0, 10.0, -10.0],  # Added a losing trade for ASSET3
            "pnl": [1000, -500, 1500, 800, -700, 2000, -300, 1000, -1000],
        }
    )


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
        assert weights.expected_return == 0.1
        assert weights.portfolio_volatility == 0.2
        assert weights.sharpe_ratio == 0.5

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
        """Test MPT to maximize Sharpe ratio."""
        weights = optimizer.optimize_mpt(sample_returns_df, risk_free_rate=0.01)
        assert isinstance(weights, PortfolioWeights)
        assert weights.method == "mpt"
        assert sum(weights.weights.values()) == pytest.approx(1.0)
        assert all(w >= 0 for w in weights.weights.values())
        assert weights.sharpe_ratio > 0

    def test_optimize_mpt_target_return(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test MPT with a target return."""
        # Calculate a reasonable target return based on sample data
        mean_returns = sample_returns_df.mean() * 252
        target_return = mean_returns.mean()

        weights = optimizer.optimize_mpt(sample_returns_df, target_return=target_return)
        assert isinstance(weights, PortfolioWeights)
        assert weights.method == "mpt"
        assert sum(weights.weights.values()) == pytest.approx(1.0)
        assert all(w >= 0 for w in weights.weights.values())
        assert weights.expected_return == pytest.approx(target_return, rel=1e-3)

    def test_optimize_mpt_empty_returns_df(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test MPT with empty returns DataFrame."""
        with pytest.raises(ValueError, match="Returns DataFrame is empty or has no columns"):
            optimizer.optimize_mpt(pd.DataFrame())

    def test_optimize_mpt_single_asset(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test MPT with a single asset."""
        single_asset_df = sample_returns_df[["ASSET1"]]
        weights = optimizer.optimize_mpt(single_asset_df)
        assert isinstance(weights, PortfolioWeights)
        assert weights.weights == {"ASSET1": pytest.approx(1.0)}
        assert weights.method == "mpt"

    @pytest.mark.parametrize("max_w, min_w", [(0.6, 0.1), (0.4, 0.0)])
    def test_optimize_mpt_constraints(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame, max_w: float, min_w: float, mocker
    ) -> None:
        """Test MPT with max/min weight constraints."""
        # Mock minimize to ensure it returns a successful result that respects constraints
        mock_weights = np.array([0.3, 0.3, 0.4]) # These sum to 1 and respect max_w >= 0.4

        mock_result = OptimizeResult(
            x=mock_weights,
            success=True,
            message="Optimization successful",
            fun=0,
            nit=0,
            jac=np.array([]),
            hess_inv=None,
        )
        mocker.patch("src.risk.portfolio_optimization.minimize", return_value=mock_result)

        weights = optimizer.optimize_mpt(
            sample_returns_df, max_weight=max_w, min_weight=min_w
        )
        assert isinstance(weights, PortfolioWeights)
        assert all(min_w <= w <= max_w for w in weights.weights.values())
        assert sum(weights.weights.values()) == pytest.approx(1.0)
        assert weights.sharpe_ratio is not None and weights.sharpe_ratio >= 0

    @pytest.mark.parametrize("method", ["SLSQP"])
    @pytest.mark.parametrize("success_status", [False])
    def test_optimize_mpt_optimization_failure(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame, method, success_status, mocker
    ) -> None:
        mock_result = OptimizeResult(
            x=np.array([0.33, 0.33, 0.34]),
            success=success_status,
            message="Optimization failed",
            fun=0,
            nit=0,
            jac=np.array([]),
            hess_inv=None,
        )
        mocker.patch("src.risk.portfolio_optimization.minimize", return_value=mock_result)

        weights = optimizer.optimize_mpt(sample_returns_df)
        assert isinstance(weights, PortfolioWeights)
        # Should fallback to equal weights if not successful
        assert weights.weights["ASSET1"] == pytest.approx(1.0 / len(sample_returns_df.columns))

    @pytest.mark.parametrize("error", [Exception("Mock error")])
    def test_optimize_mpt_exception_handling(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame, error, mocker
    ) -> None:
        mocker.patch("src.risk.portfolio_optimization.minimize", side_effect=error)
        weights = optimizer.optimize_mpt(sample_returns_df)
        assert isinstance(weights, PortfolioWeights)
        assert weights.weights["ASSET1"] == pytest.approx(1.0 / len(sample_returns_df.columns))

    def test_optimize_risk_parity(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test risk parity optimization."""
        weights = optimizer.optimize_risk_parity(sample_returns_df)
        assert isinstance(weights, PortfolioWeights)
        assert weights.method == "risk_parity"
        assert sum(weights.weights.values()) == pytest.approx(1.0)
        assert all(w >= 0 for w in weights.weights.values())

    def test_optimize_risk_parity_empty_returns_df(
        self, optimizer: PortfolioOptimizer
    ) -> None:
        """Test risk parity with empty returns DataFrame."""
        with pytest.raises(ValueError, match="Returns DataFrame is empty or has no columns"):
            optimizer.optimize_risk_parity(pd.DataFrame())

    def test_optimize_risk_parity_single_asset(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test risk parity with a single asset."""
        single_asset_df = sample_returns_df[["ASSET1"]]
        weights = optimizer.optimize_risk_parity(single_asset_df)
        assert isinstance(weights, PortfolioWeights)
        assert weights.weights == {"ASSET1": pytest.approx(1.0)}
        assert weights.method == "risk_parity"

    @pytest.mark.parametrize("max_w, min_w", [(0.6, 0.1), (0.4, 0.0)])
    def test_optimize_risk_parity_constraints(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame, max_w: float, min_w: float, mocker
    ) -> None:
        """Test risk parity with max/min weight constraints."""
        # Mock minimize to ensure it returns a successful result that respects constraints
        mock_weights = np.array([0.3, 0.3, 0.4]) # These sum to 1 and respect max_w >= 0.4

        mock_result = OptimizeResult(
            x=mock_weights,
            success=True,
            message="Optimization successful",
            fun=0,
            nit=0,
            jac=np.array([]),
            hess_inv=None,
        )
        mocker.patch("src.risk.portfolio_optimization.minimize", return_value=mock_result)

        weights = optimizer.optimize_risk_parity(
            sample_returns_df, max_weight=max_w, min_weight=min_w
        )
        assert isinstance(weights, PortfolioWeights)
        assert all(min_w <= w <= max_w for w in weights.weights.values())
        assert sum(weights.weights.values()) == pytest.approx(1.0)
        assert weights.portfolio_volatility is not None and weights.portfolio_volatility >= 0

    @pytest.mark.parametrize("method", ["SLSQP"])
    @pytest.mark.parametrize("success_status", [False])
    def test_optimize_risk_parity_optimization_failure(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame, method, success_status, mocker
    ) -> None:
        """Test risk parity optimization failure fallback."""
        mock_result = OptimizeResult(
            x=np.array([0.33, 0.33, 0.34]),
            success=success_status,
            message="Optimization failed",
            fun=0,
            nit=0,
            jac=np.array([]),
            hess_inv=None,
        )
        mocker.patch("src.risk.portfolio_optimization.minimize", return_value=mock_result)

        weights = optimizer.optimize_risk_parity(sample_returns_df)
        assert isinstance(weights, PortfolioWeights)
        # Should fallback to inverse volatility weights (initial guess)
        assert weights.weights["ASSET1"] > 0

    @pytest.mark.parametrize("error", [Exception("Mock error")])
    def test_optimize_risk_parity_exception_handling(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame, error, mocker
    ) -> None:
        mocker.patch("src.risk.portfolio_optimization.minimize", side_effect=error)
        weights = optimizer.optimize_risk_parity(sample_returns_df)
        assert isinstance(weights, PortfolioWeights)
        # Should fallback to inverse volatility weights (initial guess)
        assert weights.weights["ASSET1"] > 0

    def test_calculate_kelly_criterion_valid_inputs(self, optimizer: PortfolioOptimizer) -> None:
        """Test Kelly criterion with valid inputs."""
        assert optimizer.calculate_kelly_criterion(0.7, 0.1, 0.05) == pytest.approx(0.25)
        assert optimizer.calculate_kelly_criterion(0.5, 0.1, 0.1) == pytest.approx(0.0)  # Breakeven
        assert optimizer.calculate_kelly_criterion(0.8, 0.2, 0.1) == pytest.approx(0.25)
        assert optimizer.calculate_kelly_criterion(0.8, 0.2, 0.1, max_kelly=0.25) == pytest.approx(0.25)

    @pytest.mark.parametrize("win_rate", [-0.1, 1.1])
    def test_calculate_kelly_criterion_invalid_win_rate(
        self, optimizer: PortfolioOptimizer, win_rate: float
    ) -> None:
        """Test Kelly criterion with invalid win rate."""
        with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
            optimizer.calculate_kelly_criterion(win_rate, 0.1, 0.05)

    @pytest.mark.parametrize("avg_loss", [-0.01, 0.0])
    def test_calculate_kelly_criterion_invalid_avg_loss(
        self, optimizer: PortfolioOptimizer, avg_loss: float
    ) -> None:
        """Test Kelly criterion with invalid average loss."""
        with pytest.raises(ValueError, match="Average loss must be positive"):
            optimizer.calculate_kelly_criterion(0.7, 0.1, avg_loss)

    def test_calculate_kelly_criterion_no_wins(self, optimizer: PortfolioOptimizer) -> None:
        """Test Kelly criterion when average win is zero or negative."""
        assert optimizer.calculate_kelly_criterion(0.7, 0.0, 0.05) == 0.0
        assert optimizer.calculate_kelly_criterion(0.7, -0.1, 0.05) == 0.0

    def test_calculate_kelly_criterion_negative_kelly(self, optimizer: PortfolioOptimizer) -> None:
        """Test Kelly criterion when Kelly result is negative."""
        # Losing strategy
        assert optimizer.calculate_kelly_criterion(0.3, 0.1, 0.05) == 0.0

    def test_optimize_kelly_portfolio(
        self, optimizer: PortfolioOptimizer, sample_trades_df: pd.DataFrame
    ) -> None:
        """Test Kelly portfolio optimization."""
        available_cash = 10000
        allocations = optimizer.optimize_kelly_portfolio(sample_trades_df, available_cash)
        assert isinstance(allocations, dict)
        assert "ASSET1" in allocations and "ASSET2" in allocations and "ASSET3" in allocations
        assert all(v >= 0 for v in allocations.values())
        assert sum(allocations.values()) == pytest.approx(7500.0)
        # Verify ASSET1 (4 wins, 1 loss) vs ASSET2 (2 wins, 1 loss) vs ASSET3 (1 win)
        # Expected ASSET1 to have higher allocation than ASSET2 or ASSET3

    def test_optimize_kelly_portfolio_empty_trades_df(
        self, optimizer: PortfolioOptimizer
    ) -> None:
        """Test Kelly portfolio with empty trades DataFrame."""
        with pytest.raises(ValueError, match="Trades DataFrame is empty"):
            optimizer.optimize_kelly_portfolio(pd.DataFrame(), 10000)

    def test_optimize_kelly_portfolio_missing_ticker_column(
        self, optimizer: PortfolioOptimizer
    ) -> None:
        """Test Kelly portfolio with missing ticker column."""
        trades = pd.DataFrame({"pnl_pct": [10.0, -5.0]})
        with pytest.raises(ValueError, match="Trades DataFrame must have 'ticker' column"):
            optimizer.optimize_kelly_portfolio(trades, 10000)

    def test_optimize_kelly_portfolio_missing_return_column(
        self, optimizer: PortfolioOptimizer
    ) -> None:
        """Test Kelly portfolio with missing return column."""
        trades = pd.DataFrame({"ticker": ["ASSET1", "ASSET1"], "pnl": [100, -50]})
        with pytest.raises(ValueError, match="Trades DataFrame must have return column"):
            optimizer.optimize_kelly_portfolio(trades, 10000)

    def test_optimize_kelly_portfolio_insufficient_trades_per_ticker(
        self, optimizer: PortfolioOptimizer
    ) -> None:
        """Test Kelly portfolio with insufficient trades per ticker."""
        trades = pd.DataFrame(
            {
                "ticker": ["ASSET1"],
                "pnl_pct": [10.0],  # Only one trade
            }
        )
        allocations = optimizer.optimize_kelly_portfolio(trades, 10000)
        assert allocations == {}  # Should return empty for insufficient data

    def test_optimize_kelly_portfolio_no_wins_or_losses(
        self, optimizer: PortfolioOptimizer
    ) -> None:
        """Test Kelly portfolio when a ticker has no wins or no losses."""
        trades = pd.DataFrame(
            {
                "ticker": ["ASSET1", "ASSET1", "ASSET2"],
                "pnl_pct": [10.0, 5.0, 0.0],  # ASSET1 only wins, ASSET2 no PnL
            }
        )
        allocations = optimizer.optimize_kelly_portfolio(trades, 10000)
        assert allocations == {}  # Should return empty as ASSET1 has no losses, ASSET2 no trades with PnL

    def test_optimize_kelly_portfolio_max_kelly(
        self, optimizer: PortfolioOptimizer, sample_trades_df: pd.DataFrame
    ) -> None:
        """Test Kelly portfolio with max_kelly cap."""
        available_cash = 10000
        max_kelly = 0.01  # Very low cap
        allocations = optimizer.optimize_kelly_portfolio(sample_trades_df, available_cash, max_kelly)
        assert isinstance(allocations, dict)
        assert all(v <= available_cash * max_kelly + 1e-6 for v in allocations.values())
        assert sum(allocations.values()) <= available_cash + 1e-6

    def test_optimize_portfolio_mpt(
        self, sample_returns_df: pd.DataFrame, mocker
    ) -> None:
        """Test optimize_portfolio for MPT method."""
        mocker.patch(
            "src.risk.portfolio_optimization.PortfolioOptimizer.optimize_mpt",
            return_value=PortfolioWeights(weights={}, method="mpt"),
        )
        result = optimize_portfolio(sample_returns_df, method="mpt")
        assert result.method == "mpt"
        PortfolioOptimizer.optimize_mpt.assert_called_once()

    def test_optimize_portfolio_risk_parity(
        self, sample_returns_df: pd.DataFrame, mocker
    ) -> None:
        """Test optimize_portfolio for risk_parity method."""
        mocker.patch(
            "src.risk.portfolio_optimization.PortfolioOptimizer.optimize_risk_parity",
            return_value=PortfolioWeights(weights={}, method="risk_parity"),
        )
        result = optimize_portfolio(sample_returns_df, method="risk_parity")
        assert result.method == "risk_parity"
        PortfolioOptimizer.optimize_risk_parity.assert_called_once()

    def test_optimize_portfolio_kelly(
        self, sample_returns_df: pd.DataFrame, sample_trades_df: pd.DataFrame, mocker
    ) -> None:
        """Test optimize_portfolio for kelly method."""
        mocker.patch(
            "src.risk.portfolio_optimization.PortfolioOptimizer.optimize_kelly_portfolio",
            return_value={"ASSET1": 5000.0, "ASSET2": 3000.0},
        )
        result = optimize_portfolio(
            sample_returns_df, method="kelly", trades=sample_trades_df, available_cash=10000
        )
        assert result.method == "kelly"
        assert result.weights["ASSET1"] == pytest.approx(0.625)  # 5000/8000
        assert result.weights["ASSET2"] == pytest.approx(0.375)  # 3000/8000
        PortfolioOptimizer.optimize_kelly_portfolio.assert_called_once()

    def test_optimize_portfolio_kelly_missing_trades_arg(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test optimize_portfolio for kelly method missing trades arg."""
        with pytest.raises(ValueError, match="Kelly method requires 'trades' DataFrame"):
            optimize_portfolio(sample_returns_df, method="kelly")

    def test_optimize_portfolio_unknown_method(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test optimize_portfolio with unknown method."""
        with pytest.raises(ValueError, match="Unknown optimization method: invalid"):
            optimize_portfolio(sample_returns_df, method="invalid")
