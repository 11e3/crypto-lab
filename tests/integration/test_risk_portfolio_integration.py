"""Integration tests: Risk module and Portfolio Optimization.

Tests the portfolio risk metrics, optimization, and position sizing
working together as a cohesive pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.metrics import (
    calculate_cvar,
    calculate_portfolio_correlation,
    calculate_portfolio_risk_metrics,
    calculate_portfolio_volatility,
    calculate_var,
)
from src.risk.portfolio_models import OptimizationMethod
from src.risk.portfolio_optimization import PortfolioOptimizer, optimize_portfolio
from src.risk.position_sizing import PositionSizingMethod, calculate_position_size


@pytest.fixture()
def multi_asset_returns() -> pd.DataFrame:
    """Generate correlated multi-asset return data."""
    np.random.seed(42)
    n = 365

    # Create correlated returns (BTC, ETH correlated, XRP less so)
    btc = np.random.normal(0.001, 0.03, n)
    eth = btc * 0.7 + np.random.normal(0.0005, 0.02, n)
    xrp = np.random.normal(0.0005, 0.04, n)

    return pd.DataFrame(
        {"KRW-BTC": btc, "KRW-ETH": eth, "KRW-XRP": xrp},
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


@pytest.fixture()
def sample_historical_data() -> pd.DataFrame:
    """Generate historical OHLCV data for position sizing."""
    np.random.seed(42)
    n = 50
    close = 50_000_000 + np.cumsum(np.random.randn(n) * 500_000)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100_000,
            "high": close + abs(np.random.randn(n) * 200_000),
            "low": close - abs(np.random.randn(n) * 200_000),
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


class TestRiskMetricsPipeline:
    """Test risk metrics calculation pipeline."""

    def test_var_cvar_relationship(self, multi_asset_returns: pd.DataFrame) -> None:
        """CVaR should always be >= VaR."""
        for ticker in multi_asset_returns.columns:
            returns = multi_asset_returns[ticker].values
            var = calculate_var(returns, confidence_level=0.95)
            cvar = calculate_cvar(returns, confidence_level=0.95)
            assert cvar >= var, f"CVaR < VaR for {ticker}"

    def test_portfolio_volatility(self, multi_asset_returns: pd.DataFrame) -> None:
        """Portfolio volatility should be finite and positive."""
        # Use a single asset's returns as "portfolio returns"
        returns = multi_asset_returns["KRW-BTC"].values
        vol = calculate_portfolio_volatility(returns)
        assert vol > 0.0
        assert np.isfinite(vol)

    def test_portfolio_correlation(self, multi_asset_returns: pd.DataFrame) -> None:
        """calculate_portfolio_correlation returns tuple with correlation matrix."""
        asset_returns = {
            col: multi_asset_returns[col].values for col in multi_asset_returns.columns
        }
        avg_corr, max_corr, min_corr, corr_matrix = calculate_portfolio_correlation(asset_returns)
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        # Diagonal should be 1.0
        for col in corr_matrix.columns:
            assert corr_matrix.loc[col, col] == pytest.approx(1.0)
        # BTC and ETH should be positively correlated (by construction)
        assert max_corr > 0.0

    def test_risk_metrics_full(self, multi_asset_returns: pd.DataFrame) -> None:
        """calculate_portfolio_risk_metrics produces complete metrics."""
        returns = multi_asset_returns["KRW-BTC"].values
        equity = (1 + multi_asset_returns["KRW-BTC"]).cumprod().values * 10_000_000

        asset_returns = {
            col: multi_asset_returns[col].values for col in multi_asset_returns.columns
        }

        metrics = calculate_portfolio_risk_metrics(
            equity_curve=equity,
            daily_returns=returns,
            asset_returns=asset_returns,
        )
        assert np.isfinite(metrics.var_95)
        assert np.isfinite(metrics.cvar_95)
        assert np.isfinite(metrics.portfolio_volatility)
        # With 3 assets, correlation should be calculated
        assert metrics.avg_correlation is not None


class TestPortfolioOptimizationPipeline:
    """Test portfolio optimization methods."""

    def test_mpt_optimization(self, multi_asset_returns: pd.DataFrame) -> None:
        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize_mpt(multi_asset_returns)
        assert len(weights.weights) == 3
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=0.01)
        assert weights.method == "mpt"

    def test_risk_parity_optimization(self, multi_asset_returns: pd.DataFrame) -> None:
        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize_risk_parity(multi_asset_returns)
        assert len(weights.weights) == 3
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=0.01)
        assert weights.method == "risk_parity"

    def test_convenience_function(self, multi_asset_returns: pd.DataFrame) -> None:
        """optimize_portfolio() convenience function routes correctly."""
        mpt_weights = optimize_portfolio(multi_asset_returns, method=OptimizationMethod.MPT)
        rp_weights = optimize_portfolio(multi_asset_returns, method=OptimizationMethod.RISK_PARITY)

        assert mpt_weights.method == "mpt"
        assert rp_weights.method == "risk_parity"
        # Different methods should produce different weights
        assert mpt_weights.weights != rp_weights.weights

    def test_weight_constraints(self, multi_asset_returns: pd.DataFrame) -> None:
        """Weights should respect min/max constraints."""
        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize_mpt(multi_asset_returns, max_weight=0.5, min_weight=0.1)
        for w in weights.weights.values():
            assert w >= 0.1 - 0.01  # Small tolerance for numerical optimization
            assert w <= 0.5 + 0.01


class TestPositionSizingIntegration:
    """Test position sizing with real-ish data."""

    def test_equal_sizing(self) -> None:
        size = calculate_position_size(
            method=PositionSizingMethod.EQUAL,
            available_cash=10_000_000,
            available_slots=5,
            ticker="KRW-BTC",
            current_price=50_000_000,
        )
        assert size == pytest.approx(2_000_000)

    def test_volatility_sizing_with_data(self, sample_historical_data: pd.DataFrame) -> None:
        """Volatility-based sizing should use historical data."""
        size = calculate_position_size(
            method=PositionSizingMethod.VOLATILITY,
            available_cash=10_000_000,
            available_slots=5,
            ticker="KRW-BTC",
            current_price=50_000_000,
            historical_data=sample_historical_data,
        )
        assert size > 0.0
        assert size <= 10_000_000  # Never exceeds available cash

    def test_sizing_methods_are_strings(self) -> None:
        """StrEnum backward compatibility: works with plain strings."""
        size_enum = calculate_position_size(
            method=PositionSizingMethod.EQUAL,
            available_cash=10_000_000,
            available_slots=5,
            ticker="KRW-BTC",
            current_price=50_000_000,
        )
        size_str = calculate_position_size(
            method="equal",  # type: ignore[arg-type]
            available_cash=10_000_000,
            available_slots=5,
            ticker="KRW-BTC",
            current_price=50_000_000,
        )
        assert size_enum == size_str

    def test_no_slots_returns_zero(self) -> None:
        size = calculate_position_size(
            method=PositionSizingMethod.EQUAL,
            available_cash=10_000_000,
            available_slots=0,
            ticker="KRW-BTC",
            current_price=50_000_000,
        )
        assert size == 0.0


class TestRiskOptimizationEnd2End:
    """End-to-end: Returns → Risk Metrics → Optimization → Position Sizing."""

    def test_full_pipeline(self, multi_asset_returns: pd.DataFrame) -> None:
        """Complete risk analysis pipeline."""
        # Step 1: Calculate risk metrics per asset
        for ticker in multi_asset_returns.columns:
            returns = multi_asset_returns[ticker].values
            var = calculate_var(returns, confidence_level=0.95)
            cvar = calculate_cvar(returns, confidence_level=0.95)
            assert np.isfinite(var)
            assert np.isfinite(cvar)

        # Step 2: Optimize portfolio
        weights = optimize_portfolio(multi_asset_returns, method=OptimizationMethod.MPT)
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=0.01)

        # Step 3: Calculate portfolio-level risk
        # Use weighted returns for portfolio vol
        weight_array = np.array([weights.weights[c] for c in multi_asset_returns.columns])
        portfolio_returns = (multi_asset_returns.values * weight_array).sum(axis=1)
        portfolio_vol = calculate_portfolio_volatility(portfolio_returns)
        assert np.isfinite(portfolio_vol)

        # Step 4: Position sizing per asset
        total_capital = 10_000_000
        for _ticker, weight in weights.weights.items():
            allocated = total_capital * weight
            assert allocated >= 0.0
            assert allocated <= total_capital
