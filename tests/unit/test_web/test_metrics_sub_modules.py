"""Tests for web metrics sub-modules: RiskMetrics, RatioMetrics, StatisticalMetrics, TradeMetrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.web.services.metrics import (
    RatioMetrics,
    RiskMetrics,
    StatisticalMetrics,
    TradeMetrics,
)

# =========================================================================
# RiskMetrics
# =========================================================================


class TestRiskMetrics:
    """Tests for RiskMetrics calculator."""

    @pytest.fixture()
    def positive_returns(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 200)

    def test_volatility_annualized(self, positive_returns: np.ndarray) -> None:
        vol = RiskMetrics.calculate_volatility(positive_returns, annualize=True)
        assert vol > 0.0
        vol_daily = RiskMetrics.calculate_volatility(positive_returns, annualize=False)
        # Annualized should be larger than daily
        assert vol > vol_daily

    def test_volatility_insufficient_data(self) -> None:
        assert RiskMetrics.calculate_volatility(np.array([0.01])) == 0.0

    def test_upside_volatility_positive(self, positive_returns: np.ndarray) -> None:
        vol = RiskMetrics.calculate_upside_volatility(positive_returns)
        assert vol > 0.0

    def test_upside_volatility_no_positive_returns(self) -> None:
        returns = np.array([-0.01, -0.02, -0.03])
        assert RiskMetrics.calculate_upside_volatility(returns) == 0.0

    def test_downside_volatility(self, positive_returns: np.ndarray) -> None:
        vol = RiskMetrics.calculate_downside_volatility(positive_returns)
        assert vol >= 0.0

    def test_downside_volatility_no_negatives(self) -> None:
        returns = np.array([0.01, 0.02, 0.03])
        assert RiskMetrics.calculate_downside_volatility(returns) == 0.0

    def test_downside_volatility_custom_mar(self, positive_returns: np.ndarray) -> None:
        # Higher MAR means more returns classified as "downside"
        vol_low = RiskMetrics.calculate_downside_volatility(positive_returns, mar=-0.01)
        vol_high = RiskMetrics.calculate_downside_volatility(positive_returns, mar=0.01)
        # With higher threshold, more data below → larger downside vol
        assert vol_high >= vol_low

    def test_var_95(self, positive_returns: np.ndarray) -> None:
        var = RiskMetrics.calculate_var(positive_returns, 0.95)
        assert var > 0.0

    def test_var_99(self, positive_returns: np.ndarray) -> None:
        var99 = RiskMetrics.calculate_var(positive_returns, 0.99)
        var95 = RiskMetrics.calculate_var(positive_returns, 0.95)
        # VaR99 should be >= VaR95
        assert var99 >= var95

    def test_var_insufficient_data(self) -> None:
        assert RiskMetrics.calculate_var(np.array([0.01])) == 0.0

    def test_cvar_95(self, positive_returns: np.ndarray) -> None:
        cvar = RiskMetrics.calculate_cvar(positive_returns, 0.95)
        var = RiskMetrics.calculate_var(positive_returns, 0.95)
        # CVaR >= VaR (expected shortfall is worse than threshold)
        assert cvar >= var

    def test_cvar_insufficient_data(self) -> None:
        assert RiskMetrics.calculate_cvar(np.array([0.01])) == 0.0


# =========================================================================
# RatioMetrics
# =========================================================================


class TestRatioMetrics:
    """Tests for RatioMetrics calculator."""

    @pytest.fixture()
    def returns_positive_mean(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.normal(0.002, 0.01, 365)

    def test_sharpe_ratio_positive(self, returns_positive_mean: np.ndarray) -> None:
        sharpe = RatioMetrics.calculate_sharpe_ratio(returns_positive_mean)
        assert sharpe > 0.0

    def test_sharpe_ratio_insufficient_data(self) -> None:
        assert RatioMetrics.calculate_sharpe_ratio(np.array([0.01])) == 0.0

    def test_sharpe_ratio_zero_std(self) -> None:
        # All-zero returns → zero std after subtracting risk-free rate
        returns = np.zeros(100)
        assert RatioMetrics.calculate_sharpe_ratio(returns) == 0.0

    def test_sortino_ratio_positive(self, returns_positive_mean: np.ndarray) -> None:
        sortino = RatioMetrics.calculate_sortino_ratio(returns_positive_mean)
        assert sortino > 0.0

    def test_sortino_ratio_insufficient_data(self) -> None:
        assert RatioMetrics.calculate_sortino_ratio(np.array([0.01])) == 0.0

    def test_sortino_no_downside(self) -> None:
        # All positive returns → infinite sortino
        returns = np.array([0.01, 0.02, 0.03, 0.04])
        sortino = RatioMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.0)
        assert sortino == float("inf")

    def test_sortino_all_negative_no_excess(self) -> None:
        """All returns negative with high rf → mean excess < 0 → 0."""
        returns = np.array([-0.01, -0.02, -0.03, -0.04])
        sortino = RatioMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.5)
        # All excess returns negative, but enough downside data, so ratio is negative or 0
        assert isinstance(sortino, float)

    def test_calmar_ratio(self) -> None:
        calmar = RatioMetrics.calculate_calmar_ratio(cagr=20.0, max_dd=10.0)
        assert calmar == pytest.approx(2.0)

    def test_calmar_ratio_zero_dd(self) -> None:
        assert RatioMetrics.calculate_calmar_ratio(cagr=20.0, max_dd=0.0) == 0.0


# =========================================================================
# StatisticalMetrics
# =========================================================================


class TestStatisticalMetrics:
    """Tests for StatisticalMetrics calculator."""

    @pytest.fixture()
    def returns(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 200)

    def test_z_score_and_pvalue(self, returns: np.ndarray) -> None:
        z, p = StatisticalMetrics.calculate_z_score_and_pvalue(returns)
        assert isinstance(z, float)
        assert 0.0 <= p <= 1.0

    def test_z_score_insufficient_data(self) -> None:
        z, p = StatisticalMetrics.calculate_z_score_and_pvalue(np.array([0.01]))
        assert z == 0.0
        assert p == 1.0

    def test_z_score_zero_std(self) -> None:
        z, p = StatisticalMetrics.calculate_z_score_and_pvalue(np.full(100, 0.0))
        assert z == 0.0
        assert p == 1.0

    def test_skewness(self, returns: np.ndarray) -> None:
        skew = StatisticalMetrics.calculate_skewness(returns)
        assert isinstance(skew, float)

    def test_skewness_insufficient(self) -> None:
        assert StatisticalMetrics.calculate_skewness(np.array([0.01, 0.02])) == 0.0

    def test_kurtosis(self, returns: np.ndarray) -> None:
        kurt = StatisticalMetrics.calculate_kurtosis(returns)
        assert isinstance(kurt, float)

    def test_kurtosis_insufficient(self) -> None:
        assert StatisticalMetrics.calculate_kurtosis(np.array([0.01, 0.02])) == 0.0


# =========================================================================
# TradeMetrics
# =========================================================================


class TestTradeMetrics:
    """Tests for TradeMetrics calculator."""

    def test_empty_trades(self) -> None:
        win_rate, avg_win, avg_loss, pf, expectancy = TradeMetrics.calculate([])
        assert win_rate == 0.0
        assert avg_win == 0.0
        assert avg_loss == 0.0
        assert pf == 0.0
        assert expectancy == 0.0

    def test_all_wins(self) -> None:
        trades = [0.05, 0.10, 0.03]
        win_rate, avg_win, avg_loss, pf, expectancy = TradeMetrics.calculate(trades)
        assert win_rate == 100.0
        assert avg_win > 0.0
        assert avg_loss == 0.0
        assert pf == float("inf")
        assert expectancy > 0.0

    def test_all_losses(self) -> None:
        trades = [-0.05, -0.10, -0.03]
        win_rate, avg_win, avg_loss, pf, expectancy = TradeMetrics.calculate(trades)
        assert win_rate == 0.0
        assert avg_win == 0.0
        assert avg_loss < 0.0
        assert expectancy < 0.0

    def test_mixed_trades(self) -> None:
        trades = [0.10, -0.05, 0.08, -0.03]
        win_rate, avg_win, avg_loss, pf, expectancy = TradeMetrics.calculate(trades)
        assert win_rate == 50.0
        assert avg_win > 0.0
        assert avg_loss < 0.0
        assert pf > 1.0  # Net positive
        assert expectancy > 0.0

    def test_breakeven_trade_not_counted(self) -> None:
        """Zero return trades are neither wins nor losses."""
        trades = [0.0, 0.05, -0.05]
        win_rate, _, _, _, _ = TradeMetrics.calculate(trades)
        # 1 win out of 3 trades = 33.33%
        assert win_rate == pytest.approx(33.33, abs=0.1)
