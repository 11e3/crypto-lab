"""Tests for src/metrics/core.py - unified metrics calculations."""

import numpy as np
import pytest

from src.metrics.core import (
    ANNUALIZATION_FACTOR,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_cvar,
    calculate_downside_volatility,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_upside_volatility,
    calculate_var,
    calculate_volatility,
)


class TestCalculateReturns:
    """Tests for calculate_returns function."""

    def test_normal_returns(self) -> None:
        """Test returns calculation with normal equity curve."""
        equity = np.array([100.0, 110.0, 105.0, 115.0])
        returns = calculate_returns(equity)

        expected = np.array([0.1, -0.04545454545454545, 0.0952380952380952])
        np.testing.assert_array_almost_equal(returns, expected)

    def test_empty_array(self) -> None:
        """Test with empty array."""
        equity = np.array([])
        returns = calculate_returns(equity)
        assert len(returns) == 0

    def test_single_value(self) -> None:
        """Test with single value (returns empty)."""
        equity = np.array([100.0])
        returns = calculate_returns(equity)
        assert len(returns) == 0

    def test_division_by_zero_protection(self) -> None:
        """Test that division by zero is handled."""
        equity = np.array([0.0, 100.0, 110.0])
        returns = calculate_returns(equity)
        # First return should be nan due to division by zero
        assert np.isnan(returns[0])
        assert not np.isnan(returns[1])


class TestCalculateTotalReturn:
    """Tests for calculate_total_return function."""

    def test_positive_return(self) -> None:
        """Test positive return calculation."""
        result = calculate_total_return(100.0, 150.0)
        assert result == pytest.approx(50.0)

    def test_negative_return(self) -> None:
        """Test negative return calculation."""
        result = calculate_total_return(100.0, 80.0)
        assert result == pytest.approx(-20.0)

    def test_zero_initial_value(self) -> None:
        """Test with zero initial value."""
        result = calculate_total_return(0.0, 100.0)
        assert result == 0.0

    def test_negative_initial_value(self) -> None:
        """Test with negative initial value."""
        result = calculate_total_return(-100.0, 100.0)
        assert result == 0.0


class TestCalculateCAGR:
    """Tests for calculate_cagr function."""

    def test_one_year_growth(self) -> None:
        """Test CAGR for one year period."""
        result = calculate_cagr(100.0, 110.0, 365)
        assert result == pytest.approx(10.0)

    def test_two_year_growth(self) -> None:
        """Test CAGR for two year period."""
        # 100 -> 121 over 2 years = 10% CAGR
        result = calculate_cagr(100.0, 121.0, 730)
        assert result == pytest.approx(10.0, rel=0.01)

    def test_zero_days(self) -> None:
        """Test with zero days."""
        result = calculate_cagr(100.0, 150.0, 0)
        assert result == 0.0

    def test_zero_initial_value(self) -> None:
        """Test with zero initial value."""
        result = calculate_cagr(0.0, 150.0, 365)
        assert result == 0.0

    def test_zero_final_value(self) -> None:
        """Test with zero final value."""
        result = calculate_cagr(100.0, 0.0, 365)
        assert result == 0.0

    def test_negative_days(self) -> None:
        """Test with negative days."""
        result = calculate_cagr(100.0, 150.0, -365)
        assert result == 0.0


class TestCalculateMaxDrawdown:
    """Tests for calculate_max_drawdown function."""

    def test_normal_drawdown(self) -> None:
        """Test max drawdown calculation."""
        equity = np.array([100.0, 120.0, 90.0, 110.0])
        result = calculate_max_drawdown(equity)
        # MDD = (120 - 90) / 120 = 25%
        assert result == pytest.approx(25.0)

    def test_no_drawdown(self) -> None:
        """Test with monotonically increasing equity."""
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        result = calculate_max_drawdown(equity)
        assert result == pytest.approx(0.0)

    def test_single_value(self) -> None:
        """Test with single value."""
        equity = np.array([100.0])
        result = calculate_max_drawdown(equity)
        assert result == 0.0

    def test_empty_array(self) -> None:
        """Test with empty array."""
        equity = np.array([])
        result = calculate_max_drawdown(equity)
        assert result == 0.0

    def test_zero_values_protection(self) -> None:
        """Test protection against division by zero in cummax."""
        equity = np.array([0.0, 100.0, 90.0])
        result = calculate_max_drawdown(equity)
        # Should handle the zero gracefully
        assert not np.isnan(result)


class TestCalculateVolatility:
    """Tests for calculate_volatility function."""

    def test_annualized_volatility(self) -> None:
        """Test annualized volatility calculation."""
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        result = calculate_volatility(returns, annualize=True)

        expected_daily_vol = np.std(returns, ddof=1)
        expected_annual = expected_daily_vol * np.sqrt(ANNUALIZATION_FACTOR) * 100
        assert result == pytest.approx(expected_annual)

    def test_non_annualized_volatility(self) -> None:
        """Test non-annualized volatility calculation."""
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        result = calculate_volatility(returns, annualize=False)

        expected = np.std(returns, ddof=1) * 100
        assert result == pytest.approx(expected)

    def test_single_return(self) -> None:
        """Test with single return value."""
        returns = np.array([0.01])
        result = calculate_volatility(returns)
        assert result == 0.0

    def test_empty_returns(self) -> None:
        """Test with empty returns."""
        returns = np.array([])
        result = calculate_volatility(returns)
        assert result == 0.0

    def test_custom_annualization_factor(self) -> None:
        """Test with custom annualization factor."""
        returns = np.array([0.01, -0.02, 0.015])
        result = calculate_volatility(returns, annualize=True, annualization_factor=252)

        expected = np.std(returns, ddof=1) * np.sqrt(252) * 100
        assert result == pytest.approx(expected)


class TestCalculateSharpeRatio:
    """Tests for calculate_sharpe_ratio function."""

    def test_positive_sharpe(self) -> None:
        """Test positive Sharpe ratio."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        result = calculate_sharpe_ratio(returns)
        assert result > 0

    def test_negative_sharpe(self) -> None:
        """Test negative Sharpe ratio."""
        returns = np.array([-0.01, -0.02, -0.015, -0.01, -0.02])
        result = calculate_sharpe_ratio(returns)
        assert result < 0

    def test_with_risk_free_rate(self) -> None:
        """Test with non-zero risk-free rate."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        result_no_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        result_with_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.05)
        # Higher risk-free rate should lower the Sharpe ratio
        assert result_with_rf < result_no_rf

    def test_zero_volatility(self) -> None:
        """Test with zero volatility (constant returns)."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        result = calculate_sharpe_ratio(returns)
        assert result == 0.0

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        returns = np.array([0.01])
        result = calculate_sharpe_ratio(returns)
        assert result == 0.0


class TestCalculateSortinoRatio:
    """Tests for calculate_sortino_ratio function."""

    def test_positive_sortino(self) -> None:
        """Test positive Sortino ratio."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, -0.01])
        result = calculate_sortino_ratio(returns)
        assert result > 0

    def test_all_positive_returns(self) -> None:
        """Test with all positive returns (no downside)."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        result = calculate_sortino_ratio(returns)
        # Should return infinity for positive mean with no downside
        assert result == float("inf")

    def test_all_negative_returns(self) -> None:
        """Test with all negative returns."""
        returns = np.array([-0.01, -0.02, -0.015, -0.01, -0.02])
        result = calculate_sortino_ratio(returns)
        assert result < 0

    def test_zero_downside_volatility(self) -> None:
        """Test when downside returns have zero volatility."""
        returns = np.array([0.01, -0.01, 0.02, -0.01, 0.015])
        result = calculate_sortino_ratio(returns)
        # When all downside returns are the same, std=0
        assert result == 0.0

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        returns = np.array([0.01])
        result = calculate_sortino_ratio(returns)
        assert result == 0.0


class TestCalculateCalmarRatio:
    """Tests for calculate_calmar_ratio function."""

    def test_positive_calmar(self) -> None:
        """Test positive Calmar ratio."""
        result = calculate_calmar_ratio(20.0, 10.0)
        assert result == pytest.approx(2.0)

    def test_zero_drawdown(self) -> None:
        """Test with zero drawdown."""
        result = calculate_calmar_ratio(20.0, 0.0)
        assert result == 0.0

    def test_negative_cagr(self) -> None:
        """Test with negative CAGR."""
        result = calculate_calmar_ratio(-10.0, 5.0)
        assert result == pytest.approx(-2.0)


class TestCalculateVaR:
    """Tests for calculate_var function."""

    def test_var_95(self) -> None:
        """Test 95% VaR calculation."""
        returns = np.array([-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        result = calculate_var(returns, confidence=0.95)
        # VaR should be positive and reasonable
        assert result > 0
        assert result < 10  # Less than 10%

    def test_var_99(self) -> None:
        """Test 99% VaR calculation."""
        returns = np.array([-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        result = calculate_var(returns, confidence=0.99)
        # Higher confidence = higher VaR
        var_95 = calculate_var(returns, confidence=0.95)
        assert result >= var_95

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        returns = np.array([0.01])
        result = calculate_var(returns)
        assert result == 0.0


class TestCalculateCVaR:
    """Tests for calculate_cvar function."""

    def test_cvar_95(self) -> None:
        """Test 95% CVaR calculation."""
        returns = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        result = calculate_cvar(returns, confidence=0.95)
        # CVaR should be >= VaR
        var = calculate_var(returns, confidence=0.95)
        assert result >= var

    def test_cvar_with_mostly_positive(self) -> None:
        """Test CVaR with mostly positive returns."""
        returns = np.array([-0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        result = calculate_cvar(returns, confidence=0.95)
        # Should return a reasonable positive value
        assert result >= 0

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        returns = np.array([0.01])
        result = calculate_cvar(returns)
        assert result == 0.0


class TestCalculateDownsideVolatility:
    """Tests for calculate_downside_volatility function."""

    def test_normal_downside_volatility(self) -> None:
        """Test downside volatility calculation."""
        returns = np.array([0.01, -0.02, 0.015, -0.03, 0.02, -0.01])
        result = calculate_downside_volatility(returns)
        assert result > 0

    def test_no_downside_returns(self) -> None:
        """Test with no negative returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.03])
        result = calculate_downside_volatility(returns)
        assert result == 0.0

    def test_single_downside_return(self) -> None:
        """Test with single negative return."""
        returns = np.array([0.01, -0.02, 0.015, 0.03])
        result = calculate_downside_volatility(returns)
        assert result == 0.0

    def test_non_annualized(self) -> None:
        """Test non-annualized downside volatility."""
        returns = np.array([-0.01, -0.02, -0.015, -0.03])
        result_ann = calculate_downside_volatility(returns, annualize=True)
        result_daily = calculate_downside_volatility(returns, annualize=False)
        assert result_ann > result_daily

    def test_custom_mar(self) -> None:
        """Test with custom minimum acceptable return."""
        returns = np.array([0.01, -0.02, 0.015, -0.03, 0.005])
        result_0 = calculate_downside_volatility(returns, mar=0.0)
        result_01 = calculate_downside_volatility(returns, mar=0.01)
        # Higher MAR means more returns count as "downside"
        assert result_01 >= result_0

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        returns = np.array([0.01])
        result = calculate_downside_volatility(returns)
        assert result == 0.0


class TestCalculateUpsideVolatility:
    """Tests for calculate_upside_volatility function."""

    def test_normal_upside_volatility(self) -> None:
        """Test upside volatility calculation."""
        returns = np.array([0.01, -0.02, 0.015, -0.03, 0.02, 0.03])
        result = calculate_upside_volatility(returns)
        assert result > 0

    def test_no_upside_returns(self) -> None:
        """Test with no positive returns."""
        returns = np.array([-0.01, -0.02, -0.015, -0.03])
        result = calculate_upside_volatility(returns)
        assert result == 0.0

    def test_single_upside_return(self) -> None:
        """Test with single positive return."""
        returns = np.array([-0.01, 0.02, -0.015, -0.03])
        result = calculate_upside_volatility(returns)
        assert result == 0.0

    def test_non_annualized(self) -> None:
        """Test non-annualized upside volatility."""
        returns = np.array([0.01, 0.02, 0.015, 0.03])
        result_ann = calculate_upside_volatility(returns, annualize=True)
        result_daily = calculate_upside_volatility(returns, annualize=False)
        assert result_ann > result_daily
