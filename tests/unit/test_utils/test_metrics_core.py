"""Tests for centralized metric calculation functions."""

import numpy as np
import pytest

from src.utils.metrics_core import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_daily_returns,
    calculate_drawdown_series,
    calculate_mdd,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


class TestCalculateDrawdownSeries:
    def test_basic(self) -> None:
        equity = np.array([100.0, 110.0, 105.0, 115.0])
        dd = calculate_drawdown_series(equity)
        assert dd[0] == 0.0
        assert dd[1] == 0.0
        assert dd[2] == pytest.approx(5.0 / 110.0)
        assert dd[3] == 0.0

    def test_single_element(self) -> None:
        dd = calculate_drawdown_series(np.array([100.0]))
        assert len(dd) == 1
        assert dd[0] == 0.0

    def test_empty(self) -> None:
        dd = calculate_drawdown_series(np.array([]))
        assert len(dd) == 0


class TestCalculateMDD:
    def test_basic(self) -> None:
        equity = np.array([100.0, 110.0, 88.0, 95.0])
        mdd = calculate_mdd(equity)
        assert mdd == pytest.approx(22.0 / 110.0 * 100)

    def test_no_drawdown(self) -> None:
        equity = np.array([100.0, 105.0, 110.0, 115.0])
        assert calculate_mdd(equity) == pytest.approx(0.0)

    def test_short_curve(self) -> None:
        assert calculate_mdd(np.array([100.0])) == 0.0
        assert calculate_mdd(np.array([])) == 0.0


class TestCalculateDailyReturns:
    def test_basic(self) -> None:
        equity = np.array([100.0, 110.0, 105.0])
        returns = calculate_daily_returns(equity)
        assert len(returns) == 2
        assert returns[0] == pytest.approx(0.1)
        assert returns[1] == pytest.approx(-5.0 / 110.0)

    def test_prepend_zero(self) -> None:
        equity = np.array([100.0, 110.0, 105.0])
        returns = calculate_daily_returns(equity, prepend_zero=True)
        assert len(returns) == 3
        assert returns[0] == 0.0
        assert returns[1] == pytest.approx(0.1)

    def test_single_element(self) -> None:
        returns = calculate_daily_returns(np.array([100.0]))
        assert len(returns) == 0
        returns_p = calculate_daily_returns(np.array([100.0]), prepend_zero=True)
        assert len(returns_p) == 1
        assert returns_p[0] == 0.0

    def test_empty(self) -> None:
        returns = calculate_daily_returns(np.array([]))
        assert len(returns) == 0


class TestCalculateSharpeRatio:
    def test_positive_returns(self) -> None:
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.005])
        sharpe = calculate_sharpe_ratio(returns, annualization_factor=252)
        assert sharpe > 0

    def test_zero_std(self) -> None:
        returns = np.array([0.01, 0.01, 0.01])
        assert calculate_sharpe_ratio(returns) == 0.0

    def test_empty(self) -> None:
        assert calculate_sharpe_ratio(np.array([])) == 0.0

    def test_annualization_factor(self) -> None:
        returns = np.array([0.01, 0.02, -0.005, 0.015])
        sharpe_252 = calculate_sharpe_ratio(returns, annualization_factor=252)
        sharpe_365 = calculate_sharpe_ratio(returns, annualization_factor=365)
        assert sharpe_365 > sharpe_252


class TestCalculateCAGR:
    def test_positive_return(self) -> None:
        cagr = calculate_cagr(1000.0, 1100.0, 365)
        assert cagr == pytest.approx(10.0)

    def test_two_years(self) -> None:
        cagr = calculate_cagr(1000.0, 1210.0, 730)
        assert cagr == pytest.approx(10.0, abs=0.1)

    def test_zero_days(self) -> None:
        assert calculate_cagr(1000.0, 1100.0, 0) == 0.0

    def test_zero_initial(self) -> None:
        assert calculate_cagr(0.0, 1100.0, 365) == 0.0

    def test_negative_final(self) -> None:
        assert calculate_cagr(1000.0, -100.0, 365) == -100.0

    def test_zero_final(self) -> None:
        assert calculate_cagr(1000.0, 0.0, 365) == -100.0


class TestCalculateCalmarRatio:
    def test_basic(self) -> None:
        assert calculate_calmar_ratio(10.0, 5.0) == pytest.approx(2.0)

    def test_zero_mdd(self) -> None:
        assert calculate_calmar_ratio(10.0, 0.0) == 0.0

    def test_negative_cagr(self) -> None:
        assert calculate_calmar_ratio(-5.0, 10.0) == pytest.approx(-0.5)


class TestCalculateSortinoRatio:
    def test_positive_returns(self) -> None:
        returns = np.array([0.01, 0.02, -0.005, 0.015, -0.01])
        sortino = calculate_sortino_ratio(returns)
        assert sortino > 0

    def test_no_downside(self) -> None:
        returns = np.array([0.01, 0.02, 0.015])
        assert calculate_sortino_ratio(returns) == 0.0

    def test_empty(self) -> None:
        assert calculate_sortino_ratio(np.array([])) == 0.0
