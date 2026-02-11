"""Tests for the extended metrics calculator service."""

from __future__ import annotations

import numpy as np
import pytest

from src.web.services.metrics_calculator import (
    ExtendedMetrics,
    calculate_extended_metrics,
)


class TestEmptyMetrics:
    """Tests for empty/edge-case input."""

    def test_single_element_returns_empty(self) -> None:
        metrics = calculate_extended_metrics(np.array([100.0]))
        assert isinstance(metrics, ExtendedMetrics)
        assert metrics.total_return_pct == 0.0
        assert metrics.trading_days == 0
        assert metrics.p_value == 1.0

    def test_empty_array_returns_empty(self) -> None:
        metrics = calculate_extended_metrics(np.array([]))
        assert metrics.total_return_pct == 0.0
        assert metrics.years == 0.0
        assert metrics.num_trades == 0


class TestCalculateExtendedMetrics:
    """Tests for calculate_extended_metrics."""

    @pytest.fixture()
    def growing_equity(self) -> np.ndarray:
        """Steady growth equity curve (100 -> ~150)."""
        np.random.seed(42)
        n = 365
        daily_return = 1 + 0.001  # ~0.1% daily
        equity = 100.0 * np.cumprod(np.full(n, daily_return))
        return np.concatenate([[100.0], equity])

    @pytest.fixture()
    def declining_equity(self) -> np.ndarray:
        """Declining equity curve."""
        np.random.seed(42)
        n = 365
        daily_return = 1 - 0.001
        equity = 100.0 * np.cumprod(np.full(n, daily_return))
        return np.concatenate([[100.0], equity])

    def test_positive_return(self, growing_equity: np.ndarray) -> None:
        metrics = calculate_extended_metrics(growing_equity)
        assert metrics.total_return_pct > 0.0
        assert metrics.cagr_pct > 0.0
        assert metrics.trading_days == len(growing_equity)

    def test_negative_return(self, declining_equity: np.ndarray) -> None:
        metrics = calculate_extended_metrics(declining_equity)
        assert metrics.total_return_pct < 0.0

    def test_max_drawdown_is_non_negative(self, growing_equity: np.ndarray) -> None:
        metrics = calculate_extended_metrics(growing_equity)
        assert metrics.max_drawdown_pct >= 0.0

    def test_volatility_positive(self, growing_equity: np.ndarray) -> None:
        """Volatility should be zero for constant returns."""
        # Constant daily return -> zero std -> zero vol
        equity = 100.0 * np.cumprod(np.full(100, 1.001))
        equity = np.concatenate([[100.0], equity])
        metrics = calculate_extended_metrics(equity)
        # Volatility is from std of returns; constant returns -> 0 vol
        assert metrics.volatility_pct == pytest.approx(0.0, abs=1e-6)

    def test_var_and_cvar_computed(self, growing_equity: np.ndarray) -> None:
        metrics = calculate_extended_metrics(growing_equity)
        assert metrics.var_95_pct >= 0.0
        assert metrics.var_99_pct >= 0.0
        assert metrics.cvar_95_pct >= 0.0
        assert metrics.cvar_99_pct >= 0.0

    def test_sharpe_sortino_calmar(self, growing_equity: np.ndarray) -> None:
        metrics = calculate_extended_metrics(growing_equity)
        # Growing equity should have positive risk-adjusted returns
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert isinstance(metrics.calmar_ratio, float)

    def test_statistical_metrics(self, growing_equity: np.ndarray) -> None:
        metrics = calculate_extended_metrics(growing_equity)
        assert isinstance(metrics.z_score, float)
        assert 0.0 <= metrics.p_value <= 1.0
        assert isinstance(metrics.skewness, float)
        assert isinstance(metrics.kurtosis, float)

    def test_trade_returns_processed(self) -> None:
        equity = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
        trades = [0.05, -0.02, 0.07]
        metrics = calculate_extended_metrics(equity, trade_returns=trades)
        assert metrics.num_trades == 3
        assert metrics.win_rate_pct > 0.0

    def test_no_trade_returns(self) -> None:
        equity = np.array([100.0, 105.0, 110.0])
        metrics = calculate_extended_metrics(equity, trade_returns=None)
        assert metrics.num_trades == 0
        assert metrics.win_rate_pct == 0.0

    def test_dates_parameter_for_years(self) -> None:
        import pandas as pd

        equity = np.array([100.0, 110.0, 120.0])
        dates = np.array(pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]))
        metrics = calculate_extended_metrics(equity, dates=dates)
        # Should be ~2 years
        assert metrics.years == pytest.approx(2.0, abs=0.05)

    def test_years_fallback_without_dates(self) -> None:
        equity = np.linspace(100, 200, 366)
        metrics = calculate_extended_metrics(equity)
        # 366 trading days / 365 ≈ 1.0 year
        assert metrics.years == pytest.approx(1.0, abs=0.1)

    def test_frozen_dataclass(self) -> None:
        metrics = calculate_extended_metrics(np.array([100.0, 110.0]))
        with pytest.raises(AttributeError):
            metrics.total_return_pct = 99.9  # type: ignore[misc]
