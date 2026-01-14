"""Tests for src/backtester/html/html_risk.py - HTML risk metrics generation."""

import numpy as np
import pytest

from src.backtester.html.html_risk import (
    _generate_correlation_html,
    _generate_position_html,
    generate_risk_metrics_html,
)
from src.risk.metrics import PortfolioRiskMetrics


class TestGenerateRiskMetricsHtml:
    """Tests for generate_risk_metrics_html function."""

    def test_none_input(self) -> None:
        """Test with None input."""
        result = generate_risk_metrics_html(None)
        assert result == ""

    def test_basic_metrics(self) -> None:
        """Test with basic risk metrics."""
        metrics = PortfolioRiskMetrics(
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.07,
            cvar_99=0.10,
            portfolio_volatility=0.15,
            avg_correlation=None,
            max_correlation=None,
            min_correlation=None,
            max_position_pct=None,
            position_concentration=None,
        )
        result = generate_risk_metrics_html(metrics)

        assert "Risk Metrics" in result
        assert "VaR (95%)" in result
        assert "5.00%" in result
        assert "CVaR (95%)" in result
        assert "7.00%" in result


class TestGenerateCorrelationHtml:
    """Tests for _generate_correlation_html function."""

    def test_no_correlation(self) -> None:
        """Test without correlation data."""
        metrics = PortfolioRiskMetrics(
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.07,
            cvar_99=0.10,
            portfolio_volatility=0.15,
            avg_correlation=None,
            max_correlation=None,
            min_correlation=None,
            max_position_pct=None,
            position_concentration=None,
        )
        result = _generate_correlation_html(metrics)
        assert result == ""

    def test_nan_correlation(self) -> None:
        """Test with NaN correlation."""
        metrics = PortfolioRiskMetrics(
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.07,
            cvar_99=0.10,
            portfolio_volatility=0.15,
            avg_correlation=float("nan"),
            max_correlation=0.9,
            min_correlation=0.1,
            max_position_pct=None,
            position_concentration=None,
        )
        result = _generate_correlation_html(metrics)
        assert result == ""

    def test_valid_correlation(self) -> None:
        """Test with valid correlation data."""
        metrics = PortfolioRiskMetrics(
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.07,
            cvar_99=0.10,
            portfolio_volatility=0.15,
            avg_correlation=0.5,
            max_correlation=0.9,
            min_correlation=0.1,
            max_position_pct=None,
            position_concentration=None,
        )
        result = _generate_correlation_html(metrics)
        assert "Avg Correlation" in result
        assert "0.500" in result


class TestGeneratePositionHtml:
    """Tests for _generate_position_html function."""

    def test_no_position_data(self) -> None:
        """Test without position data."""
        metrics = PortfolioRiskMetrics(
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.07,
            cvar_99=0.10,
            portfolio_volatility=0.15,
            avg_correlation=None,
            max_correlation=None,
            min_correlation=None,
            max_position_pct=None,
            position_concentration=None,
        )
        result = _generate_position_html(metrics)
        assert result == ""

    def test_zero_position(self) -> None:
        """Test with zero max position."""
        metrics = PortfolioRiskMetrics(
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.07,
            cvar_99=0.10,
            portfolio_volatility=0.15,
            avg_correlation=None,
            max_correlation=None,
            min_correlation=None,
            max_position_pct=0.0,
            position_concentration=0.0,
        )
        result = _generate_position_html(metrics)
        assert result == ""

    def test_valid_position(self) -> None:
        """Test with valid position data."""
        metrics = PortfolioRiskMetrics(
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.07,
            cvar_99=0.10,
            portfolio_volatility=0.15,
            avg_correlation=None,
            max_correlation=None,
            min_correlation=None,
            max_position_pct=0.25,
            position_concentration=0.15,
        )
        result = _generate_position_html(metrics)
        assert "Max Position %" in result
        assert "25.00%" in result
