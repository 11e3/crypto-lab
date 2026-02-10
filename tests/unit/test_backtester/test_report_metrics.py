"""Tests for report_metrics module."""

import pytest

from src.utils.metrics_core import calculate_cagr


class TestReportMetrics:
    """Test cases for report_metrics functions."""

    def test_calculate_cagr_negative_final(self) -> None:
        """Test calculate_cagr with negative final value."""
        result = calculate_cagr(initial_value=1000.0, final_value=-100.0, total_days=365)
        assert result == -100.0

    def test_calculate_cagr_zero_ratio(self) -> None:
        """Test calculate_cagr edge case."""
        result = calculate_cagr(initial_value=1000.0, final_value=0.0, total_days=365)
        assert result == -100.0

    def test_calculate_cagr_normal(self) -> None:
        """Test calculate_cagr with normal values."""
        result = calculate_cagr(initial_value=1000.0, final_value=1100.0, total_days=365)
        # 10% return over 1 year = ~10% CAGR
        assert result == pytest.approx(10.0, abs=0.5)

    def test_calculate_cagr_zero_days(self) -> None:
        """Test calculate_cagr with zero days."""
        result = calculate_cagr(initial_value=1000.0, final_value=1100.0, total_days=0)
        assert result == 0.0
