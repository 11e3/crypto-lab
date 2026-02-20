"""Tests for backtester.analysis.monte_carlo_metrics."""

import numpy as np
import pytest

from src.backtester.analysis.monte_carlo_metrics import (
    calculate_percentiles,
    calculate_simulation_metrics,
    calculate_statistics,
)

_N_SIM = 10
_N_PERIODS = 50
_INITIAL = 1_000_000.0


def _make_equities(growth: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Return (equities, returns) with shape (N_SIM, N_PERIODS)."""
    equities = np.cumprod(np.full((_N_SIM, _N_PERIODS), 1 + growth), axis=1) * _INITIAL
    returns = np.diff(equities, axis=1) / equities[:, :-1]
    return equities, returns


class TestCalculateSimulationMetrics:
    def test_output_shapes(self) -> None:
        equities, returns = _make_equities()
        cagrs, mdds, sharpes = calculate_simulation_metrics(equities, returns, _INITIAL)
        assert cagrs.shape == (_N_SIM,)
        assert mdds.shape == (_N_SIM,)
        assert sharpes.shape == (_N_SIM,)

    def test_positive_cagr_for_growing_curve(self) -> None:
        equities, returns = _make_equities(growth=0.005)
        cagrs, _, _ = calculate_simulation_metrics(equities, returns, _INITIAL)
        assert np.all(cagrs > 0)

    def test_mdd_non_positive(self) -> None:
        equities, returns = _make_equities()
        _, mdds, _ = calculate_simulation_metrics(equities, returns, _INITIAL)
        assert np.all(mdds <= 0)


class TestCalculateStatistics:
    def test_required_keys_present(self) -> None:
        cagrs = np.array([0.1, 0.2, 0.15])
        mdds = np.array([-0.1, -0.2, -0.15])
        sharpes = np.array([1.0, 1.5, 1.2])
        stats = calculate_statistics(cagrs, mdds, sharpes)
        for key in ("mean_cagr", "std_cagr", "mean_mdd", "mean_sharpe", "cagr_ci_lower", "cagr_ci_upper"):
            assert key in stats, f"Missing key: {key}"

    def test_mean_values_correct(self) -> None:
        cagrs = np.array([0.1, 0.2, 0.3])
        mdds = np.array([-0.1, -0.2, -0.3])
        sharpes = np.array([1.0, 2.0, 3.0])
        stats = calculate_statistics(cagrs, mdds, sharpes)
        assert stats["mean_cagr"] == pytest.approx(0.2, abs=1e-9)
        assert stats["mean_sharpe"] == pytest.approx(2.0, abs=1e-9)


class TestCalculatePercentiles:
    def test_known_percentiles(self) -> None:
        data = np.arange(100, dtype=float)
        percs = calculate_percentiles(data)
        assert percs[50] == pytest.approx(49.5, abs=1.0)

    def test_returns_dict_with_standard_keys(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        percs = calculate_percentiles(data)
        assert isinstance(percs, dict)
        assert len(percs) > 0
