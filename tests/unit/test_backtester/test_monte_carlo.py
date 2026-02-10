"""
Unit tests for Monte Carlo simulation module.
"""

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.backtester.analysis.monte_carlo import (
    MonteCarloResult,
    MonteCarloSimulator,
    run_monte_carlo,
)
from src.backtester.engine import BacktestConfig, BacktestResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def mock_backtest_result_for_mc() -> BacktestResult:
    """Mock BacktestResult instance suitable for Monte Carlo."""
    result = MagicMock(spec=BacktestResult)
    result.config = BacktestConfig(initial_capital=1000000)
    result.equity_curve = np.array(
        [1_000_000, 1_010_000, 1_005_000, 1_020_000, 1_015_000, 1_030_000]
    )
    result.dates = np.array(
        [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
            datetime.date(2023, 1, 4),
            datetime.date(2023, 1, 5),
            datetime.date(2023, 1, 6),
        ]
    )
    return result


@pytest.fixture
def mock_empty_backtest_result_for_mc() -> BacktestResult:
    """Mock BacktestResult instance with empty equity curve for Monte Carlo."""
    result = MagicMock(spec=BacktestResult)
    result.config = BacktestConfig(initial_capital=1000000)
    result.equity_curve = np.array([])
    result.dates = np.array([])
    return result


class TestMonteCarloResult:
    """Tests for MonteCarloResult dataclass."""

    def test_initialization(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        mc_result = MonteCarloResult(
            original_result=mock_backtest_result_for_mc,
            n_simulations=10,
            simulated_returns=np.zeros((10, 5)),
            simulated_cagrs=np.ones(10) * 10,
            simulated_mdds=np.ones(10) * 5,
            simulated_sharpes=np.ones(10) * 1.0,
            mean_cagr=10.0,
            std_cagr=1.0,
            mean_mdd=5.0,
            std_mdd=0.5,
            mean_sharpe=1.0,
            std_sharpe=0.1,
            cagr_ci_lower=9.0,
            cagr_ci_upper=11.0,
            mdd_ci_lower=4.0,
            mdd_ci_upper=6.0,
            sharpe_ci_lower=0.9,
            sharpe_ci_upper=1.1,
            cagr_percentiles={5: 9.5, 50: 10.0, 95: 10.5},
            mdd_percentiles={5: 4.5, 50: 5.0, 95: 5.5},
        )
        assert mc_result.n_simulations == 10
        assert mc_result.mean_cagr == 10.0

    def test_repr(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        mc_result = MonteCarloResult(
            original_result=mock_backtest_result_for_mc,
            n_simulations=10,
            simulated_returns=np.zeros((10, 5)),
            simulated_cagrs=np.ones(10) * 10,
            simulated_mdds=np.ones(10) * 5,
            simulated_sharpes=np.ones(10) * 1.0,
            mean_cagr=10.0,
            std_cagr=1.0,
            mean_mdd=5.0,
            std_mdd=0.5,
            mean_sharpe=1.0,
            std_sharpe=0.1,
            cagr_ci_lower=9.0,
            cagr_ci_upper=11.0,
            mdd_ci_lower=4.0,
            mdd_ci_upper=6.0,
            sharpe_ci_lower=0.9,
            sharpe_ci_upper=1.1,
            cagr_percentiles={5: 9.5, 50: 10.0, 95: 10.5},
            mdd_percentiles={5: 4.5, 50: 5.0, 95: 5.5},
        )
        assert repr(mc_result) == "MonteCarloResult(n=10, mean_cagr=10.00%)"


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    def test_init_from_config_capital(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        assert simulator.initial_capital == 1000000
        expected_returns = np.array([0.01, -0.004950495, 0.014925373, -0.004901961, 0.014778325])
        assert np.allclose(simulator.daily_returns, expected_returns)

    def test_init_explicit_capital(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc, initial_capital=2000000)
        assert simulator.initial_capital == 2000000

    def test_init_empty_equity_curve(
        self, mock_empty_backtest_result_for_mc: BacktestResult
    ) -> None:
        with patch("src.backtester.analysis.monte_carlo.logger.warning") as mock_warning:
            simulator = MonteCarloSimulator(mock_empty_backtest_result_for_mc)
            assert simulator.initial_capital == 1000000  # Falls back to default 1.0
            assert np.allclose(simulator.daily_returns, np.array([0.0]))
            mock_warning.assert_called_once_with(
                "No valid returns found for Monte Carlo simulation"
            )

    def test_init_single_point_equity_curve(self) -> None:
        result = MagicMock(spec=BacktestResult)
        result.config = BacktestConfig(initial_capital=1000000)
        result.equity_curve = np.array([1_000_000])
        with patch("src.backtester.analysis.monte_carlo.logger.warning") as mock_warning:
            simulator = MonteCarloSimulator(result)
            assert simulator.initial_capital == 1_000_000
            assert np.allclose(simulator.daily_returns, np.array([0.0]))  # Should be 0.0
            mock_warning.assert_called_once_with(
                "No valid returns found for Monte Carlo simulation"
            )

    def test_bootstrap(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        n_simulations = 10
        n_periods = 5

        mock_rng = MagicMock()
        mock_rng.choice.return_value = np.tile(
            simulator.daily_returns[:n_periods], (n_simulations, 1)
        )
        simulator._rng = mock_rng

        simulated_returns = simulator._bootstrap(n_simulations, n_periods)
        assert isinstance(simulated_returns, np.ndarray)
        assert simulated_returns.shape == (n_simulations, n_periods)
        mock_rng.choice.assert_called_once()

    def test_parametric(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        n_simulations = 10
        n_periods = 5

        mock_rng = MagicMock()
        mock_rng.normal.return_value = np.zeros((n_simulations, n_periods))
        simulator._rng = mock_rng

        simulated_returns = simulator._parametric(n_simulations, n_periods)
        assert isinstance(simulated_returns, np.ndarray)
        assert simulated_returns.shape == (n_simulations, n_periods)
        mock_rng.normal.assert_called_once()

    def test_build_result(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        n_simulations = 3
        # Use simple returns to make calculations predictable
        simulated_returns = np.array(
            [
                [0.01, 0.01, 0.01],  # CAGR should be positive
                [-0.01, -0.01, -0.01],  # CAGR should be negative
                [0.02, -0.02, 0.01],  # Mixed
            ]
        )

        mc_result = simulator._build_result(simulated_returns, n_simulations)

        assert mc_result.simulated_cagrs.shape == (n_simulations,)
        assert mc_result.simulated_mdds.shape == (n_simulations,)
        assert mc_result.simulated_sharpes.shape == (n_simulations,)

        assert mc_result.mean_cagr == pytest.approx(np.mean(mc_result.simulated_cagrs))
        assert mc_result.std_cagr == pytest.approx(np.std(mc_result.simulated_cagrs))
        assert mc_result.cagr_ci_lower < mc_result.cagr_ci_upper
        assert 50 in mc_result.cagr_percentiles

    def test_build_result_zero_periods(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        n_simulations = 1
        simulated_returns = np.array([[]])  # Zero periods
        mc_result = simulator._build_result(simulated_returns, n_simulations)
        assert mc_result.simulated_cagrs[0] == pytest.approx(-100.0)  # Should be -100%
        assert mc_result.simulated_mdds[0] == 0.0  # No drawdown with 0 periods
        assert mc_result.simulated_sharpes[0] == 0.0  # No sharpe with 0 periods

    def test_simulate_bootstrap_method(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        n_simulations = 10
        n_periods = 5
        # Test that simulate runs without error with bootstrap method
        mc_result = simulator.simulate(n_simulations, n_periods, method="bootstrap")
        assert isinstance(mc_result, MonteCarloResult)
        assert mc_result.n_simulations == n_simulations
        assert mc_result.simulated_returns.shape == (n_simulations, n_periods)

    def test_simulate_parametric_method(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        n_simulations = 10
        n_periods = 5
        # Test that simulate runs without error with parametric method
        mc_result = simulator.simulate(n_simulations, n_periods, method="parametric")
        assert isinstance(mc_result, MonteCarloResult)
        assert mc_result.n_simulations == n_simulations
        assert mc_result.simulated_returns.shape == (n_simulations, n_periods)

    def test_simulate_unknown_method(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        with pytest.raises(ValueError, match="Unknown simulation method: invalid"):
            simulator.simulate(method="invalid")

    def test_simulate_random_seed(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator1 = MonteCarloSimulator(mock_backtest_result_for_mc)
        mc_result1 = simulator1.simulate(random_seed=42)
        simulator2 = MonteCarloSimulator(mock_backtest_result_for_mc)
        mc_result2 = simulator2.simulate(random_seed=42)
        assert np.array_equal(mc_result1.simulated_returns, mc_result2.simulated_returns)
        assert np.allclose(mc_result1.simulated_cagrs, mc_result2.simulated_cagrs)

    def test_probability_of_loss(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        mc_result = MagicMock(spec=MonteCarloResult)
        mc_result.simulated_cagrs = np.array([10, -5, 20, -10, 0])
        mc_result.n_simulations = 5
        assert simulator.probability_of_loss(mc_result) == 2 / 5

    def test_value_at_risk(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        mc_result = MagicMock(spec=MonteCarloResult)
        mc_result.simulated_cagrs = np.array([10, -5, 20, -10, 0, 30, -15, 5])
        assert simulator.value_at_risk(mc_result, confidence=0.95) == pytest.approx(
            np.percentile(mc_result.simulated_cagrs, 5)
        )  # 1 - 0.95 = 0.05 = 5th percentile

    def test_conditional_value_at_risk(self, mock_backtest_result_for_mc: BacktestResult) -> None:
        simulator = MonteCarloSimulator(mock_backtest_result_for_mc)
        mc_result = MagicMock(spec=MonteCarloResult)
        mc_result.simulated_cagrs = np.array([10, -5, 20, -10, 0, 30, -15, 5])
        assert simulator.conditional_value_at_risk(mc_result, confidence=0.95) == pytest.approx(
            -15.0
        )

    def test_get_initial_capital_no_equity(
        self, mock_empty_backtest_result_for_mc: BacktestResult
    ) -> None:
        """Test _get_initial_capital when equity curve is empty (line 67-68)."""
        # Remove config to force fallback to equity curve check
        mock_empty_backtest_result_for_mc.config = None
        simulator = MonteCarloSimulator(mock_empty_backtest_result_for_mc)

        # Should return 1.0 as default when no equity data and no config
        initial = simulator._get_initial_capital(mock_empty_backtest_result_for_mc, None)
        assert initial == 1.0

    def test_extract_returns_single_equity_value(self) -> None:
        """Test _extract_returns with single equity value (line 76->78)."""
        result = MagicMock(spec=BacktestResult)
        result.equity_curve = np.array([1000000])  # Only 1 value
        result.config = None

        simulator = MonteCarloSimulator(result)
        returns = simulator._extract_returns(result)

        # Should return [0.0] as default when insufficient data
        assert len(returns) == 1
        assert returns[0] == 0.0

    def test_bootstrap_empty_returns(self) -> None:
        """Test _bootstrap with empty returns (covers line 107)."""
        result = MagicMock(spec=BacktestResult)
        result.equity_curve = np.array([])
        result.config = None

        simulator = MonteCarloSimulator(result)
        # _bootstrap should return zeros when no returns
        simulated = simulator._bootstrap(n_simulations=10, n_periods=5)

        assert simulated.shape == (10, 5)
        assert np.all(simulated == 0)

    def test_parametric_empty_returns(self) -> None:
        """Test _parametric with empty returns (covers line 113)."""
        result = MagicMock(spec=BacktestResult)
        result.equity_curve = np.array([])
        result.config = None

        simulator = MonteCarloSimulator(result)
        # _parametric should use default mean=0.0, std=0.01 when no returns
        simulated = simulator._parametric(n_simulations=10, n_periods=5)

        assert simulated.shape == (10, 5)
        # Just verify it doesn't crash and returns proper shape


class TestRunMonteCarlo:
    """Tests for run_monte_carlo convenience function."""

    @patch("src.backtester.analysis.monte_carlo.MonteCarloSimulator")
    def test_run_monte_carlo(
        self, mock_monte_carlo_simulator: MagicMock, mock_backtest_result_for_mc: BacktestResult
    ) -> None:
        """Test run_monte_carlo convenience function."""
        # Mock simulator instance and its simulate method
        mock_simulator_instance = mock_monte_carlo_simulator.return_value
        mock_mc_result = MagicMock(spec=MonteCarloResult)
        mock_simulator_instance.simulate.return_value = mock_mc_result

        result = run_monte_carlo(
            result=mock_backtest_result_for_mc,
            n_simulations=500,
            method="parametric",
            random_seed=123,
        )

        mock_monte_carlo_simulator.assert_called_once_with(mock_backtest_result_for_mc)
        mock_simulator_instance.simulate.assert_called_once_with(
            n_simulations=500, method="parametric", random_seed=123
        )
        assert result == mock_mc_result
