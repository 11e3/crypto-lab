"""
Monte Carlo simulation for backtest results.

Provides Monte Carlo analysis to:
- Estimate future performance distribution
- Calculate confidence intervals
- Assess risk under various scenarios
- Validate strategy robustness
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.engine import BacktestResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    original_result: BacktestResult
    n_simulations: int
    simulated_returns: np.ndarray
    simulated_cagrs: np.ndarray
    simulated_mdds: np.ndarray
    simulated_sharpes: np.ndarray

    # Statistics
    mean_cagr: float
    std_cagr: float
    mean_mdd: float
    std_mdd: float
    mean_sharpe: float
    std_sharpe: float

    # Confidence intervals (95%)
    cagr_ci_lower: float
    cagr_ci_upper: float
    mdd_ci_lower: float
    mdd_ci_upper: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float

    # Percentiles
    cagr_percentiles: dict[float, float]
    mdd_percentiles: dict[float, float]

    def __repr__(self) -> str:
        return (
            f"MonteCarloResult(n_simulations={self.n_simulations}, "
            f"mean_cagr={self.mean_cagr:.2f}%)"
        )


class MonteCarloSimulator:
    """
    Monte Carlo simulator for backtest results.

    Simulates future performance based on historical return distribution.
    """

    def __init__(
        self,
        result: BacktestResult,
        initial_capital: float | None = None,
    ) -> None:
        """
        Initialize Monte Carlo simulator.

        Args:
            result: BacktestResult from historical backtest
            initial_capital: Initial capital (uses result's initial capital if None)
        """
        self.result = result
        # Get initial capital from config or use provided/default
        if initial_capital is not None:
            self.initial_capital = initial_capital
        elif result.config and hasattr(result.config, "initial_capital"):
            self.initial_capital = result.config.initial_capital
        else:
            # Default: use first value of equity curve
            if len(result.equity_curve) > 0:
                self.initial_capital = result.equity_curve[0]
            else:
                self.initial_capital = 1.0

        # Extract daily returns from equity curve
        if len(result.equity_curve) > 1:
            equity = result.equity_curve
            self.daily_returns = np.diff(equity) / equity[:-1]
            # Remove NaN and Inf values
            self.daily_returns = self.daily_returns[
                np.isfinite(self.daily_returns)
            ]
        else:
            self.daily_returns = np.array([])

        if len(self.daily_returns) == 0:
            logger.warning("No valid returns found for Monte Carlo simulation")
            self.daily_returns = np.array([0.0])

    def simulate(
        self,
        n_simulations: int = 1000,
        n_periods: int | None = None,
        method: str = "bootstrap",
        random_seed: int | None = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Args:
            n_simulations: Number of simulation runs
            n_periods: Number of periods to simulate (uses original length if None)
            method: Simulation method ('bootstrap' or 'parametric')
            random_seed: Random seed for reproducibility

        Returns:
            MonteCarloResult with simulation statistics
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if n_periods is None:
            n_periods = len(self.daily_returns)

        if method == "bootstrap":
            return self._bootstrap_simulation(n_simulations, n_periods)
        elif method == "parametric":
            return self._parametric_simulation(n_simulations, n_periods)
        else:
            raise ValueError(f"Unknown simulation method: {method}")

    def _bootstrap_simulation(
        self, n_simulations: int, n_periods: int
    ) -> MonteCarloResult:
        """Bootstrap method: resample from historical returns."""
        if len(self.daily_returns) == 0:
            # Fallback: use zero returns
            simulated_returns = np.zeros((n_simulations, n_periods))
        else:
            # Resample returns with replacement
            simulated_returns = np.random.choice(
                self.daily_returns, size=(n_simulations, n_periods), replace=True
            )

        return self._process_simulations(simulated_returns, n_simulations)

    def _parametric_simulation(
        self, n_simulations: int, n_periods: int
    ) -> MonteCarloResult:
        """Parametric method: sample from normal distribution."""
        if len(self.daily_returns) == 0:
            # Fallback: use zero mean, small std
            mean_return = 0.0
            std_return = 0.01
        else:
            mean_return = np.mean(self.daily_returns)
            std_return = np.std(self.daily_returns)

        # Sample from normal distribution
        simulated_returns = np.random.normal(
            mean_return, std_return, size=(n_simulations, n_periods)
        )

        return self._process_simulations(simulated_returns, n_simulations)

    def _process_simulations(
        self, simulated_returns: np.ndarray, n_simulations: int
    ) -> MonteCarloResult:
        """Process simulated returns and calculate statistics."""
        # Calculate equity curves for each simulation
        simulated_equities = np.zeros_like(simulated_returns)
        for i in range(n_simulations):
            equity = self.initial_capital
            for j, ret in enumerate(simulated_returns[i]):
                equity = equity * (1 + ret)
                simulated_equities[i, j] = equity

        # Calculate metrics for each simulation
        simulated_cagrs = np.zeros(n_simulations)
        simulated_mdds = np.zeros(n_simulations)
        simulated_sharpes = np.zeros(n_simulations)

        n_periods = simulated_returns.shape[1]
        annualization_factor = 365.0  # Assuming daily data

        for i in range(n_simulations):
            equity_curve = simulated_equities[i]
            returns = simulated_returns[i]

            # CAGR
            if equity_curve[-1] > 0 and n_periods > 0:
                simulated_cagrs[i] = (
                    (equity_curve[-1] / self.initial_capital) ** (annualization_factor / n_periods) - 1
                ) * 100
            else:
                simulated_cagrs[i] = -100.0

            # MDD
            cummax = np.maximum.accumulate(equity_curve)
            drawdown = (cummax - equity_curve) / cummax
            simulated_mdds[i] = np.nanmax(drawdown) * 100

            # Sharpe Ratio
            if len(returns) > 0 and np.std(returns) > 0:
                simulated_sharpes[i] = (
                    np.mean(returns) / np.std(returns)
                ) * np.sqrt(annualization_factor)
            else:
                simulated_sharpes[i] = 0.0

        # Calculate statistics
        mean_cagr = np.mean(simulated_cagrs)
        std_cagr = np.std(simulated_cagrs)
        mean_mdd = np.mean(simulated_mdds)
        std_mdd = np.std(simulated_mdds)
        mean_sharpe = np.mean(simulated_sharpes)
        std_sharpe = np.std(simulated_sharpes)

        # Confidence intervals (95%)
        cagr_ci_lower = np.percentile(simulated_cagrs, 2.5)
        cagr_ci_upper = np.percentile(simulated_cagrs, 97.5)
        mdd_ci_lower = np.percentile(simulated_mdds, 2.5)
        mdd_ci_upper = np.percentile(simulated_mdds, 97.5)
        sharpe_ci_lower = np.percentile(simulated_sharpes, 2.5)
        sharpe_ci_upper = np.percentile(simulated_sharpes, 97.5)

        # Percentiles
        percentiles = [5, 25, 50, 75, 95]
        cagr_percentiles = {
            p: np.percentile(simulated_cagrs, p) for p in percentiles
        }
        mdd_percentiles = {
            p: np.percentile(simulated_mdds, p) for p in percentiles
        }

        return MonteCarloResult(
            original_result=self.result,
            n_simulations=n_simulations,
            simulated_returns=simulated_returns,
            simulated_cagrs=simulated_cagrs,
            simulated_mdds=simulated_mdds,
            simulated_sharpes=simulated_sharpes,
            mean_cagr=mean_cagr,
            std_cagr=std_cagr,
            mean_mdd=mean_mdd,
            std_mdd=std_mdd,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            cagr_ci_lower=cagr_ci_lower,
            cagr_ci_upper=cagr_ci_upper,
            mdd_ci_lower=mdd_ci_lower,
            mdd_ci_upper=mdd_ci_upper,
            sharpe_ci_lower=sharpe_ci_lower,
            sharpe_ci_upper=sharpe_ci_upper,
            cagr_percentiles=cagr_percentiles,
            mdd_percentiles=mdd_percentiles,
        )

    def probability_of_loss(self, mc_result: MonteCarloResult) -> float:
        """
        Calculate probability of negative return.

        Args:
            mc_result: MonteCarloResult from simulation

        Returns:
            Probability (0-1) of negative return
        """
        negative_count = np.sum(mc_result.simulated_cagrs < 0)
        return negative_count / mc_result.n_simulations

    def value_at_risk(
        self, mc_result: MonteCarloResult, confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            mc_result: MonteCarloResult from simulation
            confidence: Confidence level (default: 0.95 = 95%)

        Returns:
            VaR as percentage
        """
        percentile = (1 - confidence) * 100
        return np.percentile(mc_result.simulated_cagrs, percentile)

    def conditional_value_at_risk(
        self, mc_result: MonteCarloResult, confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).

        Args:
            mc_result: MonteCarloResult from simulation
            confidence: Confidence level (default: 0.95 = 95%)

        Returns:
            CVaR as percentage
        """
        percentile = (1 - confidence) * 100
        var = np.percentile(mc_result.simulated_cagrs, percentile)
        # Average of returns below VaR
        below_var = mc_result.simulated_cagrs[
            mc_result.simulated_cagrs <= var
        ]
        return np.mean(below_var) if len(below_var) > 0 else var


def run_monte_carlo(
    result: BacktestResult,
    n_simulations: int = 1000,
    method: str = "bootstrap",
    random_seed: int | None = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation on backtest result.

    Args:
        result: BacktestResult from historical backtest
        n_simulations: Number of simulation runs
        method: Simulation method ('bootstrap' or 'parametric')
        random_seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with simulation statistics

    Example:
        Run Monte Carlo analysis::

            from src.backtester import run_backtest, BacktestConfig
            from src.backtester.monte_carlo import run_monte_carlo

            result = run_backtest(...)
            mc_result = run_monte_carlo(result, n_simulations=1000)

            print(f"Mean CAGR: {mc_result.mean_cagr:.2f}%")
            print(f"CAGR 95% CI: [{mc_result.cagr_ci_lower:.2f}%, {mc_result.cagr_ci_upper:.2f}%]")
            print(f"Probability of loss: {mc_result.probability_of_loss(mc_result)*100:.1f}%")
    """
    simulator = MonteCarloSimulator(result)
    return simulator.simulate(
        n_simulations=n_simulations, method=method, random_seed=random_seed
    )
