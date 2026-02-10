"""Monte Carlo simulation for backtest results."""

from dataclasses import dataclass

import numpy as np

from src.backtester.analysis.monte_carlo_metrics import (
    calculate_percentiles,
    calculate_simulation_metrics,
    calculate_statistics,
)
from src.backtester.models import BacktestResult
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
    mean_cagr: float
    std_cagr: float
    mean_mdd: float
    std_mdd: float
    mean_sharpe: float
    std_sharpe: float
    cagr_ci_lower: float
    cagr_ci_upper: float
    mdd_ci_lower: float
    mdd_ci_upper: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    cagr_percentiles: dict[float, float]
    mdd_percentiles: dict[float, float]

    def __repr__(self) -> str:
        return f"MonteCarloResult(n={self.n_simulations}, mean_cagr={self.mean_cagr:.2f}%)"


class MonteCarloSimulator:
    """Monte Carlo simulator for backtest results."""

    def __init__(
        self,
        result: BacktestResult,
        initial_capital: float | None = None,
    ) -> None:
        """Initialize Monte Carlo simulator."""
        self.result = result
        self.initial_capital = self._get_initial_capital(result, initial_capital)
        self.daily_returns = self._extract_returns(result)
        self._rng = np.random.default_rng()

    def _get_initial_capital(self, result: BacktestResult, initial_capital: float | None) -> float:
        """Determine initial capital from various sources."""
        if initial_capital is not None:
            return initial_capital
        if result.config and hasattr(result.config, "initial_capital"):
            return float(result.config.initial_capital)
        if len(result.equity_curve) > 0:
            return float(result.equity_curve[0])
        return 1.0

    def _extract_returns(self, result: BacktestResult) -> np.ndarray:
        """Extract daily returns from equity curve."""
        if len(result.equity_curve) > 1:
            equity = result.equity_curve
            returns: np.ndarray = np.diff(equity) / equity[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) > 0:
                return returns
        logger.warning("No valid returns found for Monte Carlo simulation")
        return np.array([0.0])

    def simulate(
        self,
        n_simulations: int = 1000,
        n_periods: int | None = None,
        method: str = "bootstrap",
        random_seed: int | None = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation."""
        self._rng = np.random.default_rng(random_seed)

        if n_periods is None:
            n_periods = len(self.daily_returns)

        if method == "bootstrap":
            simulated_returns = self._bootstrap(n_simulations, n_periods)
        elif method == "parametric":
            simulated_returns = self._parametric(n_simulations, n_periods)
        else:
            raise ValueError(f"Unknown simulation method: {method}")

        return self._build_result(simulated_returns, n_simulations)

    def _bootstrap(self, n_simulations: int, n_periods: int) -> np.ndarray:
        """Bootstrap method: resample from historical returns."""
        if len(self.daily_returns) == 0:
            return np.zeros((n_simulations, n_periods))
        return self._rng.choice(self.daily_returns, size=(n_simulations, n_periods), replace=True)

    def _parametric(self, n_simulations: int, n_periods: int) -> np.ndarray:
        """Parametric method: sample from normal distribution."""
        if len(self.daily_returns) == 0:
            mean_return, std_return = 0.0, 0.01
        else:
            mean_return = np.mean(self.daily_returns)
            std_return = np.std(self.daily_returns)
        return self._rng.normal(mean_return, std_return, size=(n_simulations, n_periods))

    def _build_result(self, simulated_returns: np.ndarray, n_simulations: int) -> MonteCarloResult:
        """Build MonteCarloResult from simulated returns."""
        simulated_equities = self._calculate_equities(simulated_returns, n_simulations)

        cagrs, mdds, sharpes = calculate_simulation_metrics(
            simulated_equities, simulated_returns, self.initial_capital
        )
        stats = calculate_statistics(cagrs, mdds, sharpes)

        return MonteCarloResult(
            original_result=self.result,
            n_simulations=n_simulations,
            simulated_returns=simulated_returns,
            simulated_cagrs=cagrs,
            simulated_mdds=mdds,
            simulated_sharpes=sharpes,
            mean_cagr=stats["mean_cagr"],
            std_cagr=stats["std_cagr"],
            mean_mdd=stats["mean_mdd"],
            std_mdd=stats["std_mdd"],
            mean_sharpe=stats["mean_sharpe"],
            std_sharpe=stats["std_sharpe"],
            cagr_ci_lower=stats["cagr_ci_lower"],
            cagr_ci_upper=stats["cagr_ci_upper"],
            mdd_ci_lower=stats["mdd_ci_lower"],
            mdd_ci_upper=stats["mdd_ci_upper"],
            sharpe_ci_lower=stats["sharpe_ci_lower"],
            sharpe_ci_upper=stats["sharpe_ci_upper"],
            cagr_percentiles=calculate_percentiles(cagrs),
            mdd_percentiles=calculate_percentiles(mdds),
        )

    def _calculate_equities(self, simulated_returns: np.ndarray, n_simulations: int) -> np.ndarray:
        """Calculate equity curves from returns."""
        simulated_equities = np.zeros_like(simulated_returns)
        for i in range(n_simulations):
            equity = self.initial_capital
            for j, ret in enumerate(simulated_returns[i]):
                equity = equity * (1 + ret)
                simulated_equities[i, j] = equity
        return simulated_equities

    def probability_of_loss(self, mc_result: MonteCarloResult) -> float:
        """Calculate probability of negative return."""
        negative_count = int(np.sum(mc_result.simulated_cagrs < 0))
        return float(negative_count) / mc_result.n_simulations

    def value_at_risk(self, mc_result: MonteCarloResult, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)."""
        percentile = (1 - confidence) * 100
        return float(np.percentile(mc_result.simulated_cagrs, percentile))

    def conditional_value_at_risk(
        self, mc_result: MonteCarloResult, confidence: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        percentile = (1 - confidence) * 100
        var: float = float(np.percentile(mc_result.simulated_cagrs, percentile))
        below_var = mc_result.simulated_cagrs[mc_result.simulated_cagrs <= var]
        return float(np.mean(below_var)) if len(below_var) > 0 else var


def run_monte_carlo(
    result: BacktestResult,
    n_simulations: int = 1000,
    method: str = "bootstrap",
    random_seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation on backtest result."""
    simulator = MonteCarloSimulator(result)
    return simulator.simulate(n_simulations=n_simulations, method=method, random_seed=random_seed)
