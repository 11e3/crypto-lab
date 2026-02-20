"""Analysis service — pure computation logic for Monte Carlo and Walk-Forward.

Extracted from web/pages/analysis.py to decouple from Streamlit.
"""

from __future__ import annotations

from typing import Any

from src.backtester import BacktestConfig, run_backtest, run_walk_forward_analysis
from src.backtester.analysis.monte_carlo import MonteCarloResult, run_monte_carlo
from src.backtester.models import BacktestResult
from src.backtester.wfa.walk_forward import WalkForwardResult
from src.utils.logger import get_logger
from src.web.services.strategy_registry import create_analysis_strategy

logger = get_logger(__name__)


def execute_monte_carlo(
    strategy_type: str,
    tickers: list[str],
    interval: str,
    n_simulations: int,
    method: str,
    seed: int | None,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
) -> tuple[MonteCarloResult, BacktestResult]:
    """Run Monte Carlo simulation (no Streamlit dependency).

    Args:
        strategy_type: Internal strategy type name
        tickers: List of tickers
        interval: Data interval
        n_simulations: Number of simulations
        method: Simulation method (bootstrap/parametric)
        seed: Random seed (None for random)
        initial_capital: Initial capital
        fee_rate: Fee rate
        max_slots: Maximum position slots

    Returns:
        Tuple of (MonteCarloResult, BacktestResult)

    Raises:
        Exception: If backtest or simulation fails
    """
    strategy = create_analysis_strategy(strategy_type)

    config = BacktestConfig(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=fee_rate,
        max_slots=max_slots,
        use_cache=True,
    )

    result = run_backtest(
        strategy=strategy,
        tickers=tickers,
        interval=interval,
        config=config,
    )

    mc_result = run_monte_carlo(
        result=result,
        n_simulations=n_simulations,
        method=method,
        random_seed=seed,
    )

    return mc_result, result


def execute_walk_forward(
    strategy_type: str,
    param_grid: dict[str, list[int]],
    tickers: list[str],
    interval: str,
    optimization_days: int,
    test_days: int,
    step_days: int,
    metric: str,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    workers: int,
) -> WalkForwardResult:
    """Run Walk-Forward analysis (no Streamlit dependency).

    Args:
        strategy_type: Internal strategy type name
        param_grid: Parameter grid for optimization
        tickers: List of tickers
        interval: Data interval
        optimization_days: Optimization period in days
        test_days: Test period in days
        step_days: Step size in days
        metric: Optimization metric
        initial_capital: Initial capital
        fee_rate: Fee rate
        max_slots: Maximum position slots
        workers: Number of parallel workers

    Returns:
        WalkForwardResult

    Raises:
        Exception: If analysis fails
    """

    def create_strategy(params: dict[str, Any]) -> Any:
        return create_analysis_strategy(strategy_type, **params)

    config = BacktestConfig(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=fee_rate,
        max_slots=max_slots,
        use_cache=True,
    )

    result: WalkForwardResult = run_walk_forward_analysis(
        strategy_factory=create_strategy,
        param_grid=param_grid,
        tickers=tickers,
        interval=interval,
        config=config,
        optimization_days=optimization_days,
        test_days=test_days,
        step_days=step_days,
        metric=metric,
        n_workers=workers,
    )
    return result
