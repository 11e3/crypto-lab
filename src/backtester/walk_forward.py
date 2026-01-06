"""
Walk-forward analysis for backtesting.

Walk-forward analysis splits data into multiple periods:
- Optimization period: Used to find best parameters
- Testing period: Used to test the optimized parameters

This helps prevent overfitting and validates strategy robustness.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtester.engine import BacktestConfig, BacktestResult, run_backtest
from src.backtester.optimization import OptimizationResult, optimize_strategy_parameters
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardPeriod:
    """Single period in walk-forward analysis."""

    period_num: int
    optimization_start: date
    optimization_end: date
    test_start: date
    test_end: date
    optimization_result: OptimizationResult | None = None
    test_result: BacktestResult | None = None

    def __repr__(self) -> str:
        return (
            f"WalkForwardPeriod({self.period_num}, "
            f"opt={self.optimization_start} to {self.optimization_end}, "
            f"test={self.test_start} to {self.test_end})"
        )


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""

    periods: list[WalkForwardPeriod]
    overall_test_result: BacktestResult | None = None

    # Aggregate statistics
    avg_test_cagr: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_test_mdd: float = 0.0
    avg_optimization_cagr: float = 0.0

    # Consistency metrics
    positive_periods: int = 0
    total_periods: int = 0
    consistency_rate: float = 0.0

    def __repr__(self) -> str:
        return (
            f"WalkForwardResult({self.total_periods} periods, "
            f"avg_test_cagr={self.avg_test_cagr:.2f}%)"
        )


class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis on trading strategies.

    Splits data into multiple periods, optimizes parameters on each
    optimization period, and tests on the following test period.
    """

    def __init__(
        self,
        strategy_factory: Callable[[dict[str, Any]], Strategy],
        tickers: list[str],
        interval: str,
        config: BacktestConfig,
    ) -> None:
        """
        Initialize walk-forward analyzer.

        Args:
            strategy_factory: Function that creates a strategy from parameters
            tickers: List of tickers to backtest
            interval: Data interval
            config: Backtest configuration
        """
        self.strategy_factory = strategy_factory
        self.tickers = tickers
        self.interval = interval
        self.config = config

    def analyze(
        self,
        param_grid: dict[str, list[Any]],
        optimization_days: int = 365,
        test_days: int = 90,
        step_days: int = 90,
        metric: str = "sharpe_ratio",
        start_date: date | None = None,
        end_date: date | None = None,
        n_workers: int | None = None,
    ) -> WalkForwardResult:
        """
        Perform walk-forward analysis.

        Args:
            param_grid: Parameter grid for optimization
            optimization_days: Length of optimization period in days
            test_days: Length of test period in days
            step_days: Step size between periods in days
            metric: Metric to optimize
            start_date: Start date for analysis (uses data start if None)
            end_date: End date for analysis (uses data end if None)
            n_workers: Number of parallel workers for optimization

        Returns:
            WalkForwardResult with analysis results
        """
        # Load data to determine date range
        from src.data.upbit_source import UpbitDataSource

        data_source = UpbitDataSource()
        all_dates: list[date] = []

        for ticker in self.tickers:
            df = data_source.load_ohlcv(ticker, self.interval)
            if df is not None and len(df) > 0:
                ticker_dates = [d.date() if isinstance(d, datetime) else d for d in df.index]
                all_dates.extend(ticker_dates)

        if not all_dates:
            raise ValueError("No data available for walk-forward analysis")

        # Determine date range
        min_date = min(all_dates)
        max_date = max(all_dates)

        if start_date is None:
            start_date = min_date
        if end_date is None:
            end_date = max_date

        # Generate periods
        periods = self._generate_periods(
            start_date=start_date,
            end_date=end_date,
            optimization_days=optimization_days,
            test_days=test_days,
            step_days=step_days,
        )

        logger.info(f"Walk-forward analysis: {len(periods)} periods")
        logger.info(
            f"Date range: {start_date} to {end_date} "
            f"(optimization: {optimization_days}d, test: {test_days}d, step: {step_days}d)"
        )

        # Process each period
        for period in periods:
            logger.info(f"\nProcessing period {period.period_num}...")
            logger.info(
                f"  Optimization: {period.optimization_start} to {period.optimization_end}"
            )
            logger.info(f"  Test: {period.test_start} to {period.test_end}")

            # Optimize on optimization period
            opt_result = self._optimize_period(
                period=period,
                param_grid=param_grid,
                metric=metric,
                n_workers=n_workers,
            )
            period.optimization_result = opt_result

            if opt_result:
                logger.info(
                    f"  Best params: {opt_result.best_params}, "
                    f"score: {opt_result.best_score:.4f}"
                )

                # Test on test period
                test_result = self._test_period(
                    period=period,
                    best_params=opt_result.best_params,
                )
                period.test_result = test_result

                if test_result:
                    logger.info(
                        f"  Test result: CAGR={test_result.cagr:.2f}%, "
                        f"Sharpe={test_result.sharpe_ratio:.2f}, "
                        f"MDD={test_result.mdd:.2f}%"
                    )

        # Calculate aggregate statistics
        result = self._calculate_statistics(periods)

        return result

    def _generate_periods(
        self,
        start_date: date,
        end_date: date,
        optimization_days: int,
        test_days: int,
        step_days: int,
    ) -> list[WalkForwardPeriod]:
        """Generate walk-forward periods."""
        from datetime import timedelta

        periods: list[WalkForwardPeriod] = []
        period_num = 1

        current_date = start_date

        while current_date < end_date:
            # Optimization period
            opt_start = current_date
            opt_end = opt_start + timedelta(days=optimization_days)

            # Test period (immediately after optimization)
            test_start = opt_end
            test_end = test_start + timedelta(days=test_days)

            # Skip if test period extends beyond end_date
            if test_end > end_date:
                break

            periods.append(
                WalkForwardPeriod(
                    period_num=period_num,
                    optimization_start=opt_start,
                    optimization_end=opt_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            # Move to next period
            current_date += timedelta(days=step_days)
            period_num += 1

        return periods

    def _optimize_period(
        self,
        period: WalkForwardPeriod,
        param_grid: dict[str, list[Any]],
        metric: str,
        n_workers: int | None,
    ) -> OptimizationResult | None:
        """Optimize parameters on optimization period."""
        try:
            # Create strategy factory
            def filtered_backtest_factory(params: dict[str, Any]) -> Strategy:
                return self.strategy_factory(params)

            # Create a custom optimization function that filters data by date
            from src.backtester.optimization import ParameterOptimizer

            # Create optimizer with date filtering
            optimizer = ParameterOptimizer(
                strategy_factory=filtered_backtest_factory,
                tickers=self.tickers,
                interval=self.interval,
                config=self.config,
                n_workers=n_workers,
            )

            # Create a wrapper to add date filtering to optimization
            # We need to modify the optimization to use date-filtered backtests
            # For now, we'll create a custom optimization that uses date ranges
            from src.backtester.parallel import ParallelBacktestRunner, ParallelBacktestTask
            from itertools import product

            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            combinations = list(product(*param_values))

            tasks = []
            for combo in combinations:
                params = dict(zip(param_names, combo))
                strategy = filtered_backtest_factory(params)
                task_name = f"{strategy.name}_{'_'.join(str(v) for v in combo)}"
                tasks.append(
                    ParallelBacktestTask(
                        name=task_name,
                        strategy=strategy,
                        tickers=self.tickers,
                        interval=self.interval,
                        config=self.config,
                        params=params,
                        start_date=period.optimization_start,
                        end_date=period.optimization_end,
                    )
                )

            runner = ParallelBacktestRunner(n_workers=n_workers)
            results = runner.run(tasks)

            # Extract scores and find best
            all_results = []
            for task in tasks:
                result = results.get(task.name)
                if result and task.params:
                    from src.backtester.optimization import OptimizationResult
                    score = self._extract_metric(result, metric)
                    all_results.append((task.params, result, score))

            # Sort by score
            all_results.sort(key=lambda x: x[2], reverse=True)

            if not all_results:
                return None

            best_params, best_result, best_score = all_results[0]

            from src.backtester.optimization import OptimizationResult
            return OptimizationResult(
                best_params=best_params,
                best_result=best_result,
                best_score=best_score,
                all_results=all_results,
                optimization_metric=metric,
            )
        except Exception as e:
            logger.error(f"Error optimizing period {period.period_num}: {e}", exc_info=True)
            return None

    def _extract_metric(self, result: BacktestResult, metric: str) -> float:
        """Extract metric value from backtest result."""
        if metric == "sharpe_ratio":
            return result.sharpe_ratio if hasattr(result, "sharpe_ratio") else 0.0
        elif metric == "cagr":
            return result.cagr if hasattr(result, "cagr") else 0.0
        elif metric == "total_return":
            return result.total_return if hasattr(result, "total_return") else 0.0
        elif metric == "calmar_ratio":
            return result.calmar_ratio if hasattr(result, "calmar_ratio") else 0.0
        elif metric == "win_rate":
            return result.win_rate if hasattr(result, "win_rate") else 0.0
        elif metric == "profit_factor":
            return result.profit_factor if hasattr(result, "profit_factor") else 0.0
        else:
            logger.warning(f"Unknown metric: {metric}, using sharpe_ratio")
            return result.sharpe_ratio if hasattr(result, "sharpe_ratio") else 0.0

    def _test_period(
        self,
        period: WalkForwardPeriod,
        best_params: dict[str, Any],
    ) -> BacktestResult | None:
        """Test optimized parameters on test period."""
        try:
            strategy = self.strategy_factory(best_params)

            # Run backtest with date filtering for test period
            result = run_backtest(
                strategy=strategy,
                tickers=self.tickers,
                interval=self.interval,
                config=self.config,
                start_date=period.test_start,
                end_date=period.test_end,
            )

            return result
        except Exception as e:
            logger.error(f"Error testing period {period.period_num}: {e}", exc_info=True)
            return None

    def _calculate_statistics(
        self, periods: list[WalkForwardPeriod]
    ) -> WalkForwardResult:
        """Calculate aggregate statistics from periods."""
        test_cagrs: list[float] = []
        test_sharpes: list[float] = []
        test_mdds: list[float] = []
        opt_cagrs: list[float] = []

        positive_count = 0
        total_count = 0

        for period in periods:
            if period.test_result:
                test_cagrs.append(period.test_result.cagr)
                test_sharpes.append(period.test_result.sharpe_ratio)
                test_mdds.append(period.test_result.mdd)
                total_count += 1
                if period.test_result.cagr > 0:
                    positive_count += 1

            if period.optimization_result and period.optimization_result.best_result:
                opt_cagrs.append(period.optimization_result.best_result.cagr)

        # Calculate averages
        avg_test_cagr = np.mean(test_cagrs) if test_cagrs else 0.0
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0.0
        avg_test_mdd = np.mean(test_mdds) if test_mdds else 0.0
        avg_optimization_cagr = np.mean(opt_cagrs) if opt_cagrs else 0.0

        consistency_rate = (positive_count / total_count * 100) if total_count > 0 else 0.0

        return WalkForwardResult(
            periods=periods,
            avg_test_cagr=avg_test_cagr,
            avg_test_sharpe=avg_test_sharpe,
            avg_test_mdd=avg_test_mdd,
            avg_optimization_cagr=avg_optimization_cagr,
            positive_periods=positive_count,
            total_periods=total_count,
            consistency_rate=consistency_rate,
        )


def run_walk_forward_analysis(
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    param_grid: dict[str, list[Any]],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    optimization_days: int = 365,
    test_days: int = 90,
    step_days: int = 90,
    metric: str = "sharpe_ratio",
    start_date: date | None = None,
    end_date: date | None = None,
    n_workers: int | None = None,
) -> WalkForwardResult:
    """
    Run walk-forward analysis on a trading strategy.

    Args:
        strategy_factory: Function that creates a strategy from parameters
        param_grid: Parameter grid for optimization
        tickers: List of tickers to backtest
        interval: Data interval
        config: Backtest configuration
        optimization_days: Length of optimization period in days
        test_days: Length of test period in days
        step_days: Step size between periods in days
        metric: Metric to optimize
        start_date: Start date for analysis
        end_date: End date for analysis
        n_workers: Number of parallel workers

    Returns:
        WalkForwardResult

    Example:
        Run walk-forward analysis::

            from src.strategies.volatility_breakout import create_vbo_strategy
            from src.backtester import BacktestConfig

            def create_strategy(params):
                return create_vbo_strategy(
                    name=f"VBO_{params['sma']}_{params['trend']}",
                    sma_period=params['sma'],
                    trend_sma_period=params['trend'],
                )

            param_grid = {
                'sma': [4, 5, 6],
                'trend': [8, 10, 12],
            }

            result = run_walk_forward_analysis(
                strategy_factory=create_strategy,
                param_grid=param_grid,
                tickers=["KRW-BTC"],
                interval="day",
                config=BacktestConfig(),
                optimization_days=365,
                test_days=90,
                step_days=90,
            )

            print(f"Average test CAGR: {result.avg_test_cagr:.2f}%")
            print(f"Consistency rate: {result.consistency_rate:.1f}%")
    """
    analyzer = WalkForwardAnalyzer(
        strategy_factory=strategy_factory,
        tickers=tickers,
        interval=interval,
        config=config,
    )

    return analyzer.analyze(
        param_grid=param_grid,
        optimization_days=optimization_days,
        test_days=test_days,
        step_days=step_days,
        metric=metric,
        start_date=start_date,
        end_date=end_date,
        n_workers=n_workers,
    )
