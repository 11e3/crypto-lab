"""
Walk-forward analysis for backtesting.

Walk-forward analysis splits data into multiple periods:
- Optimization period: Used to find best parameters
- Testing period: Used to test the optimized parameters

This helps prevent overfitting and validates strategy robustness.
"""

from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.backtester.models import BacktestConfig
from src.backtester.wfa.walk_forward_models import WalkForwardPeriod, WalkForwardResult
from src.backtester.wfa.walk_forward_runner import (
    generate_periods,
    optimize_period,
    run_test_period,
)
from src.backtester.wfa.walk_forward_stats import calculate_walk_forward_statistics
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = [
    "WalkForwardAnalyzer",
    "WalkForwardPeriod",
    "WalkForwardResult",
    "run_walk_forward_analysis",
]


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

    def _load_all_data(self) -> dict[str, pd.DataFrame]:
        """Load raw OHLCV data for all tickers once (no indicators applied)."""
        from src.data.upbit_source import UpbitDataSource

        data_source = UpbitDataSource()
        ticker_data: dict[str, pd.DataFrame] = {}

        for ticker in self.tickers:
            df = data_source.load_ohlcv(ticker, self.interval)
            if df is not None and len(df) > 0:
                ticker_data[ticker] = df

        return ticker_data

    def _load_date_range(
        self,
        ticker_data: dict[str, pd.DataFrame],
        start_date: date | None,
        end_date: date | None,
    ) -> tuple[date, date]:
        """Determine valid date range from pre-loaded ticker data."""
        all_dates: list[date] = []

        for df in ticker_data.values():
            ticker_dates = [d.date() if isinstance(d, datetime) else d for d in df.index]
            all_dates.extend(ticker_dates)

        if not all_dates:
            raise ValueError("No data available for walk-forward analysis")

        return (
            start_date if start_date is not None else min(all_dates),
            end_date if end_date is not None else max(all_dates),
        )

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
            start_date: Start date for analysis
            end_date: End date for analysis
            n_workers: Number of parallel workers

        Returns:
            WalkForwardResult with analysis results
        """
        ticker_data = self._load_all_data()
        start_date, end_date = self._load_date_range(ticker_data, start_date, end_date)

        periods = generate_periods(
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

        for period in periods:
            self._process_period(period, param_grid, metric, n_workers, ticker_data)

        return calculate_walk_forward_statistics(periods)

    def _process_period(
        self,
        period: WalkForwardPeriod,
        param_grid: dict[str, list[Any]],
        metric: str,
        n_workers: int | None,
        ticker_data: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Process a single walk-forward period."""
        logger.info(f"\nProcessing period {period.period_num}...")
        logger.info(f"  Optimization: {period.optimization_start} to {period.optimization_end}")
        logger.info(f"  Test: {period.test_start} to {period.test_end}")

        # Optimize on optimization period
        opt_result = optimize_period(
            period=period,
            strategy_factory=self.strategy_factory,
            tickers=self.tickers,
            interval=self.interval,
            config=self.config,
            param_grid=param_grid,
            metric=metric,
            n_workers=n_workers,
            ticker_data=ticker_data,
        )
        period.optimization_result = opt_result

        if opt_result:
            logger.info(
                f"  Best params: {opt_result.best_params}, score: {opt_result.best_score:.4f}"
            )

            # Test on test period
            test_result = run_test_period(
                period=period,
                strategy_factory=self.strategy_factory,
                tickers=self.tickers,
                interval=self.interval,
                config=self.config,
                best_params=opt_result.best_params,
            )
            period.test_result = test_result

            if test_result:
                logger.info(
                    f"  Test result: CAGR={test_result.cagr:.2f}%, "
                    f"Sharpe={test_result.sharpe_ratio:.2f}, MDD=-{test_result.mdd:.2f}%"
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
