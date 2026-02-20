"""Strategy analysis orchestration module.

Answers three questions from a single backtest result:
1. Does the strategy have real alpha? (statistical significance)
2. How much alpha? (benchmark comparison)
3. Is it ready for live trading? (robustness + checklist)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtester.analysis.bootstrap_analysis import BootstrapAnalyzer, BootstrapResult
from src.backtester.analysis.permutation_test import PermutationTester, PermutationTestResult
from src.backtester.analysis.robustness_analysis import RobustnessAnalyzer
from src.backtester.analysis.robustness_models import RobustnessReport
from src.backtester.engine import run_backtest
from src.backtester.models import BacktestConfig, BacktestResult
from src.backtester.report_pkg.report_metrics import PerformanceMetrics, calculate_metrics
from src.config import UPBIT_DATA_DIR, parquet_filename
from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.utils.metrics_core import (
    calculate_cagr,
    calculate_daily_returns,
    calculate_mdd,
    calculate_sharpe_ratio,
)

logger = get_logger(__name__)


@dataclass
class BenchmarkMetrics:
    """Buy-and-hold benchmark performance metrics."""

    cagr: float
    mdd: float
    sharpe: float
    excess_return: float  # strategy CAGR - benchmark CAGR


@dataclass
class GoLiveCheck:
    """Single go-live readiness check."""

    description: str
    passed: bool
    detail: str
    level: str  # "PASS" | "WARN" | "FAIL"


@dataclass
class StrategyAnalysisResult:
    """Complete strategy analysis results."""

    strategy_name: str
    tickers: list[str]
    backtest: BacktestResult
    benchmark: BenchmarkMetrics | None = None
    permutation: PermutationTestResult | None = None
    bootstrap: BootstrapResult | None = None
    robustness: RobustnessReport | None = None
    go_live_checks: list[GoLiveCheck] = field(default_factory=list)
    performance: PerformanceMetrics | None = None


def run_strategy_analysis(
    strategy_name: str,
    strategy_factory_0: Callable[[], Strategy],
    strategy_factory_p: Callable[[dict[str, Any]], Strategy],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    start_date: date | None = None,
    end_date: date | None = None,
    n_shuffles: int = 100,
    n_bootstrap: int = 100,
    run_permutation: bool = True,
    run_robustness: bool = True,
    data_dir: Path | None = None,
) -> StrategyAnalysisResult:
    """Run comprehensive strategy analysis.

    Args:
        strategy_name: Name of the strategy
        strategy_factory_0: Zero-arg factory creating strategy with default params
        strategy_factory_p: Param-dict factory creating strategy with given params
        tickers: List of ticker symbols
        interval: Data interval (day, minute240, week)
        config: Backtest configuration
        start_date: Analysis start date
        end_date: Analysis end date
        n_shuffles: Number of permutation shuffles
        n_bootstrap: Number of bootstrap samples
        run_permutation: Whether to run permutation test
        run_robustness: Whether to run robustness analysis
        data_dir: Data directory (defaults to UPBIT_DATA_DIR)

    Returns:
        StrategyAnalysisResult with all analysis results
    """
    data_dir = data_dir or UPBIT_DATA_DIR

    # 1. Run main backtest
    logger.info(f"Running backtest for {strategy_name}")
    backtest = run_backtest(
        strategy=strategy_factory_0(),
        tickers=tickers,
        interval=interval,
        config=config,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
    )

    # 2. Calculate extended performance metrics (Sortino, volatility, etc.)
    performance: PerformanceMetrics | None = None
    if len(backtest.equity_curve) > 1 and len(backtest.dates) > 1:
        try:
            trades_df = (
                pd.DataFrame(
                    [{"exit_date": t.exit_date, "pnl": t.pnl, "pnl_pct": t.pnl_pct}
                     for t in backtest.trades]
                )
                if backtest.trades
                else pd.DataFrame(columns=["exit_date", "pnl", "pnl_pct"])
            )
            performance = calculate_metrics(
                equity_curve=backtest.equity_curve,
                dates=backtest.dates,
                trades_df=trades_df,
                initial_capital=config.initial_capital,
            )
        except Exception as e:
            logger.warning(f"Failed to calculate performance metrics: {e}")

    # 3. Load single-ticker data for statistical analysis (first ticker)
    primary_ticker = tickers[0]
    ticker_data = _load_ticker_data(primary_ticker, interval, data_dir, start_date, end_date)

    # 4. Benchmark: buy-and-hold on primary ticker
    benchmark: BenchmarkMetrics | None = None
    if ticker_data is not None:
        benchmark = _compute_benchmark(ticker_data, backtest.cagr)

    # 5. Permutation test for statistical significance
    permutation: PermutationTestResult | None = None
    if run_permutation and ticker_data is not None:
        logger.info(f"Running permutation test (n={n_shuffles})")
        try:
            tester = PermutationTester(
                data=ticker_data,
                strategy_factory=strategy_factory_0,
                backtest_config=config,
            )
            permutation = tester.run(num_shuffles=n_shuffles, verbose=False)
        except Exception as e:
            logger.warning(f"Permutation test failed: {e}")

    # 6. Robustness: bootstrap CI + parameter stability
    bootstrap_result: BootstrapResult | None = None
    robustness: RobustnessReport | None = None

    if run_robustness and ticker_data is not None:
        logger.info(f"Running bootstrap analysis (n={n_bootstrap})")
        try:
            boot_analyzer = BootstrapAnalyzer(
                data=ticker_data,
                strategy_factory=strategy_factory_0,  # type: ignore[arg-type]
                backtest_config=config,
                ticker=primary_ticker,
                interval=interval,
            )
            bootstrap_result = boot_analyzer.analyze(n_samples=n_bootstrap)
        except Exception as e:
            logger.warning(f"Bootstrap analysis failed: {e}")

        strategy_instance = strategy_factory_0()
        schema = strategy_instance.parameter_schema()
        if schema:
            logger.info("Running parameter robustness analysis")
            try:
                optimal_params = _extract_optimal_params(strategy_instance, schema)
                ranges = _build_robustness_ranges(optimal_params, schema)
                rob_analyzer = RobustnessAnalyzer(
                    data=ticker_data,
                    strategy_factory=strategy_factory_p,
                    backtest_config=config,
                )
                robustness = rob_analyzer.analyze(optimal_params, ranges, verbose=False)
            except Exception as e:
                logger.warning(f"Robustness analysis failed: {e}")

    # 7. Go-live checklist
    go_live_checks = _compute_go_live_checks(
        backtest, permutation, bootstrap_result, robustness, performance
    )

    return StrategyAnalysisResult(
        strategy_name=strategy_name,
        tickers=tickers,
        backtest=backtest,
        benchmark=benchmark,
        permutation=permutation,
        bootstrap=bootstrap_result,
        robustness=robustness,
        go_live_checks=go_live_checks,
        performance=performance,
    )


def _load_ticker_data(
    ticker: str,
    interval: str,
    data_dir: Path,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame | None:
    """Load and optionally date-filter ticker parquet data."""
    filepath = data_dir / parquet_filename(ticker, interval)
    if not filepath.exists():
        logger.warning(f"Data file not found: {filepath}")
        return None
    try:
        df = pd.read_parquet(filepath)
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]
        return df if not df.empty else None
    except Exception as e:
        logger.warning(f"Failed to load ticker data: {e}")
        return None


def _compute_benchmark(df: pd.DataFrame, strategy_cagr: float) -> BenchmarkMetrics | None:
    """Compute buy-and-hold benchmark metrics from OHLCV DataFrame."""
    try:
        close = df["close"].dropna().to_numpy(dtype=float)
        if len(close) < 2:
            return None

        total_days = len(close)
        if isinstance(df.index, pd.DatetimeIndex):
            total_days = max((df.index[-1] - df.index[0]).days, 1)

        equity = close / close[0]
        cagr = calculate_cagr(equity[0], equity[-1], total_days)
        mdd = calculate_mdd(equity)
        daily_returns = calculate_daily_returns(equity, prepend_zero=True)
        sharpe = calculate_sharpe_ratio(daily_returns)
        excess_return = strategy_cagr - cagr

        return BenchmarkMetrics(cagr=cagr, mdd=mdd, sharpe=sharpe, excess_return=excess_return)
    except Exception as e:
        logger.warning(f"Failed to compute benchmark metrics: {e}")
        return None


def _extract_optimal_params(strategy: Strategy, schema: dict[str, Any]) -> dict[str, Any]:
    """Extract current parameter values from a strategy instance via getattr."""
    return {name: getattr(strategy, name) for name in schema if hasattr(strategy, name)}


def _build_robustness_ranges(
    optimal_params: dict[str, Any],
    schema: dict[str, Any],
    n_steps: int = 2,
) -> dict[str, list[Any]]:
    """Build ±n_steps range around optimal params, clipped to schema min/max."""
    ranges: dict[str, list[Any]] = {}
    for name, raw_spec in schema.items():
        if name not in optimal_params:
            continue
        spec: dict[str, Any] = raw_spec if isinstance(raw_spec, dict) else dict(raw_spec)
        optimal = optimal_params[name]
        kind = spec.get("type", "")
        step = spec.get("step", 1)
        lo = spec.get("min", optimal)
        hi = spec.get("max", optimal)
        values: list[Any] = []

        if kind == "float":
            step_f, lo_f, hi_f = float(step), float(lo), float(hi)
            for i in range(-n_steps, n_steps + 1):
                v = round(float(optimal) + i * step_f, 8)
                if lo_f <= v <= hi_f:
                    values.append(v)
        elif kind == "int":
            step_i, lo_i, hi_i = int(step), int(lo), int(hi)
            for i in range(-n_steps, n_steps + 1):
                v = int(optimal) + i * step_i
                if lo_i <= v <= hi_i:
                    values.append(v)
        else:
            values = [optimal]

        ranges[name] = values if values else [optimal]

    return ranges


def _compute_go_live_checks(
    backtest: BacktestResult,
    permutation: PermutationTestResult | None,
    bootstrap: BootstrapResult | None,
    robustness: RobustnessReport | None,
    performance: PerformanceMetrics | None = None,
) -> list[GoLiveCheck]:
    """Compute the 5 go-live readiness checks."""
    checks: list[GoLiveCheck] = []

    # 1. Statistical significance (permutation test z-score > 2.0)
    if permutation is not None:
        z = permutation.z_score
        passed = z > 2.0
        checks.append(GoLiveCheck(
            description="통계적 유의성",
            passed=passed,
            detail=f"Z={z:.2f} {'>' if passed else '<='} 2.0",
            level="PASS" if passed else "FAIL",
        ))
    else:
        checks.append(GoLiveCheck(
            description="통계적 유의성",
            passed=False,
            detail="Permutation test skipped",
            level="WARN",
        ))

    # 2. Sortino ratio > 1.0 (uses downside deviation, more appropriate for trend-following)
    sortino = performance.sortino_ratio if performance is not None else backtest.sharpe_ratio
    passed = sortino > 1.0
    checks.append(GoLiveCheck(
        description="Sortino 비율",
        passed=passed,
        detail=f"{sortino:.2f} {'>' if passed else '<='} 1.0",
        level="PASS" if passed else "FAIL",
    ))

    # 3. MDD < 20%
    mdd = backtest.mdd
    passed = mdd < 20.0
    checks.append(GoLiveCheck(
        description="낙폭 통제",
        passed=passed,
        detail=f"MDD {mdd:.1f}% {'<' if passed else '>='} 20%",
        level="PASS" if passed else "FAIL",
    ))

    # 4. Parameter stability (neighbor success rate > 60%)
    if robustness is not None:
        nsr = robustness.neighbor_success_rate
        passed = nsr > 0.6
        checks.append(GoLiveCheck(
            description="파라미터 안정성",
            passed=passed,
            detail=f"이웃 성공률 {nsr:.1%} {'>' if passed else '<='} 60%",
            level="PASS" if passed else "FAIL",
        ))
    else:
        checks.append(GoLiveCheck(
            description="파라미터 안정성",
            passed=False,
            detail="Robustness analysis skipped",
            level="WARN",
        ))

    # 5. Bootstrap CI lower bound > 0
    if bootstrap is not None and bootstrap.ci_return_95 != (0.0, 0.0):
        lower = bootstrap.ci_return_95[0]
        passed = lower > 0.0
        checks.append(GoLiveCheck(
            description="수익 신뢰구간 양수",
            passed=passed,
            detail=f"95% CI 하한 {lower:.1f}% {'>' if passed else '<='} 0%",
            level="PASS" if passed else "FAIL",
        ))
    else:
        checks.append(GoLiveCheck(
            description="수익 신뢰구간 양수",
            passed=False,
            detail="Bootstrap analysis skipped",
            level="WARN",
        ))

    return checks


__all__ = [
    "BenchmarkMetrics",
    "GoLiveCheck",
    "StrategyAnalysisResult",
    "run_strategy_analysis",
    "_extract_optimal_params",
    "_build_robustness_ranges",
    "_compute_go_live_checks",
]
