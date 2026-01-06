"""
CLI command for walk-forward analysis.
"""

import click
from pathlib import Path

from src.backtester import BacktestConfig, run_walk_forward_analysis
from src.strategies.volatility_breakout import create_vbo_strategy
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@click.command(name="walk-forward")
@click.option(
    "--tickers",
    "-t",
    multiple=True,
    default=["KRW-BTC", "KRW-ETH"],
    help="Trading pair tickers to backtest",
)
@click.option(
    "--interval",
    "-i",
    default="day",
    type=click.Choice(["minute240", "day", "week"], case_sensitive=False),
    help="Data interval for backtesting",
)
@click.option(
    "--initial-capital",
    "-c",
    type=float,
    default=1.0,
    help="Initial capital (default: 1.0)",
)
@click.option(
    "--fee-rate",
    "-f",
    type=float,
    default=0.0005,
    help="Trading fee rate (default: 0.0005 = 0.05%%)",
)
@click.option(
    "--max-slots",
    "-s",
    type=int,
    default=4,
    help="Maximum number of concurrent positions (default: 4)",
)
@click.option(
    "--strategy",
    type=click.Choice(["vanilla", "legacy"], case_sensitive=False),
    default="vanilla",
    help="Strategy variant to analyze",
)
@click.option(
    "--optimization-days",
    type=int,
    default=365,
    help="Length of optimization period in days (default: 365)",
)
@click.option(
    "--test-days",
    type=int,
    default=90,
    help="Length of test period in days (default: 90)",
)
@click.option(
    "--step-days",
    type=int,
    default=90,
    help="Step size between periods in days (default: 90)",
)
@click.option(
    "--metric",
    type=click.Choice(
        ["sharpe_ratio", "cagr", "total_return", "calmar_ratio", "win_rate", "profit_factor"],
        case_sensitive=False,
    ),
    default="sharpe_ratio",
    help="Metric to optimize (default: sharpe_ratio)",
)
@click.option(
    "--sma-range",
    type=str,
    default="4,5,6",
    help="SMA period range (comma-separated, e.g., '4,5,6')",
)
@click.option(
    "--trend-range",
    type=str,
    default="8,10,12",
    help="Trend SMA period range (comma-separated, e.g., '8,10,12')",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count - 1)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=str),
    default=None,
    help="Output path for walk-forward report",
)
def walk_forward(
    tickers: tuple[str, ...],
    interval: str,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    strategy: str,
    optimization_days: int,
    test_days: int,
    step_days: int,
    metric: str,
    sma_range: str,
    trend_range: str,
    workers: int | None,
    output: str | None,
) -> None:
    """
    Run walk-forward analysis on trading strategy.

    Walk-forward analysis splits data into multiple periods:
    - Optimization period: Used to find best parameters
    - Test period: Used to test the optimized parameters

    This helps prevent overfitting and validates strategy robustness.

    Example:
        crypto-quant walk-forward --strategy vanilla --optimization-days 365 --test-days 90
    """
    ticker_list = list(tickers)

    logger.info("Starting walk-forward analysis...")
    logger.info(f"Tickers: {ticker_list}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Optimization: {optimization_days} days, Test: {test_days} days, Step: {step_days} days")

    # Parse parameter ranges
    def parse_range(range_str: str) -> list[int]:
        return [int(x.strip()) for x in range_str.split(",")]

    param_grid: dict[str, list[int]] = {}
    param_grid["sma_period"] = parse_range(sma_range)
    param_grid["trend_sma_period"] = parse_range(trend_range)
    param_grid["short_noise_period"] = param_grid["sma_period"]  # Default: same as sma
    param_grid["long_noise_period"] = param_grid["trend_sma_period"]  # Default: same as trend

    # Create strategy factory
    def create_strategy(params: dict[str, int]) -> Any:
        if strategy == "vanilla":
            return create_vbo_strategy(
                name=f"VBO_{params['sma_period']}_{params['trend_sma_period']}",
                sma_period=params["sma_period"],
                trend_sma_period=params["trend_sma_period"],
                short_noise_period=params.get("short_noise_period", params["sma_period"]),
                long_noise_period=params.get("long_noise_period", params["trend_sma_period"]),
                exclude_current=False,
                use_trend_filter=True,
                use_noise_filter=True,
            )
        else:  # legacy
            return create_vbo_strategy(
                name=f"Legacy_{params['sma_period']}_{params['trend_sma_period']}",
                sma_period=params["sma_period"],
                trend_sma_period=params["trend_sma_period"],
                short_noise_period=params.get("short_noise_period", params["sma_period"]),
                long_noise_period=params.get("long_noise_period", params["trend_sma_period"]),
                exclude_current=True,
                use_trend_filter=True,
                use_noise_filter=True,
            )

    # Create config
    config = BacktestConfig(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=fee_rate,
        max_slots=max_slots,
        use_cache=True,
    )

    # Run walk-forward analysis
    result = run_walk_forward_analysis(
        strategy_factory=create_strategy,
        param_grid=param_grid,
        tickers=ticker_list,
        interval=interval,
        config=config,
        optimization_days=optimization_days,
        test_days=test_days,
        step_days=step_days,
        metric=metric,
        n_workers=workers,
    )

    # Print results
    logger.info("\n=== Walk-Forward Analysis Results ===\n")
    logger.info(f"Total Periods: {result.total_periods}")
    logger.info(f"Positive Periods: {result.positive_periods}")
    logger.info(f"Consistency Rate: {result.consistency_rate:.1f}%")
    logger.info(f"\nAverage Test Metrics:")
    logger.info(f"  CAGR: {result.avg_test_cagr:.2f}%")
    logger.info(f"  Sharpe Ratio: {result.avg_test_sharpe:.2f}")
    logger.info(f"  MDD: {result.avg_test_mdd:.2f}%")
    logger.info(f"\nAverage Optimization CAGR: {result.avg_optimization_cagr:.2f}%")

    # Print period details
    logger.info(f"\n=== Period Details ===")
    for period in result.periods:
        if period.test_result:
            logger.info(
                f"Period {period.period_num}: "
                f"Test CAGR={period.test_result.cagr:.2f}%, "
                f"Sharpe={period.test_result.sharpe_ratio:.2f}, "
                f"MDD={period.test_result.mdd:.2f}%"
            )
            if period.optimization_result:
                logger.info(
                    f"  Best params: {period.optimization_result.best_params}, "
                    f"score: {period.optimization_result.best_score:.4f}"
                )

    # Generate report if output specified
    if output:
        from datetime import datetime

        output_path = Path(output)
        if output_path.is_dir() or not output_path.suffix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path / f"walk_forward_{timestamp}.txt"
        else:
            output_path = Path(output)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write text report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Walk-Forward Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Periods: {result.total_periods}\n")
            f.write(f"Positive Periods: {result.positive_periods}\n")
            f.write(f"Consistency Rate: {result.consistency_rate:.1f}%\n\n")
            f.write(f"Average Test CAGR: {result.avg_test_cagr:.2f}%\n")
            f.write(f"Average Test Sharpe: {result.avg_test_sharpe:.2f}\n")
            f.write(f"Average Test MDD: {result.avg_test_mdd:.2f}%\n\n")
            f.write("Period Details:\n")
            f.write("-" * 60 + "\n")
            for period in result.periods:
                if period.test_result:
                    f.write(
                        f"Period {period.period_num} ({period.test_start} to {period.test_end}):\n"
                    )
                    f.write(f"  CAGR: {period.test_result.cagr:.2f}%\n")
                    f.write(f"  Sharpe: {period.test_result.sharpe_ratio:.2f}\n")
                    f.write(f"  MDD: {period.test_result.mdd:.2f}%\n")
                    if period.optimization_result:
                        f.write(f"  Best params: {period.optimization_result.best_params}\n")
                    f.write("\n")

        logger.info(f"\nReport saved: {output_path}")
