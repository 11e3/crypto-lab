"""
CLI command for optimizing strategy parameters.
"""

import click
from pathlib import Path

from src.backtester import BacktestConfig, optimize_strategy_parameters
from src.strategies.volatility_breakout import create_vbo_strategy
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@click.command(name="optimize")
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
    help="Strategy variant to optimize",
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
    "--method",
    type=click.Choice(["grid", "random"], case_sensitive=False),
    default="grid",
    help="Optimization method: grid or random (default: grid)",
)
@click.option(
    "--n-iter",
    type=int,
    default=100,
    help="Number of iterations for random search (default: 100)",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count - 1)",
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
    "--short-noise-range",
    type=str,
    default=None,
    help="Short noise period range (comma-separated)",
)
@click.option(
    "--long-noise-range",
    type=str,
    default=None,
    help="Long noise period range (comma-separated)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=str),
    default=None,
    help="Output path for optimization report",
)
def optimize(
    tickers: tuple[str, ...],
    interval: str,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    strategy: str,
    metric: str,
    method: str,
    n_iter: int,
    workers: int | None,
    sma_range: str,
    trend_range: str,
    short_noise_range: str | None,
    long_noise_range: str | None,
    output: str | None,
) -> None:
    """
    Optimize strategy parameters using grid or random search.

    Example:
        crypto-quant optimize --strategy vanilla --metric sharpe_ratio --sma-range 4,5,6 --trend-range 8,10,12
    """
    ticker_list = list(tickers)

    logger.info("Starting parameter optimization...")
    logger.info(f"Tickers: {ticker_list}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Metric: {metric}")
    logger.info(f"Method: {method}")

    # Parse parameter ranges
    def parse_range(range_str: str) -> list[int]:
        return [int(x.strip()) for x in range_str.split(",")]

    param_grid: dict[str, list[int]] = {}
    param_grid["sma_period"] = parse_range(sma_range)
    param_grid["trend_sma_period"] = parse_range(trend_range)

    if short_noise_range:
        param_grid["short_noise_period"] = parse_range(short_noise_range)
    else:
        # Default: same as sma_period
        param_grid["short_noise_period"] = param_grid["sma_period"]

    if long_noise_range:
        param_grid["long_noise_period"] = parse_range(long_noise_range)
    else:
        # Default: same as trend_sma_period
        param_grid["long_noise_period"] = param_grid["trend_sma_period"]

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

    # Run optimization
    result = optimize_strategy_parameters(
        strategy_factory=create_strategy,
        param_grid=param_grid,
        tickers=ticker_list,
        interval=interval,
        config=config,
        metric=metric,
        maximize=True,
        method=method,
        n_iter=n_iter,
        n_workers=workers,
    )

    # Print results
    logger.info("\n=== Optimization Results ===\n")
    logger.info(f"Best Parameters: {result.best_params}")
    logger.info(f"Best {metric}: {result.best_score:.4f}")
    logger.info(f"\nBest Result Metrics:")
    logger.info(f"  CAGR: {result.best_result.cagr:.2f}%")
    logger.info(f"  Total Return: {result.best_result.total_return:.2f}%")
    logger.info(f"  Sharpe Ratio: {result.best_result.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {result.best_result.mdd:.2f}%")
    logger.info(f"  Win Rate: {result.best_result.win_rate:.2f}%")
    logger.info(f"  Total Trades: {result.best_result.total_trades}")

    # Print top 5 results
    logger.info(f"\n=== Top 5 Results ===")
    logger.info(f"{'Rank':<6} {'Params':<30} {metric.capitalize():>15}")
    logger.info("-" * 60)
    for i, (params, _, score) in enumerate(result.all_results[:5], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        logger.info(f"{i:<6} {params_str:<30} {score:>15.4f}")

    # Generate report if output specified
    if output:
        from src.backtester.report import generate_report
        from datetime import datetime

        output_path = Path(output)
        if output_path.is_dir() or not output_path.suffix:
            # Directory or no extension: create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path / f"optimization_{timestamp}.html"
        else:
            output_path = Path(output)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate report for best result
        generate_report(
            result.best_result,
            save_path=output_path,
            show=False,
            format="html",
            strategy_obj=None,
            config=config,
            tickers=ticker_list,
        )
        logger.info(f"\nReport saved: {output_path}")
