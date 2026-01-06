"""
CLI command for comparing multiple strategies using parallel backtesting.
"""

import click
from pathlib import Path

from src.backtester import BacktestConfig, compare_strategies
from src.strategies.momentum import MomentumStrategy, SimpleMomentumStrategy
from src.strategies.mean_reversion import (
    MeanReversionStrategy,
    SimpleMeanReversionStrategy,
)
from src.strategies.pair_trading import PairTradingStrategy
from src.strategies.volatility_breakout import (
    MinimalVBO,
    VanillaVBO,
    create_vbo_strategy,
)
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@click.command(name="compare")
@click.option(
    "--tickers",
    "-t",
    multiple=True,
    default=["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-TRX"],
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
    "--strategies",
    multiple=True,
    type=click.Choice(
        [
            "vanilla",
            "minimal",
            "legacy",
            "momentum",
            "simple-momentum",
            "mean-reversion",
            "simple-mean-reversion",
        ],
        case_sensitive=False,
    ),
    default=["vanilla", "minimal", "legacy"],
    help="Strategies to compare (can specify multiple)",
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
    help="Output directory for comparison reports",
)
def compare(
    tickers: tuple[str, ...],
    interval: str,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    strategies: tuple[str, ...],
    workers: int | None,
    output: str | None,
) -> None:
    """
    Compare multiple strategies using parallel backtesting.

    Example:
        crypto-quant compare --strategies vanilla minimal legacy
    """
    ticker_list = list(tickers)
    strategy_list = list(strategies)

    logger.info("Starting strategy comparison...")
    logger.info(f"Tickers: {ticker_list}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Strategies: {strategy_list}")
    logger.info(f"Initial capital: {initial_capital}")
    logger.info(f"Fee rate: {fee_rate}")
    logger.info(f"Max slots: {max_slots}")

    # Create strategies
    strategy_objects = []
    for strategy_name in strategy_list:
        if strategy_name == "vanilla":
            strategy_objects.append(
                create_vbo_strategy(
                    name="VanillaVBO",
                    sma_period=4,
                    trend_sma_period=8,
                    short_noise_period=4,
                    long_noise_period=8,
                    exclude_current=False,
                    use_trend_filter=True,
                    use_noise_filter=True,
                )
            )
        elif strategy_name == "minimal":
            strategy_objects.append(
                create_vbo_strategy(
                    name="MinimalVBO",
                    sma_period=4,
                    trend_sma_period=8,
                    short_noise_period=4,
                    long_noise_period=8,
                    exclude_current=False,
                    use_trend_filter=False,
                    use_noise_filter=False,
                )
            )
        elif strategy_name == "legacy":
            strategy_objects.append(
                create_vbo_strategy(
                    name="LegacyBT",
                    sma_period=5,
                    trend_sma_period=10,
                    short_noise_period=5,
                    long_noise_period=10,
                    exclude_current=True,
                    use_trend_filter=True,
                    use_noise_filter=True,
                )
            )
        elif strategy_name == "momentum":
            strategy_objects.append(MomentumStrategy(name="MomentumStrategy"))
        elif strategy_name == "simple-momentum":
            strategy_objects.append(SimpleMomentumStrategy(name="SimpleMomentum"))
        elif strategy_name == "mean-reversion":
            strategy_objects.append(MeanReversionStrategy(name="MeanReversionStrategy"))
        elif strategy_name == "simple-mean-reversion":
            strategy_objects.append(
                SimpleMeanReversionStrategy(name="SimpleMeanReversion")
            )

    # Create config
    config = BacktestConfig(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=fee_rate,
        max_slots=max_slots,
        use_cache=True,
    )

    # Run parallel comparison
    results = compare_strategies(
        strategies=strategy_objects,
        tickers=ticker_list,
        interval=interval,
        config=config,
        n_workers=workers,
    )

    # Print comparison summary
    logger.info("\n=== Strategy Comparison Results ===\n")
    logger.info(f"{'Strategy':<25} {'CAGR':>10} {'MDD':>10} {'Sharpe':>10} {'Trades':>10}")
    logger.info("-" * 75)

    for strategy_name, result in results.items():
        cagr = result.cagr if hasattr(result, "cagr") else 0.0
        mdd = result.mdd if hasattr(result, "mdd") else 0.0
        sharpe = result.sharpe_ratio if hasattr(result, "sharpe_ratio") else 0.0
        trades = result.total_trades if hasattr(result, "total_trades") else 0

        logger.info(
            f"{strategy_name:<25} {cagr:>10.2f} {mdd:>10.2f} {sharpe:>10.2f} {trades:>10}"
        )

    # Generate reports if output directory specified
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        from src.backtester.report import generate_report

        for strategy_name, result in results.items():
            report_path = output_dir / f"{strategy_name}_comparison.html"
            generate_report(
                result,
                save_path=report_path,
                show=False,
                format="html",
                strategy_obj=None,  # Strategy object not available in results
                config=config,
                tickers=ticker_list,
            )
            logger.info(f"Report saved: {report_path}")
