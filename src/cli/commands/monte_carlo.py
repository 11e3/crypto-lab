"""
CLI command for Monte Carlo simulation of backtest results.
"""

import click
from pathlib import Path

from src.backtester import BacktestConfig, run_backtest
from src.backtester.monte_carlo import run_monte_carlo
from src.strategies.volatility_breakout import create_vbo_strategy
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@click.command(name="monte-carlo")
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
    type=click.Choice(
        ["vanilla", "minimal", "legacy", "momentum", "mean-reversion"],
        case_sensitive=False,
    ),
    default="vanilla",
    help="Strategy variant to use",
)
@click.option(
    "--n-simulations",
    "-n",
    type=int,
    default=1000,
    help="Number of Monte Carlo simulations (default: 1000)",
)
@click.option(
    "--method",
    type=click.Choice(["bootstrap", "parametric"], case_sensitive=False),
    default="bootstrap",
    help="Simulation method: bootstrap (resample) or parametric (normal dist)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=str),
    default=None,
    help="Output path for Monte Carlo report",
)
def monte_carlo(
    tickers: tuple[str, ...],
    interval: str,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    strategy: str,
    n_simulations: int,
    method: str,
    seed: int | None,
    output: str | None,
) -> None:
    """
    Run Monte Carlo simulation on backtest results.

    Example:
        crypto-quant monte-carlo --strategy vanilla --n-simulations 2000
    """
    ticker_list = list(tickers)

    logger.info("Starting Monte Carlo simulation...")
    logger.info(f"Tickers: {ticker_list}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Simulations: {n_simulations}")
    logger.info(f"Method: {method}")

    # Create strategy
    if strategy == "vanilla":
        strategy_obj = create_vbo_strategy(
            name="VanillaVBO",
            sma_period=4,
            trend_sma_period=8,
            short_noise_period=4,
            long_noise_period=8,
            exclude_current=False,
            use_trend_filter=True,
            use_noise_filter=True,
        )
    elif strategy == "minimal":
        strategy_obj = create_vbo_strategy(
            name="MinimalVBO",
            sma_period=4,
            trend_sma_period=8,
            short_noise_period=4,
            long_noise_period=8,
            exclude_current=False,
            use_trend_filter=False,
            use_noise_filter=False,
        )
    elif strategy == "legacy":
        strategy_obj = create_vbo_strategy(
            name="LegacyBT",
            sma_period=5,
            trend_sma_period=10,
            short_noise_period=5,
            long_noise_period=10,
            exclude_current=True,
            use_trend_filter=True,
            use_noise_filter=True,
        )
    else:
        logger.error(f"Strategy {strategy} not yet supported for Monte Carlo")
        return

    # Create config
    config = BacktestConfig(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=fee_rate,
        max_slots=max_slots,
        use_cache=True,
    )

    # Run backtest first
    logger.info("Running initial backtest...")
    result = run_backtest(
        strategy=strategy_obj,
        tickers=ticker_list,
        interval=interval,
        config=config,
    )

    # Run Monte Carlo simulation
    logger.info(f"Running {n_simulations} Monte Carlo simulations...")
    mc_result = run_monte_carlo(
        result=result,
        n_simulations=n_simulations,
        method=method,
        random_seed=seed,
    )

    # Print results
    logger.info("\n=== Monte Carlo Simulation Results ===\n")
    logger.info(f"Original Backtest:")
    logger.info(f"  CAGR: {result.cagr:.2f}%")
    logger.info(f"  MDD: {result.mdd:.2f}%")
    logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")

    logger.info(f"\nSimulation Statistics ({n_simulations} runs):")
    logger.info(f"  Mean CAGR: {mc_result.mean_cagr:.2f}% (std: {mc_result.std_cagr:.2f}%)")
    logger.info(f"  Mean MDD: {mc_result.mean_mdd:.2f}% (std: {mc_result.std_mdd:.2f}%)")
    logger.info(f"  Mean Sharpe: {mc_result.mean_sharpe:.2f} (std: {mc_result.std_sharpe:.2f})")

    logger.info(f"\n95% Confidence Intervals:")
    logger.info(
        f"  CAGR: [{mc_result.cagr_ci_lower:.2f}%, {mc_result.cagr_ci_upper:.2f}%]"
    )
    logger.info(
        f"  MDD: [{mc_result.mdd_ci_lower:.2f}%, {mc_result.mdd_ci_upper:.2f}%]"
    )
    logger.info(
        f"  Sharpe: [{mc_result.sharpe_ci_lower:.2f}, {mc_result.sharpe_ci_upper:.2f}]"
    )

    logger.info(f"\nPercentiles:")
    logger.info(f"  CAGR:")
    for p, val in mc_result.cagr_percentiles.items():
        logger.info(f"    {p}th: {val:.2f}%")
    logger.info(f"  MDD:")
    for p, val in mc_result.mdd_percentiles.items():
        logger.info(f"    {p}th: {val:.2f}%")

    # Risk metrics
    from src.backtester.monte_carlo import MonteCarloSimulator

    simulator = MonteCarloSimulator(result)
    prob_loss = simulator.probability_of_loss(mc_result)
    var_95 = simulator.value_at_risk(mc_result, confidence=0.95)
    cvar_95 = simulator.conditional_value_at_risk(mc_result, confidence=0.95)

    logger.info(f"\nRisk Metrics:")
    logger.info(f"  Probability of Loss: {prob_loss*100:.1f}%")
    logger.info(f"  VaR (95%): {var_95:.2f}%")
    logger.info(f"  CVaR (95%): {cvar_95:.2f}%")

    # Generate report if output specified
    if output:
        from src.backtester.report import generate_report
        from datetime import datetime

        output_path = Path(output)
        if output_path.is_dir() or not output_path.suffix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path / f"monte_carlo_{timestamp}.html"
        else:
            output_path = Path(output)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate report for original result (Monte Carlo visualization can be added later)
        generate_report(
            result,
            save_path=output_path,
            show=False,
            format="html",
            strategy_obj=strategy_obj,
            config=config,
            tickers=ticker_list,
        )
        logger.info(f"\nReport saved: {output_path}")
