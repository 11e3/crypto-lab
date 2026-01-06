"""
Backtest command.

Runs backtesting on historical data.
"""

import click
import numpy as np

from src.backtester import BacktestConfig, run_backtest
from src.strategies.mean_reversion import (
    MeanReversionStrategy,
    SimpleMeanReversionStrategy,
)
from src.strategies.momentum import MomentumStrategy, SimpleMomentumStrategy
from src.strategies.pair_trading import PairTradingStrategy
from src.strategies.volatility_breakout import create_vbo_strategy
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@click.command(name="backtest")
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
    "--strategy",
    type=click.Choice(
        [
            "vanilla",
            "minimal",
            "legacy",
            "momentum",
            "simple-momentum",
            "mean-reversion",
            "simple-mean-reversion",
            "pair-trading",
        ],
        case_sensitive=False,
    ),
    default="vanilla",
    help="Strategy variant to use (default: vanilla - Vanilla VBO strategy)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=str),
    default=None,
    help="Output path for report. If True or empty, auto-generates reports/{timestamp}_{strategy}.html",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable indicator cache",
)
@click.option(
    "--sma-period",
    type=int,
    default=None,
    help="SMA period for VBO strategy (default: strategy-specific)",
)
@click.option(
    "--trend-sma-period",
    type=int,
    default=None,
    help="Trend SMA period for VBO strategy (default: strategy-specific)",
)
@click.option(
    "--short-noise-period",
    type=int,
    default=None,
    help="Short noise period for VBO strategy (default: strategy-specific)",
)
@click.option(
    "--long-noise-period",
    type=int,
    default=None,
    help="Long noise period for VBO strategy (default: strategy-specific)",
)
@click.option(
    "--exclude-current",
    is_flag=True,
    default=None,
    help="Exclude current bar from calculations (VBO strategy)",
)
@click.option(
    "--position-sizing",
    type=click.Choice(["equal", "volatility", "fixed-risk", "inverse-volatility", "mpt", "risk_parity", "kelly"], case_sensitive=False),
    default="equal",
    help="Position sizing method: equal (default), volatility, fixed-risk, inverse-volatility, mpt, risk_parity, kelly",
)
@click.option(
    "--position-sizing-risk",
    type=float,
    default=0.02,
    help="Target risk per position for fixed-risk method (default: 0.02 = 2%%)",
)
@click.option(
    "--position-sizing-lookback",
    type=int,
    default=20,
    help="Lookback period for volatility calculation (default: 20)",
)
@click.option(
    "--stop-loss",
    type=float,
    default=None,
    help="Stop loss as percentage (e.g., 0.05 = 5%%)",
)
@click.option(
    "--take-profit",
    type=float,
    default=None,
    help="Take profit as percentage (e.g., 0.10 = 10%%)",
)
@click.option(
    "--trailing-stop",
    type=float,
    default=None,
    help="Trailing stop as percentage (e.g., 0.05 = 5%%)",
)
@click.option(
    "--portfolio-optimization",
    type=click.Choice(["mpt", "risk_parity", "kelly"], case_sensitive=False),
    default=None,
    help="Portfolio optimization method: mpt (Modern Portfolio Theory), risk_parity, kelly (overrides position-sizing if set)",
)
@click.option(
    "--risk-free-rate",
    type=float,
    default=0.0,
    help="Risk-free rate for MPT optimization (annualized, default: 0.0)",
)
@click.option(
    "--max-kelly",
    type=float,
    default=0.25,
    help="Maximum Kelly percentage for Kelly Criterion (default: 0.25 = 25%%)",
)
def backtest(
    tickers: tuple[str, ...],
    interval: str,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    strategy: str,
    output: str | None,
    no_cache: bool,
    sma_period: int | None,
    trend_sma_period: int | None,
    short_noise_period: int | None,
    long_noise_period: int | None,
    exclude_current: bool | None,
    position_sizing: str,
    position_sizing_risk: float,
    position_sizing_lookback: int,
    stop_loss: float | None,
    take_profit: float | None,
    trailing_stop: float | None,
    portfolio_optimization: str | None,
    risk_free_rate: float,
    max_kelly: float,
) -> None:
    """
    Run backtest on historical data.

    Example:
        crypto-quant backtest --tickers KRW-BTC KRW-ETH --interval day
    """
    ticker_list = list(tickers)

    logger.info("Starting backtest...")
    logger.info(f"Tickers: {ticker_list}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Initial capital: {initial_capital}")
    logger.info(f"Fee rate: {fee_rate}")
    logger.info(f"Max slots: {max_slots}")

    # Create strategy based on variant
    # Build kwargs for VBO strategies
    vbo_kwargs: dict = {}
    if sma_period is not None:
        vbo_kwargs["sma_period"] = sma_period
    if trend_sma_period is not None:
        vbo_kwargs["trend_sma_period"] = trend_sma_period
    if short_noise_period is not None:
        vbo_kwargs["short_noise_period"] = short_noise_period
    if long_noise_period is not None:
        vbo_kwargs["long_noise_period"] = long_noise_period
    if exclude_current is not None:
        vbo_kwargs["exclude_current"] = exclude_current
    
    if strategy == "vanilla":
        strategy_obj = create_vbo_strategy(
            name="VanillaVBO",
            use_trend_filter=True,
            use_noise_filter=True,
            **vbo_kwargs,
        )
    elif strategy == "minimal":
        strategy_obj = create_vbo_strategy(
            name="MinimalVBO",
            use_trend_filter=False,
            use_noise_filter=False,
            **vbo_kwargs,
        )
    elif strategy == "legacy":
        # Default legacy parameters
        legacy_kwargs = {
            "sma_period": sma_period if sma_period is not None else 5,
            "trend_sma_period": trend_sma_period if trend_sma_period is not None else 10,
            "short_noise_period": short_noise_period if short_noise_period is not None else 5,
            "long_noise_period": long_noise_period if long_noise_period is not None else 10,
            "exclude_current": exclude_current if exclude_current is not None else True,
        }
        strategy_obj = create_vbo_strategy(
            name="LegacyBT",
            use_trend_filter=True,
            use_noise_filter=True,
            **legacy_kwargs,
        )
    elif strategy == "momentum":
        strategy_obj = MomentumStrategy(name="MomentumStrategy")
    elif strategy == "simple-momentum":
        strategy_obj = SimpleMomentumStrategy(name="SimpleMomentum")
    elif strategy == "mean-reversion":
        strategy_obj = MeanReversionStrategy(name="MeanReversionStrategy")
    elif strategy == "simple-mean-reversion":
        strategy_obj = SimpleMeanReversionStrategy(name="SimpleMeanReversion")
    elif strategy == "pair-trading":
        # Pair trading requires exactly 2 tickers
        if len(ticker_list) != 2:
            logger.error(
                f"Pair trading strategy requires exactly 2 tickers, "
                f"got {len(ticker_list)}: {ticker_list}"
            )
            raise ValueError(
                f"Pair trading requires exactly 2 tickers, got {len(ticker_list)}"
            )
        strategy_obj = PairTradingStrategy(name="PairTradingStrategy")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create config
    config = BacktestConfig(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=fee_rate,  # Use same as fee rate
        max_slots=max_slots,
        position_sizing=position_sizing,
        position_sizing_risk_pct=position_sizing_risk,
        position_sizing_lookback=position_sizing_lookback,
        use_cache=not no_cache,
    )

    # Run backtest
    result = run_backtest(
        strategy=strategy_obj,
        tickers=ticker_list,
        interval=interval,
        config=config,
    )

    # Print summary
    logger.info("\n=== Backtest Results ===")
    period_days = (result.dates[-1] - result.dates[0]).days if len(result.dates) > 0 else 0
    logger.info(f"Period: {period_days} days")
    logger.info(f"Total Return: {result.total_return:.2f}%")
    logger.info(f"CAGR: {result.cagr:.2f}%")
    logger.info(f"MDD: {result.mdd:.2f}%")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    logger.info(f"Calmar Ratio: {result.calmar_ratio:.2f}")
    logger.info(f"Trade Count: {result.total_trades}")
    logger.info(f"Win Rate: {result.win_rate:.2f}%")
    logger.info(f"Profit Factor: {result.profit_factor:.2f}")

    # Print risk metrics if available
    if result.risk_metrics:
        # Determine period label based on interval
        interval_str = result.interval or "day"
        period_label = {
            "day": "daily",
            "minute240": "4-hour",
            "week": "weekly",
        }.get(interval_str, interval_str)
        
        logger.info("\n--- Risk Metrics ---")
        logger.info(f"VaR (95%, {period_label}): {result.risk_metrics.var_95*100:.2f}%")
        logger.info(f"CVaR (95%, {period_label}): {result.risk_metrics.cvar_95*100:.2f}%")
        logger.info(f"VaR (99%, {period_label}): {result.risk_metrics.var_99*100:.2f}%")
        logger.info(f"CVaR (99%, {period_label}): {result.risk_metrics.cvar_99*100:.2f}%")
        logger.info(f"Portfolio Volatility (annualized): {result.risk_metrics.portfolio_volatility*100:.2f}%")
        
        # Correlation metrics (check if calculated, not just != 0.0)
        if result.risk_metrics.avg_correlation is not None and not np.isnan(result.risk_metrics.avg_correlation):
            logger.info(f"Avg Correlation: {result.risk_metrics.avg_correlation:.3f}")
            logger.info(f"Max Correlation: {result.risk_metrics.max_correlation:.3f}")
            logger.info(f"Min Correlation: {result.risk_metrics.min_correlation:.3f}")
        
        # Position concentration (check if calculated)
        if result.risk_metrics.max_position_pct is not None and result.risk_metrics.max_position_pct > 0:
            logger.info(f"Max Position %: {result.risk_metrics.max_position_pct*100:.2f}%")
            logger.info(f"Position Concentration (HHI): {result.risk_metrics.position_concentration:.3f}")
        
        if result.risk_metrics.portfolio_beta is not None:
            logger.info(f"Portfolio Beta: {result.risk_metrics.portfolio_beta:.2f}")

    # Generate report if output specified
    if output:
        from datetime import datetime
        from pathlib import Path

        from src.backtester.report import generate_report

        # If output is "True" (string) or empty string, auto-generate filename
        if output == "True" or output == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name_safe = result.strategy_name.replace(" ", "_").lower()
            save_path = Path(f"reports/{timestamp}_{strategy_name_safe}.html")
        else:
            save_path = Path(output) if isinstance(output, str) else output
        
        # Ensure reports directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension (default to HTML if no extension)
        if save_path.suffix.lower() in [".html", ".htm"]:
            format_type = "html"
        elif save_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            format_type = "png"
        else:
            # Default to HTML if no extension
            format_type = "html"
            if not save_path.suffix:
                save_path = save_path.with_suffix(".html")
        
        # Generate report with strategy and config info
        generate_report(
            result,
            save_path=save_path,
            show=False,
            format=format_type,
            strategy_obj=strategy_obj,
            config=config,
            tickers=ticker_list,
        )
        logger.info(f"\nReport saved to: {save_path}")
