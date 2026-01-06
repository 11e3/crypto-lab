"""
Data collection command.

Collects OHLCV data from cryptocurrency exchanges.
"""

import click

from src.data.collector import Interval
from src.data.collector_factory import DataCollectorFactory
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@click.command(name="collect")
@click.option(
    "--tickers",
    "-t",
    multiple=True,
    default=["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE", "KRW-TRX"],
    help="Trading pair tickers to collect data for",
)
@click.option(
    "--intervals",
    "-i",
    multiple=True,
    type=click.Choice(
        [
            "minute1",
            "minute3",
            "minute5",
            "minute10",
            "minute15",
            "minute30",
            "minute60",
            "minute240",
            "day",
            "week",
            "month",
        ],
        case_sensitive=False,
    ),
    default=["minute240", "day", "week"],
    help="Data intervals to collect",
)
@click.option(
    "--full-refresh",
    is_flag=True,
    default=False,
    help="Force full data refresh (ignore existing data)",
)
def collect(tickers: tuple[str, ...], intervals: tuple[str, ...], full_refresh: bool) -> None:
    """
    Collect OHLCV data from cryptocurrency exchanges.

    Performs incremental update if data already exists, unless --full-refresh is used.
    """
    ticker_list = list(tickers)
    interval_list: list[Interval] = list(intervals)  # type: ignore

    logger.info("Starting data collection...")
    logger.info(f"Tickers: {ticker_list}")
    logger.info(f"Intervals: {interval_list}")
    logger.info(f"Full refresh: {full_refresh}")

    collector = DataCollectorFactory.create()

    results = collector.collect_multiple(ticker_list, interval_list, full_refresh=full_refresh)

    # Summary
    logger.info("\n=== Collection Summary ===")
    total_new = 0
    for key, count in results.items():
        if count >= 0:
            logger.info(f"  {key}: +{count} candles")
            total_new += count
        else:
            logger.warning(f"  {key}: FAILED")

    logger.info(f"\nTotal new candles: {total_new}")
