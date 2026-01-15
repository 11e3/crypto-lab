"""
Bot recovery and target initialization.

Separates initialization/recovery logic from component creation (SRP).
"""

from __future__ import annotations

import time
from typing import Any

from src.exchange import Exchange
from src.execution.position_manager import PositionManager
from src.execution.signal_handler import SignalHandler
from src.utils.logger import get_logger

logger = get_logger(__name__)

API_RETRY_ATTEMPTS = 3


def initialize_targets(
    tickers: list[str],
    signal_handler: SignalHandler,
    strategy_config: dict[str, Any],
    bot_config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """
    Initialize target prices and metrics for all tickers.

    Args:
        tickers: List of trading pair tickers
        signal_handler: Signal handler instance
        strategy_config: Strategy configuration
        bot_config: Bot configuration

    Returns:
        Dictionary of ticker -> metrics
    """
    logger.info("Initializing targets...")
    target_info: dict[str, dict[str, float]] = {}
    required_period = strategy_config["trend_sma_period"]

    for ticker in tickers:
        for attempt in range(API_RETRY_ATTEMPTS):
            metrics = signal_handler.calculate_metrics(ticker, required_period)
            if metrics:
                target_info[ticker] = metrics
                logger.info(
                    f"[{ticker}] Target: {metrics['target']:.0f} | "
                    f"K: {metrics['k']:.2f} vs Base: {metrics['long_noise']:.2f} | "
                    f"SMA: {metrics['sma']:.0f} Trend: {metrics['sma_trend']:.0f}"
                )
                break
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(bot_config["api_retry_delay"])

    return target_info


def check_existing_holdings(
    tickers: list[str],
    exchange: Exchange,
    position_manager: PositionManager,
    trading_config: dict[str, Any],
) -> None:
    """
    Check and recover existing holdings on bot restart.

    Args:
        tickers: List of trading pair tickers
        exchange: Exchange instance
        position_manager: Position manager instance
        trading_config: Trading configuration
    """
    logger.info("Checking existing holdings...")
    min_amount = trading_config["min_order_amount"]

    for ticker in tickers:
        try:
            currency = ticker.split("-")[1]
            balance = exchange.get_balance(currency)
            curr_price = exchange.get_current_price(ticker)

            if (
                balance.available > 0
                and curr_price > 0
                and (balance.available * curr_price > min_amount)
            ):
                position_manager.add_position(
                    ticker=ticker,
                    entry_price=curr_price,
                    amount=balance.available,
                )
                logger.info(f"Recovered: Holding {ticker}")
        except Exception as e:
            logger.error(f"Error checking holdings for {ticker}: {e}", exc_info=True)
