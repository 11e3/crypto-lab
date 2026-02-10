"""Daily reset operations for trading bot."""

import time
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.exchange import ExchangeError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.execution.bot.bot_facade import TradingBotFacade

logger = get_logger(__name__)

# Constants
YESTERDAY_INDEX = -2
SMA_EXIT_PERIOD = 5
API_RETRY_ATTEMPTS = 3


def calculate_sma_exit(df: pd.DataFrame | None) -> float | None:
    """Calculate SMA for exit condition."""
    if df is None or len(df) < SMA_EXIT_PERIOD + 2:
        return None
    start_idx = -(SMA_EXIT_PERIOD + 2)
    return float(df["close"].iloc[start_idx:YESTERDAY_INDEX].mean())


def process_exits(bot: "TradingBotFacade") -> None:
    """Process exit conditions for all held positions."""
    from src.execution.bot.bot_factory import MIN_DATA_POINTS

    positions = bot.position_manager.get_all_positions()

    for ticker in positions:
        if bot.signal_handler.check_exit_signal(ticker):
            if bot._sell_all(ticker):
                _send_exit_notification(bot, ticker, MIN_DATA_POINTS)
        else:
            bot.telegram.send_trade_signal("HOLD", ticker, 0)


def _send_exit_notification(bot: "TradingBotFacade", ticker: str, min_data_points: int) -> None:
    """Send exit notification with trade details."""
    try:
        df = bot.signal_handler.get_ohlcv_data(ticker, count=min_data_points)
        if df is not None:
            df = bot.strategy.calculate_indicators(df)
            yesterday_close = float(df.iloc[YESTERDAY_INDEX]["close"])
            sma_exit = calculate_sma_exit(df)
            if sma_exit is not None:
                bot.telegram.send_trade_signal(
                    "EXIT",
                    ticker,
                    yesterday_close,
                    reason=f"Close {yesterday_close:.0f} < SMA {sma_exit:.0f}",
                )
    except (ExchangeError, ConnectionError, OSError, KeyError, IndexError, ValueError, TypeError) as e:
        logger.error(f"Error sending exit notification for {ticker}: {e}", exc_info=True)


def recalculate_targets(bot: "TradingBotFacade") -> dict[str, dict[str, float]]:
    """
    Recalculate target prices and metrics for all tickers.

    Returns:
        Updated target_info dictionary
    """
    target_info: dict[str, dict[str, float]] = {}
    msg = "[DAILY UPDATE]\n"
    required_period: Any = bot.strategy_config["trend_sma_period"]

    for ticker in bot.tickers:
        metrics = _calculate_metrics_with_retry(bot, ticker, required_period)
        if metrics:
            target_info[ticker] = metrics
            msg += f"{ticker}: Target {metrics['target']:.0f}, K {metrics['k']:.2f}\n"

    bot.telegram.send(msg)
    logger.info(msg)
    return target_info


def _calculate_metrics_with_retry(
    bot: "TradingBotFacade", ticker: str, required_period: int
) -> dict[str, float] | None:
    """Calculate metrics with retry on failure."""
    for attempt in range(API_RETRY_ATTEMPTS):
        metrics = bot.signal_handler.calculate_metrics(ticker, required_period)
        if metrics:
            return metrics
        if attempt < API_RETRY_ATTEMPTS - 1:
            time.sleep(bot.bot_config["api_retry_delay"])
    return None
