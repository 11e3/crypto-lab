"""Trading bot main loop execution logic."""

import contextlib
import datetime
import os
import sys
import time
from typing import TYPE_CHECKING, Any

import pyupbit

from src.exchange import ExchangeError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.execution.bot.bot_facade import TradingBotFacade

logger = get_logger(__name__)

# Constants
DAILY_RESET_WINDOW_SECONDS = 10


def run_trading_loop(bot: "TradingBotFacade") -> None:
    """
    Run the main trading loop.

    Args:
        bot: TradingBotFacade instance with initialized components
    """
    # Check if testing environment
    if _should_block_in_test(bot):
        logger.warning("Bot.run() blocked during testing. Use --allow-test-run flag.")
        return

    logger.info("Starting Trading Bot (VBO Strategy)...")

    # Validate API connection
    if not _validate_api_connection(bot):
        return

    bot.telegram.send("🚀 Bot Started: VBO Strategy")
    bot.initialize_targets()
    bot.check_existing_holdings()
    logger.info("Entering Trading Loop...")

    # Start WebSocket
    wm = _create_websocket(bot.tickers)
    if wm is None:
        return  # pragma: no cover

    _main_loop(bot, wm)  # pragma: no cover


def _should_block_in_test(bot: "TradingBotFacade") -> bool:
    """Check if bot should be blocked during testing."""
    from unittest.mock import MagicMock

    is_testing = (
        "pytest" in sys.modules
        or "unittest" in sys.modules
        or "PYTEST_CURRENT_TEST" in os.environ
        or any("test" in arg.lower() for arg in sys.argv)
    )

    if not is_testing:
        return False

    if "--allow-test-run" in sys.argv:
        return False

    telegram = bot.telegram
    if not telegram or isinstance(telegram, MagicMock):
        return False

    if not hasattr(telegram, "enabled") or not telegram.enabled:
        return False

    if not hasattr(telegram, "token") or not telegram.token:
        return False

    return "YOUR_" not in telegram.token


def _validate_api_connection(bot: "TradingBotFacade") -> bool:
    """Validate API connection."""
    try:
        bot.exchange.get_balance("KRW")
        logger.info("SUCCESS: API Keys valid.")
        return True
    except (ExchangeError, ConnectionError, OSError) as e:
        logger.error(f"!!! API CONNECTION FAILED: {e}")
        time.sleep(3)
        return False


def _create_websocket(tickers: list[str]) -> Any:
    """Create WebSocket manager."""
    try:
        return pyupbit.WebSocketManager("ticker", tickers)
    except Exception as e:
        logger.error(f"WebSocket Connection Failed: {e}")  # pragma: no cover
        return None  # pragma: no cover


def _main_loop(bot: "TradingBotFacade", wm: Any) -> None:  # pragma: no cover
    """Main trading loop."""
    while True:
        try:
            data = wm.get()
            if data["type"] == "ticker":
                ticker = data["code"]
                current_price = data["trade_price"]
                now = datetime.datetime.now()

                reset_hour = bot.bot_config["daily_reset_hour"]
                reset_minute = bot.bot_config["daily_reset_minute"]

                if _is_daily_reset_time(now, reset_hour, reset_minute):
                    wm.terminate()
                    bot.daily_reset()
                    time.sleep(6)
                    wm = pyupbit.WebSocketManager("ticker", bot.tickers)
                    continue

                bot.process_ticker_update(ticker, current_price)

        except Exception as e:
            logger.error(f"Loop Error: {e}", exc_info=True)
            time.sleep(bot.bot_config["websocket_reconnect_delay"])
            with contextlib.suppress(Exception):
                wm.terminate()
            wm = pyupbit.WebSocketManager("ticker", bot.tickers)


def _is_daily_reset_time(now: datetime.datetime, reset_hour: int, reset_minute: int) -> bool:
    """Check if current time is daily reset time."""
    return (
        now.hour == reset_hour
        and now.minute == reset_minute
        and now.second <= DAILY_RESET_WINDOW_SECONDS
    )
