"""
TradingBot Facade - Simplified interface for trading operations.

Provides a high-level interface that delegates to specialized managers.
"""

import contextlib
import datetime
import os
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyupbit

from src.config.loader import get_config
from src.exchange import Exchange, ExchangeFactory
from src.execution.advanced_orders import AdvancedOrderManager
from src.execution.event_bus import get_event_bus
from src.execution.handlers.notification_handler import NotificationHandler
from src.execution.handlers.trade_handler import TradeHandler
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.signal_handler import SignalHandler
from src.strategies.volatility_breakout import VanillaVBO
from src.utils.logger import get_logger, setup_logging
from src.utils.telegram import get_notifier

if TYPE_CHECKING:
    pass

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Constants
YESTERDAY_INDEX = -2  # Yesterday's confirmed candle (excluding today's incomplete)
MIN_DATA_POINTS = 10  # Minimum data points for reliable calculations
SMA_EXIT_PERIOD = 5  # SMA period for exit condition
API_RETRY_ATTEMPTS = 3  # Number of retry attempts for API calls
DAILY_RESET_WINDOW_SECONDS = 10  # Window for daily reset trigger


class TradingBotFacade:
    """
    Facade for trading bot operations.

    Provides a simplified interface that coordinates between:
    - Exchange: Market data and order execution
    - PositionManager: Position tracking
    - OrderManager: Order execution
    - SignalHandler: Signal generation
    - Strategy: Trading strategy logic
    """

    def __init__(
        self,
        exchange: Exchange | None = None,
        position_manager: PositionManager | None = None,
        order_manager: OrderManager | None = None,
        signal_handler: SignalHandler | None = None,
        strategy: VanillaVBO | None = None,
        config_path: Path | None = None,
    ) -> None:
        """
        Initialize trading bot facade with dependency injection.

        Args:
            exchange: Exchange instance (created if None)
            position_manager: PositionManager instance (created if None)
            order_manager: OrderManager instance (created if None)
            signal_handler: SignalHandler instance (created if None)
            strategy: Strategy instance (created if None)
            config_path: Path to config file (defaults to config/settings.yaml)

        Raises:
            ValueError: If Upbit API keys are not configured
        """
        self.config = get_config(config_path)
        self.trading_config = self.config.get_trading_config()
        self.strategy_config = self.config.get_strategy_config()
        self.bot_config = self.config.get_bot_config()

        # Initialize Exchange (dependency injection)
        if exchange is None:
            exchange_name = self.config.get_exchange_name()
            self.exchange: Exchange = ExchangeFactory.create(exchange_name)
        else:
            self.exchange = exchange

        # Initialize Telegram
        telegram_config = self.config.get_telegram_config()
        self.telegram = get_notifier(
            token=telegram_config["token"],
            chat_id=telegram_config["chat_id"],
            enabled=telegram_config["enabled"],
        )

        # Initialize Strategy (dependency injection)
        if strategy is None:
            self.strategy = VanillaVBO(
                name=self.strategy_config["name"],
                sma_period=self.strategy_config["sma_period"],
                trend_sma_period=self.strategy_config["trend_sma_period"],
                short_noise_period=self.strategy_config["short_noise_period"],
                long_noise_period=self.strategy_config["long_noise_period"],
                exclude_current=self.strategy_config.get("exclude_current", True),
            )
        else:
            self.strategy = strategy

        # Initialize event bus and handlers
        self.event_bus = get_event_bus()
        self.trade_handler = TradeHandler()
        self.notification_handler = NotificationHandler(self.telegram)

        # Initialize Managers (dependency injection)
        if position_manager is None:
            self.position_manager = PositionManager(self.exchange, publish_events=True)
        else:
            self.position_manager = position_manager

        if order_manager is None:
            self.order_manager = OrderManager(self.exchange, publish_events=True)
        else:
            self.order_manager = order_manager

        if signal_handler is None:
            self.signal_handler = SignalHandler(
                strategy=self.strategy,
                exchange=self.exchange,
                min_data_points=MIN_DATA_POINTS,
                publish_events=True,
            )
        else:
            self.signal_handler = signal_handler

        # Initialize advanced order manager
        self.advanced_order_manager = AdvancedOrderManager()

        # Trading configuration
        self.tickers = self.trading_config["tickers"]
        self.target_info: dict[str, dict[str, float]] = {}

    def get_krw_balance(self) -> float:
        """Get KRW balance."""
        try:
            balance = self.exchange.get_balance("KRW")
            return balance.available
        except Exception as e:
            logger.error(f"Error getting KRW balance: {e}", exc_info=True)
            return 0.0

    def initialize_targets(self) -> None:
        """Initialize target prices and metrics for all tickers."""
        logger.info("Initializing targets...")
        required_period = self.strategy_config["trend_sma_period"]

        for ticker in self.tickers:
            for attempt in range(API_RETRY_ATTEMPTS):
                metrics = self.signal_handler.calculate_metrics(ticker, required_period)
                if metrics:
                    self.target_info[ticker] = metrics
                    logger.info(
                        f"[{ticker}] Target: {metrics['target']:.0f} | "
                        f"K: {metrics['k']:.2f} vs Base: {metrics['long_noise']:.2f} | "
                        f"SMA: {metrics['sma']:.0f} Trend: {metrics['sma_trend']:.0f}"
                    )
                    break
                if attempt < API_RETRY_ATTEMPTS - 1:
                    time.sleep(self.bot_config["api_retry_delay"])

    def check_existing_holdings(self) -> None:
        """Check and recover existing holdings on bot restart."""
        logger.info("Checking existing holdings...")
        min_amount = self.trading_config["min_order_amount"]

        for ticker in self.tickers:
            try:
                currency = ticker.split("-")[1]
                balance = self.exchange.get_balance(currency)
                curr_price = self.exchange.get_current_price(ticker)

                if (
                    balance.available > 0
                    and curr_price > 0
                    and (balance.available * curr_price > min_amount)
                ):
                    # Recover position
                    self.position_manager.add_position(
                        ticker=ticker,
                        entry_price=curr_price,  # Use current price as approximation
                        amount=balance.available,
                    )
                    logger.info(f"✅ Recovered: Holding {ticker}")
            except Exception as e:
                logger.error(f"Error checking holdings for {ticker}: {e}", exc_info=True)

    def _calculate_sma_exit(self, df: pd.DataFrame | None) -> float | None:
        """
        Calculate SMA for exit condition.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            SMA value or None if insufficient data
        """
        if df is None or len(df) < SMA_EXIT_PERIOD + 2:
            return None

        sma_period = SMA_EXIT_PERIOD
        start_idx = -(sma_period + 2)
        end_idx = YESTERDAY_INDEX
        return float(df["close"].iloc[start_idx:end_idx].mean())

    def _process_exits(self) -> None:
        """Process exit conditions for all held positions."""
        positions = self.position_manager.get_all_positions()

        for ticker in positions:
            if self.signal_handler.check_exit_signal(ticker):
                if self._sell_all(ticker):
                    # Send exit notification with details
                    try:
                        df = self.signal_handler.get_ohlcv_data(ticker, count=MIN_DATA_POINTS)
                        if df is not None:
                            df = self.strategy.calculate_indicators(df)
                            yesterday_close = float(df.iloc[YESTERDAY_INDEX]["close"])
                            sma_exit = self._calculate_sma_exit(df)
                            if sma_exit is not None:
                                self.telegram.send_trade_signal(
                                    "EXIT",
                                    ticker,
                                    yesterday_close,
                                    reason=f"Close {yesterday_close:.0f} < SMA {sma_exit:.0f}",
                                )
                    except Exception as e:
                        logger.error(
                            f"Error sending exit notification for {ticker}: {e}", exc_info=True
                        )
            else:
                self.telegram.send_trade_signal("HOLD", ticker, 0)

    def _recalculate_targets(self) -> None:
        """Recalculate target prices and metrics for all tickers."""
        self.target_info = {}
        msg = "[DAILY UPDATE]\n"
        required_period = self.strategy_config["trend_sma_period"]

        for ticker in self.tickers:
            for attempt in range(API_RETRY_ATTEMPTS):
                metrics = self.signal_handler.calculate_metrics(ticker, required_period)
                if metrics:
                    self.target_info[ticker] = metrics
                    msg += f"{ticker}: Target {metrics['target']:.0f}, K {metrics['k']:.2f}\n"
                    break
                if attempt < API_RETRY_ATTEMPTS - 1:
                    time.sleep(self.bot_config["api_retry_delay"])

        self.telegram.send(msg)
        logger.info(msg)

    def daily_reset(self) -> None:
        """Perform daily reset: check exits and recalculate targets."""
        logger.info("Performing daily reset...")
        self._process_exits()
        self._recalculate_targets()

    def _calculate_buy_amount(self) -> float:
        """
        Calculate buy amount based on available cash and slots.

        Returns:
            Buy amount in KRW, 0.0 if insufficient funds
        """
        krw_bal = self.get_krw_balance()
        min_amount = self.trading_config["min_order_amount"]

        if krw_bal <= min_amount:
            return 0.0

        current_positions = self.position_manager.get_position_count()
        max_slots = self.trading_config["max_slots"]
        available_slots = max_slots - current_positions

        if available_slots <= 0:
            return 0.0

        fee_rate = self.trading_config["fee_rate"]
        buy_amount = float((krw_bal / available_slots) * (1 - fee_rate))

        return buy_amount if buy_amount > min_amount else 0.0

    def _sell_all(self, ticker: str) -> bool:
        """
        Sell all holdings for a ticker.

        Args:
            ticker: Trading pair ticker

        Returns:
            True if sold successfully
        """
        try:
            min_amount = self.trading_config["min_order_amount"]
            order = self.order_manager.sell_all(ticker, min_amount)

            if order and order.order_id:
                # Remove position
                self.position_manager.remove_position(ticker)

                # Send notification (only if not in test environment)
                import sys

                is_testing = (
                    "pytest" in sys.modules
                    or "unittest" in sys.modules
                    or "PYTEST_CURRENT_TEST" in os.environ
                    or any("test" in arg.lower() for arg in sys.argv)
                )

                if not is_testing:
                    try:
                        curr_price = self.exchange.get_current_price(ticker)
                        currency = ticker.split("-")[1]
                        balance = self.exchange.get_balance(currency)
                        self.telegram.send_trade_signal(
                            "SELL",
                            ticker,
                            curr_price,
                            amount=balance.available,
                        )
                    except Exception:
                        pass  # Notification is optional

                logger.info(f"Sold all {ticker}")
                return True
        except Exception as e:
            logger.error(f"Sell error for {ticker}: {e}", exc_info=True)
        return False

    def _execute_buy_order(self, ticker: str, current_price: float, buy_amount: float) -> bool:
        """
        Execute buy market order.

        Args:
            ticker: Trading pair ticker
            current_price: Current market price
            buy_amount: Amount to buy in KRW

        Returns:
            True if order executed successfully
        """
        try:
            min_amount = self.trading_config["min_order_amount"]
            order = self.order_manager.place_buy_order(ticker, buy_amount, min_amount)

            if order and order.order_id:
                # Add position (approximate amount will be updated when order fills)
                estimated_amount = buy_amount / current_price if current_price > 0 else 0.0
                self.position_manager.add_position(
                    ticker=ticker,
                    entry_price=current_price,
                    amount=estimated_amount,
                )

                # Create advanced orders if configured
                stop_loss_pct = self.trading_config.get("stop_loss_pct")
                take_profit_pct = self.trading_config.get("take_profit_pct")
                trailing_stop_pct = self.trading_config.get("trailing_stop_pct")

                if stop_loss_pct is not None:
                    self.advanced_order_manager.create_stop_loss(
                        ticker=ticker,
                        entry_price=current_price,
                        entry_date=date.today(),
                        amount=estimated_amount,
                        stop_loss_pct=stop_loss_pct,
                    )

                if take_profit_pct is not None:
                    self.advanced_order_manager.create_take_profit(
                        ticker=ticker,
                        entry_price=current_price,
                        entry_date=date.today(),
                        amount=estimated_amount,
                        take_profit_pct=take_profit_pct,
                    )

                if trailing_stop_pct is not None:
                    self.advanced_order_manager.create_trailing_stop(
                        ticker=ticker,
                        entry_price=current_price,
                        entry_date=date.today(),
                        amount=estimated_amount,
                        trailing_stop_pct=trailing_stop_pct,
                    )

                metrics = self.target_info.get(ticker, {})
                self.telegram.send_trade_signal(
                    "BUY",
                    ticker,
                    current_price,
                    target=metrics.get("target", 0),
                    noise=f"{metrics.get('k', 0):.2f} < {metrics.get('long_noise', 0):.2f}",
                )
                logger.info(
                    f"🔥 BUY {ticker} @ {current_price:.0f} | "
                    f"Target: {metrics.get('target', 0):.0f}"
                )
                return True
            else:
                logger.warning(f"Buy failed for {ticker}: order not created")
                return False
        except Exception as e:
            logger.error(f"Buy error for {ticker}: {e}", exc_info=True)
            return False

    def process_ticker_update(self, ticker: str, current_price: float) -> None:
        """
        Process real-time ticker update and check for entry signals.

        Args:
            ticker: Trading pair ticker
            current_price: Current market price
        """
        # Check advanced orders first (stop loss, take profit, trailing stop)
        if self.position_manager.has_position(ticker):
            # Get OHLCV data for advanced order checking
            try:
                ohlcv_data = self.signal_handler.get_ohlcv_data(ticker, count=1)
                if ohlcv_data is not None and len(ohlcv_data) > 0:
                    latest = ohlcv_data.iloc[-1]
                    low_price = latest.get("low", current_price)
                    high_price = latest.get("high", current_price)

                    # Check advanced orders
                    triggered = self.advanced_order_manager.check_orders(
                        ticker=ticker,
                        current_price=current_price,
                        current_date=date.today(),
                        low_price=low_price,
                        high_price=high_price,
                    )

                    # If any order triggered, execute sell
                    if triggered:
                        logger.info(
                            f"Advanced order triggered for {ticker}: "
                            f"{triggered[0].order_type.value} @ {triggered[0].triggered_price:.0f}"
                        )
                        # Execute sell order
                        position = self.position_manager.get_position(ticker)
                        if position:
                            min_order_amount = self.trading_config.get("min_order_amount", 5000.0)
                            sell_order = self.order_manager.place_sell_order(
                                ticker=ticker,
                                amount=position.amount,
                                min_order_amount=min_order_amount,
                            )
                            if sell_order:
                                # Remove position after successful sell
                                self.position_manager.remove_position(ticker)
                                # Cancel remaining advanced orders
                                self.advanced_order_manager.cancel_all_orders(ticker=ticker)
                                logger.info(f"Sold {ticker} due to advanced order trigger")
                        return

            except Exception as e:
                logger.warning(f"Error checking advanced orders for {ticker}: {e}")

        # Skip if already holding (after advanced order check)
        if self.position_manager.has_position(ticker):
            return

        # Check entry conditions
        metrics = self.target_info.get(ticker)
        target_price = metrics.get("target") if metrics else None
        if not self.signal_handler.check_entry_signal(ticker, current_price, target_price):
            return

        # Calculate and validate buy amount
        buy_amount = self._calculate_buy_amount()
        if buy_amount <= 0:
            return

        # Execute buy order
        self._execute_buy_order(ticker, current_price, buy_amount)

    def run(self) -> None:
        """Run the trading bot main loop."""
        # Safety check: prevent running during tests (only if Telegram is real)
        # Allow mocks to work in tests
        import sys
        from unittest.mock import MagicMock

        is_testing = (
            "pytest" in sys.modules
            or "unittest" in sys.modules
            or "PYTEST_CURRENT_TEST" in os.environ
            or any("test" in arg.lower() for arg in sys.argv)
        )

        # Only block if Telegram is real (not a mock) and enabled
        # Tests use mocks which should work
        should_block = (
            is_testing
            and "--allow-test-run" not in sys.argv
            and self.telegram
            and not isinstance(self.telegram, MagicMock)
            and hasattr(self.telegram, "enabled")
            and self.telegram.enabled
            and hasattr(self.telegram, "token")
            and self.telegram.token
            and "YOUR_" not in self.telegram.token
        )

        if should_block:
            logger.warning("Bot.run() called during testing - blocking execution for safety")
            logger.warning("If you need to test bot.run(), use --allow-test-run flag")
            return

        logger.info("Starting Trading Bot (VBO Strategy)...")

        # Test API connection
        try:
            self.exchange.get_balance("KRW")
            logger.info("SUCCESS: API Keys are valid and working.")
        except Exception as e:
            logger.error(f"!!! API CONNECTION FAILED: {e}")
            time.sleep(3)
            return

        self.telegram.send("🚀 Bot Started: VBO Strategy")

        # Initialize
        self.initialize_targets()
        self.check_existing_holdings()

        logger.info("Entering Trading Loop...")

        # WebSocket connection
        try:
            wm = pyupbit.WebSocketManager(
                "ticker", self.tickers
            )  # pragma: no cover (WebSocket, difficult to test)
        except Exception as e:
            logger.error(f"WebSocket Connection Failed: {e}")  # pragma: no cover
            return  # pragma: no cover

        while True:  # pragma: no cover (infinite loop with WebSocket, difficult to test)
            try:
                data = wm.get()  # pragma: no cover
                if data["type"] == "ticker":  # pragma: no cover
                    ticker = data["code"]  # pragma: no cover
                    current_price = data["trade_price"]  # pragma: no cover

                    now = datetime.datetime.now()  # pragma: no cover

                    # Daily reset check
                    reset_hour = self.bot_config["daily_reset_hour"]  # pragma: no cover
                    reset_minute = self.bot_config["daily_reset_minute"]  # pragma: no cover

                    if (  # pragma: no cover
                        now.hour == reset_hour
                        and now.minute == reset_minute
                        and now.second <= DAILY_RESET_WINDOW_SECONDS
                    ):
                        wm.terminate()  # pragma: no cover
                        self.daily_reset()  # pragma: no cover
                        time.sleep(6)  # Wait before reconnecting  # pragma: no cover
                        wm = pyupbit.WebSocketManager("ticker", self.tickers)  # pragma: no cover
                        continue  # pragma: no cover

                    # Process ticker update
                    self.process_ticker_update(ticker, current_price)  # pragma: no cover

            except Exception as e:  # pragma: no cover
                logger.error(f"Loop Error: {e}", exc_info=True)  # pragma: no cover
                time.sleep(self.bot_config["websocket_reconnect_delay"])  # pragma: no cover
                with contextlib.suppress(Exception):  # pragma: no cover
                    wm.terminate()  # pragma: no cover
                wm = pyupbit.WebSocketManager("ticker", self.tickers)  # pragma: no cover


def create_bot(config_path: Path | None = None) -> TradingBotFacade:
    """
    Factory function to create a TradingBotFacade with default dependencies.

    Args:
        config_path: Path to config file

    Returns:
        TradingBotFacade instance
    """
    return TradingBotFacade(config_path=config_path)


def main() -> None:  # pragma: no cover (CLI entry point, tested via integration)
    """
    Main entry point.

    WARNING: This will start the live trading bot!
    Use 'crypto-quant run-bot' command instead.
    """
    import sys

    # Safety check: require explicit flag to prevent accidental execution
    if "--force" not in sys.argv:
        print("ERROR: Direct execution of bot_facade.py is disabled for safety.")
        print("Use 'crypto-quant run-bot' command instead.")
        print(
            "If you really want to run this directly, use: python -m src.execution.bot_facade --force"
        )
        sys.exit(1)

    bot = create_bot()
    bot.run()


if __name__ == "__main__":
    main()
