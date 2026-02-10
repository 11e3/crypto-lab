"""TradingBot Facade - Simplified interface for trading operations."""

from pathlib import Path

from src.exchange import Exchange, ExchangeError
from src.execution.bot.bot_init import (
    check_existing_holdings,
    create_bot_components,
    initialize_targets,
)
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.signal_handler import SignalHandler
from src.strategies.volatility_breakout import VanillaVBO
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class TradingBotFacade:
    """Facade for trading bot operations."""

    def __init__(
        self,
        exchange: Exchange | None = None,
        position_manager: PositionManager | None = None,
        order_manager: OrderManager | None = None,
        signal_handler: SignalHandler | None = None,
        strategy: VanillaVBO | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Initialize trading bot facade with dependency injection."""
        components, trading_config, strategy_config, bot_config, tickers = create_bot_components(
            config_path=config_path,
            exchange=exchange,
            position_manager=position_manager,
            order_manager=order_manager,
            signal_handler=signal_handler,
            strategy=strategy,
        )

        self.trading_config = trading_config
        self.strategy_config = strategy_config
        self.bot_config = bot_config
        self.tickers = tickers
        self.exchange = components.exchange
        self.position_manager = components.position_manager
        self.order_manager = components.order_manager
        self.signal_handler = components.signal_handler
        self.strategy = components.strategy
        self.advanced_order_manager = components.advanced_order_manager
        self.telegram = components.telegram
        self.trade_handler = components.trade_handler
        self.notification_handler = components.notification_handler
        self.event_bus = components.event_bus
        self.target_info: dict[str, dict[str, float]] = {}

    def get_krw_balance(self) -> float:
        """Get KRW balance."""
        try:
            balance = self.exchange.get_balance("KRW")
            return balance.available
        except (ExchangeError, ConnectionError, OSError) as e:
            logger.error(f"Error getting KRW balance: {e}", exc_info=True)
            return 0.0

    def initialize_targets(self) -> None:
        """Initialize target prices and metrics for all tickers."""
        self.target_info = initialize_targets(
            tickers=self.tickers,
            signal_handler=self.signal_handler,
            strategy_config=self.strategy_config,
            bot_config=self.bot_config,
        )

    def check_existing_holdings(self) -> None:
        """Check and recover existing holdings on bot restart."""
        check_existing_holdings(
            tickers=self.tickers,
            exchange=self.exchange,
            position_manager=self.position_manager,
            trading_config=self.trading_config,
        )

    def daily_reset(self) -> None:
        """Perform daily reset: check exits and recalculate targets."""
        from src.execution.bot.bot_reset import process_exits, recalculate_targets

        logger.info("Performing daily reset...")
        process_exits(self)
        self.target_info = recalculate_targets(self)

    def _calculate_buy_amount(self) -> float:
        """Calculate buy amount based on available cash and slots."""
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
        """Sell all holdings for a ticker."""
        from src.execution.trade_executor import sell_all

        min_amount = self.trading_config["min_order_amount"]
        return sell_all(
            ticker=ticker,
            order_manager=self.order_manager,
            position_manager=self.position_manager,
            exchange=self.exchange,
            telegram=self.telegram,
            min_amount=min_amount,
        )

    def _execute_buy_order(self, ticker: str, current_price: float, buy_amount: float) -> bool:
        """Execute buy market order."""
        from src.execution.trade_executor import execute_buy_order

        min_amount = self.trading_config["min_order_amount"]
        return execute_buy_order(
            ticker=ticker,
            current_price=current_price,
            buy_amount=buy_amount,
            order_manager=self.order_manager,
            position_manager=self.position_manager,
            advanced_order_manager=self.advanced_order_manager,
            telegram=self.telegram,
            trading_config=self.trading_config,
            target_info=self.target_info,
            min_amount=min_amount,
        )

    def process_ticker_update(self, ticker: str, current_price: float) -> None:
        """Process real-time ticker update and check for entry signals."""
        from src.execution.trade_executor import process_ticker_update

        process_ticker_update(
            ticker=ticker,
            current_price=current_price,
            position_manager=self.position_manager,
            order_manager=self.order_manager,
            advanced_order_manager=self.advanced_order_manager,
            signal_handler=self.signal_handler,
            trading_config=self.trading_config,
            target_info=self.target_info,
            telegram=self.telegram,
            calculate_buy_amount_fn=self._calculate_buy_amount,
            execute_buy_fn=self._execute_buy_order,
        )

    def run(self) -> None:
        """Run the trading bot main loop."""
        from src.execution.bot.bot_run import run_trading_loop

        run_trading_loop(self)


def create_bot(config_path: Path | None = None) -> TradingBotFacade:
    """Factory function to create a TradingBotFacade with default dependencies."""
    return TradingBotFacade(config_path=config_path)


def main() -> None:  # pragma: no cover
    """Main entry point. Use 'crypto-quant run-bot' command instead."""
    import sys

    if "--force" not in sys.argv:
        print("ERROR: Direct execution of bot_facade.py is disabled for safety.")
        print("Use 'crypto-quant run-bot' command instead.")
        sys.exit(1)

    bot = create_bot()
    bot.run()


if __name__ == "__main__":
    main()
