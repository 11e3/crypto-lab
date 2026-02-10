from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from src.exchange.types import OrderType
from src.execution.bot.bot_facade import TradingBotFacade
from src.execution.order_manager import Order


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock configuration dictionary."""
    config_mock = MagicMock()

    # Trading config
    config_mock.get_trading_config.return_value = {
        "tickers": ["KRW-BTC", "KRW-ETH"],
        "min_order_amount": 5000.0,
        "max_slots": 3,
        "fee_rate": 0.0005,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 5.0,
        "trailing_stop_pct": 1.5,
    }

    # Strategy config
    config_mock.get_strategy_config.return_value = {
        "name": "VanillaVBO",
        "sma_period": 5,
        "trend_sma_period": 20,
        "short_noise_period": 20,
        "long_noise_period": 10,
        "exclude_current": True,
    }

    # Bot config
    config_mock.get_bot_config.return_value = {
        "api_retry_delay": 0.1,  # Speed up tests
        "websocket_reconnect_delay": 0.1,
        "daily_reset_hour": 9,
        "daily_reset_minute": 0,
    }

    # Telegram config
    config_mock.get_telegram_config.return_value = {
        "token": "fake_token",
        "chat_id": "fake_chat_id",
        "enabled": True,
    }

    config_mock.get_exchange_name.return_value = "upbit"

    return config_mock


@pytest.fixture
def mock_components() -> dict[str, MagicMock]:
    """Return a dictionary of mocked components."""
    return {
        "exchange": MagicMock(),
        "position_manager": MagicMock(),
        "order_manager": MagicMock(),
        "signal_handler": MagicMock(),
        "strategy": MagicMock(),
        "telegram": MagicMock(),
        "advanced_order_manager": MagicMock(),
    }


@pytest.fixture
def bot(mock_config: MagicMock, mock_components: dict[str, MagicMock]) -> TradingBotFacade:
    """Create a TradingBotFacade instance with mocked dependencies."""
    with (
        patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
        patch("src.execution.bot.bot_factory.get_notifier", return_value=mock_components["telegram"]),
        patch("src.execution.bot.bot_factory.get_event_bus"),
        patch("src.execution.bot.bot_factory.TradeHandler"),
        patch("src.execution.bot.bot_factory.NotificationHandler"),
        patch(
            "src.execution.bot.bot_factory.AdvancedOrderManager",
            return_value=mock_components["advanced_order_manager"],
        ),
    ):
        bot_instance = TradingBotFacade(
            exchange=mock_components["exchange"],
            position_manager=mock_components["position_manager"],
            order_manager=mock_components["order_manager"],
            signal_handler=mock_components["signal_handler"],
            strategy=mock_components["strategy"],
        )
        return bot_instance


class TestTradingBotFacade:
    def test_initialization(self, bot: TradingBotFacade, mock_config: MagicMock) -> None:
        """Test if the bot initializes correctly with configs."""
        assert bot.tickers == ["KRW-BTC", "KRW-ETH"]
        assert bot.exchange is not None
        assert bot.position_manager is not None
        mock_config.get_trading_config.assert_called()

    def test_get_krw_balance(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test KRW balance retrieval."""
        # Success case
        balance_mock = MagicMock()
        balance_mock.available = 100000.0
        mock_components["exchange"].get_balance.return_value = balance_mock

        assert bot.get_krw_balance() == 100000.0
        mock_components["exchange"].get_balance.assert_called_with("KRW")

        # Exception case
        mock_components["exchange"].get_balance.side_effect = ConnectionError("API Error")
        assert bot.get_krw_balance() == 0.0

    def test_initialize_targets(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test target initialization for tickers."""
        # Setup metrics return
        mock_metrics = {
            "target": 50000000.0,
            "k": 0.5,
            "long_noise": 0.6,
            "sma": 48000000.0,
            "sma_trend": 49000000.0,
        }
        mock_components["signal_handler"].calculate_metrics.return_value = mock_metrics

        bot.initialize_targets()

        assert "KRW-BTC" in bot.target_info
        assert bot.target_info["KRW-BTC"] == mock_metrics
        assert mock_components["signal_handler"].calculate_metrics.call_count == len(bot.tickers)

    def test_check_existing_holdings_recovery(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test recovery of existing positions on startup."""
        # Scenario: User holds KRW-BTC
        balance_btc = MagicMock()
        balance_btc.available = 0.1

        # Configure exchange mock
        def get_balance_side_effect(currency: str) -> MagicMock:
            if currency == "BTC":
                return balance_btc
            return MagicMock(available=0)

        mock_components["exchange"].get_balance.side_effect = get_balance_side_effect
        mock_components["exchange"].get_current_price.return_value = 50000000.0

        bot.check_existing_holdings()

        # Assert position was added
        mock_components["position_manager"].add_position.assert_called()
        call_args = mock_components["position_manager"].add_position.call_args[1]
        assert call_args["ticker"] == "KRW-BTC"
        assert call_args["amount"] == 0.1
        assert call_args["entry_price"] == 50000000.0

    def test_calculate_buy_amount(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test position sizing calculation."""
        # Setup: 100k KRW balance, 3 max slots, 1 currently used
        balance_krw = MagicMock()
        balance_krw.available = 100000.0
        mock_components["exchange"].get_balance.return_value = balance_krw
        mock_components["position_manager"].get_position_count.return_value = 1

        # Expected: (100000 / (3-1)) * (1 - 0.0005) = 50000 * 0.9995 = 49975
        expected_amount = 50000.0 * (1 - 0.0005)

        amount = bot._calculate_buy_amount()
        assert amount == expected_amount

        # Test insufficient funds
        balance_krw.available = 1000.0  # Below min_order_amount (5000)
        assert bot._calculate_buy_amount() == 0.0

        # Test no slots
        mock_components["position_manager"].get_position_count.return_value = 3
        balance_krw.available = 100000.0
        assert bot._calculate_buy_amount() == 0.0

    def test_process_ticker_update_entry(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test entry signal processing."""
        ticker = "KRW-BTC"
        current_price = 50000000.0

        # Setup: Not holding, Signal is True
        mock_components["position_manager"].has_position.return_value = False
        mock_components["signal_handler"].check_entry_signal.return_value = True

        # Setup target info
        bot.target_info = {ticker: {"target": 49000000.0, "k": 0.5}}

        # Setup buy amount
        with patch.object(bot, "_calculate_buy_amount", return_value=100000.0):
            # Setup successful order
            mock_order = MagicMock(spec=Order)
            mock_order.order_id = "uuid-123"
            mock_components["order_manager"].place_buy_order.return_value = mock_order

            bot.process_ticker_update(ticker, current_price)

            # Assertions
            mock_components["order_manager"].place_buy_order.assert_called_with(
                ticker, 100000.0, ANY
            )
            mock_components["position_manager"].add_position.assert_called_with(
                ticker=ticker, entry_price=current_price, amount=ANY
            )
            # Advanced orders created
            mock_components["advanced_order_manager"].create_stop_loss.assert_called()
            mock_components["advanced_order_manager"].create_trailing_stop.assert_called()
            # Telegram notification
            mock_components["telegram"].send_trade_signal.assert_called_with(
                "BUY", ticker, current_price, target=ANY, noise=ANY
            )

    def test_process_ticker_update_advanced_order_trigger(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test if advanced orders (e.g., Stop Loss) trigger a sell."""
        ticker = "KRW-BTC"
        current_price = 45000000.0  # Price dropped

        # Setup: Holding position
        mock_components["position_manager"].has_position.return_value = True
        mock_position = MagicMock()
        mock_position.amount = 0.1
        mock_components["position_manager"].get_position.return_value = mock_position

        # Setup OHLCV data for check
        df = pd.DataFrame({"low": [44000000.0], "high": [46000000.0], "close": [45000000.0]})
        mock_components["signal_handler"].get_ohlcv_data.return_value = df

        # Setup: Advanced order triggers
        triggered_order = MagicMock()
        # [수정] STOP_LOSS -> MARKET으로 변경 (실제 체결 주문 타입)
        triggered_order.order_type = OrderType.MARKET
        triggered_order.triggered_price = 45000000.0
        mock_components["advanced_order_manager"].check_orders.return_value = [triggered_order]

        # Setup successful sell order
        mock_components["order_manager"].place_sell_order.return_value = MagicMock(
            order_id="sell-123"
        )

        bot.process_ticker_update(ticker, current_price)

        # Assertions
        mock_components["advanced_order_manager"].check_orders.assert_called()
        mock_components["order_manager"].place_sell_order.assert_called()
        mock_components["position_manager"].remove_position.assert_called_with(ticker)
        mock_components["advanced_order_manager"].cancel_all_orders.assert_called_with(
            ticker=ticker
        )

    def test_process_exits_signal(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test exit signal processing using bot_reset module."""
        from src.execution.bot.bot_reset import process_exits

        ticker = "KRW-BTC"

        # Setup: Holding 1 position
        mock_components["position_manager"].get_all_positions.return_value = [ticker]

        # Setup: Exit signal is True
        mock_components["signal_handler"].check_exit_signal.return_value = True

        # Mock _sell_all to succeed
        with patch.object(bot, "_sell_all", return_value=True) as mock_sell_all:
            # Setup data for notification details
            df = pd.DataFrame(
                {
                    "close": [100.0] * 15  # Dummy data
                }
            )
            mock_components["signal_handler"].get_ohlcv_data.return_value = df
            mock_components["strategy"].calculate_indicators.return_value = df

            process_exits(bot)

            # Assertions
            mock_sell_all.assert_called_with(ticker)

    def test_daily_reset(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test daily reset routine using bot_reset module."""
        with (
            patch("src.execution.bot.bot_reset.process_exits") as mock_exits,
            patch("src.execution.bot.bot_reset.recalculate_targets") as mock_targets,
        ):
            mock_targets.return_value = {"KRW-BTC": {"target": 50000.0}}

            bot.daily_reset()

            mock_exits.assert_called_once()
            mock_targets.assert_called_once()

    def test_recalculate_targets(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test recalculating targets using bot_reset module."""
        from src.execution.bot.bot_reset import recalculate_targets

        bot.tickers = ["KRW-BTC"]
        mock_metrics = {"target": 100, "k": 0.5, "sma": 90, "sma_trend": 95}
        mock_components["signal_handler"].calculate_metrics.return_value = mock_metrics

        result = recalculate_targets(bot)

        assert result["KRW-BTC"] == mock_metrics
        mock_components["telegram"].send.assert_called()

    def test_sma_exit_calculation(self, bot: TradingBotFacade) -> None:
        """Test helper function for SMA exit calculation using bot_reset module."""
        from src.execution.bot.bot_reset import calculate_sma_exit

        # Case 1: Insufficient data
        assert calculate_sma_exit(pd.DataFrame()) is None

        # Case 2: Valid data
        # Need at least SMA_PERIOD (5) + 2 rows
        data = [100.0] * 10
        df = pd.DataFrame({"close": data})
        sma = calculate_sma_exit(df)
        assert sma == 100.0

    def test_sell_all_exception(
        self, bot: TradingBotFacade, mock_components: dict[str, MagicMock]
    ) -> None:
        """Test error handling in sell_all."""
        ticker = "KRW-BTC"
        mock_components["order_manager"].sell_all.side_effect = ConnectionError("Sell Failed")

        result = bot._sell_all(ticker)

        assert result is False
        mock_components["position_manager"].remove_position.assert_not_called()

    def test_create_bot_factory(self, mock_config: MagicMock) -> None:
        """Test the create_bot factory function."""
        with (
            patch("src.execution.bot.bot_factory.get_config", return_value=mock_config),
            patch("src.execution.bot.bot_factory.get_notifier"),
            patch("src.execution.bot.bot_factory.ExchangeFactory"),
        ):
            from src.execution.bot.bot_facade import create_bot

            bot = create_bot()
            assert isinstance(bot, TradingBotFacade)
