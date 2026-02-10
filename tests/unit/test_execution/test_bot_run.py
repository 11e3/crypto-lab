"""
Tests for bot run execution logic.
"""

import datetime
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.execution.bot.bot_run import (
    DAILY_RESET_WINDOW_SECONDS,
    _is_daily_reset_time,
    _should_block_in_test,
    _validate_api_connection,
    run_trading_loop,
)


@pytest.fixture
def mock_bot_facade() -> MagicMock:
    """Create mock TradingBotFacade for testing."""
    bot = MagicMock()
    bot.tickers = ["KRW-BTC", "KRW-ETH"]
    bot.telegram = MagicMock()
    bot.telegram.enabled = False
    bot.telegram.token = None
    bot.exchange = MagicMock()
    bot.bot_config = {
        "daily_reset_hour": 9,
        "daily_reset_minute": 0,
        "websocket_reconnect_delay": 5,
    }
    return bot


class TestShouldBlockInTest:
    """Tests for _should_block_in_test function."""

    def test_block_with_pytest_module(self, mock_bot_facade: MagicMock) -> None:
        """Test blocking when pytest is in modules with real telegram object."""

        # pytest is already in sys.modules during test
        # Create a real telegram-like object instead of MagicMock
        class FakeTelegram:
            def __init__(self) -> None:
                self.enabled = True
                self.token = "real_token_123"

        mock_bot_facade.telegram = FakeTelegram()

        result = _should_block_in_test(mock_bot_facade)
        assert result is True

    def test_no_block_with_allow_flag(self, mock_bot_facade: MagicMock) -> None:
        """Test not blocking when --allow-test-run flag is present."""
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["test", "--allow-test-run"]
            mock_bot_facade.telegram = MagicMock()
            mock_bot_facade.telegram.enabled = True
            mock_bot_facade.telegram.token = "real_token_123"

            result = _should_block_in_test(mock_bot_facade)
            assert result is False
        finally:
            sys.argv = original_argv

    def test_no_block_with_mock_telegram(self, mock_bot_facade: MagicMock) -> None:
        """Test not blocking when telegram is a MagicMock."""
        # MagicMock should not trigger blocking
        result = _should_block_in_test(mock_bot_facade)
        assert result is False

    def test_no_block_with_disabled_telegram(self, mock_bot_facade: MagicMock) -> None:
        """Test not blocking when telegram is disabled."""
        mock_bot_facade.telegram = MagicMock()
        mock_bot_facade.telegram.enabled = False

        result = _should_block_in_test(mock_bot_facade)
        assert result is False

    def test_no_block_with_no_token(self, mock_bot_facade: MagicMock) -> None:
        """Test not blocking when telegram has no token."""
        mock_bot_facade.telegram = MagicMock()
        mock_bot_facade.telegram.enabled = True
        mock_bot_facade.telegram.token = None

        result = _should_block_in_test(mock_bot_facade)
        assert result is False

    def test_no_block_with_placeholder_token(self, mock_bot_facade: MagicMock) -> None:
        """Test not blocking when telegram has placeholder token."""
        mock_bot_facade.telegram = MagicMock()
        mock_bot_facade.telegram.enabled = True
        mock_bot_facade.telegram.token = "YOUR_TELEGRAM_TOKEN"

        result = _should_block_in_test(mock_bot_facade)
        assert result is False


class TestValidateApiConnection:
    """Tests for _validate_api_connection function."""

    def test_validate_success(self, mock_bot_facade: MagicMock) -> None:
        """Test successful API validation."""
        mock_bot_facade.exchange.get_balance.return_value = {"KRW": 1000000}

        result = _validate_api_connection(mock_bot_facade)

        assert result is True
        mock_bot_facade.exchange.get_balance.assert_called_once_with("KRW")

    @patch("time.sleep")
    def test_validate_failure(self, mock_sleep: MagicMock, mock_bot_facade: MagicMock) -> None:
        """Test failed API validation."""
        mock_bot_facade.exchange.get_balance.side_effect = ConnectionError("API Error")

        result = _validate_api_connection(mock_bot_facade)

        assert result is False
        mock_sleep.assert_called_once_with(3)


class TestIsDailyResetTime:
    """Tests for _is_daily_reset_time function."""

    def test_exact_reset_time(self) -> None:
        """Test exact reset time detection."""
        now = datetime.datetime(2023, 1, 1, 9, 0, 5)  # 9:00:05
        result = _is_daily_reset_time(now, reset_hour=9, reset_minute=0)
        assert result is True

    def test_within_window(self) -> None:
        """Test time within reset window."""
        now = datetime.datetime(2023, 1, 1, 9, 0, DAILY_RESET_WINDOW_SECONDS - 1)
        result = _is_daily_reset_time(now, reset_hour=9, reset_minute=0)
        assert result is True

    def test_outside_window_seconds(self) -> None:
        """Test time outside reset window (seconds)."""
        now = datetime.datetime(2023, 1, 1, 9, 0, DAILY_RESET_WINDOW_SECONDS + 1)
        result = _is_daily_reset_time(now, reset_hour=9, reset_minute=0)
        assert result is False

    def test_wrong_hour(self) -> None:
        """Test wrong hour."""
        now = datetime.datetime(2023, 1, 1, 10, 0, 0)
        result = _is_daily_reset_time(now, reset_hour=9, reset_minute=0)
        assert result is False

    def test_wrong_minute(self) -> None:
        """Test wrong minute."""
        now = datetime.datetime(2023, 1, 1, 9, 1, 0)
        result = _is_daily_reset_time(now, reset_hour=9, reset_minute=0)
        assert result is False

    def test_custom_reset_time(self) -> None:
        """Test custom reset time."""
        now = datetime.datetime(2023, 1, 1, 15, 30, 5)
        result = _is_daily_reset_time(now, reset_hour=15, reset_minute=30)
        assert result is True


class TestRunTradingLoop:
    """Tests for run_trading_loop function."""

    def test_blocked_in_test_environment(self, mock_bot_facade: MagicMock) -> None:
        """Test that loop is blocked in test environment with real telegram."""

        # Create a real telegram-like object instead of MagicMock
        class FakeTelegram:
            def __init__(self) -> None:
                self.enabled = True
                self.token = "real_token_without_YOUR"

        mock_bot_facade.telegram = FakeTelegram()

        # Should return early without error
        run_trading_loop(mock_bot_facade)

        # initialize_targets should NOT be called if blocked
        mock_bot_facade.initialize_targets.assert_not_called()

    def test_api_validation_failure(self, mock_bot_facade: MagicMock) -> None:
        """Test early return when API validation fails."""
        mock_bot_facade.exchange.get_balance.side_effect = ConnectionError("API Error")

        with patch("time.sleep"):
            run_trading_loop(mock_bot_facade)

        # Should not proceed to initialize_targets
        mock_bot_facade.initialize_targets.assert_not_called()

    @patch("src.execution.bot.bot_run._create_websocket")
    def test_websocket_creation_failure(
        self, mock_ws: MagicMock, mock_bot_facade: MagicMock
    ) -> None:
        """Test early return when websocket creation fails."""
        mock_ws.return_value = None
        mock_bot_facade.exchange.get_balance.return_value = {"KRW": 1000000}

        run_trading_loop(mock_bot_facade)

        mock_bot_facade.initialize_targets.assert_called_once()
        mock_bot_facade.check_existing_holdings.assert_called_once()

    @patch("src.execution.bot.bot_run._create_websocket")
    @patch("src.execution.bot.bot_run._main_loop")
    def test_successful_startup(
        self,
        mock_main_loop: MagicMock,
        mock_ws: MagicMock,
        mock_bot_facade: MagicMock,
    ) -> None:
        """Test successful bot startup sequence."""
        mock_ws.return_value = MagicMock()
        mock_bot_facade.exchange.get_balance.return_value = {"KRW": 1000000}

        run_trading_loop(mock_bot_facade)

        mock_bot_facade.telegram.send.assert_called()
        mock_bot_facade.initialize_targets.assert_called_once()
        mock_bot_facade.check_existing_holdings.assert_called_once()
        mock_main_loop.assert_called_once()


