"""
Unit tests for config settings module.
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.settings import Settings, get_settings


class TestSettings:
    """Test cases for Settings class."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_settings(self) -> None:
        """Test Settings with default values.

        Note: Pydantic Settings may load from .env file even with cleared environment.
        This test verifies that Settings can be instantiated with defaults.
        """
        # Clear any cached settings
        import src.config.settings

        src.config.settings._settings = None

        settings = Settings()

        # Check that Settings instance is created (values may come from .env if exists)
        assert isinstance(settings, Settings)
        assert settings.telegram_enabled is True
        assert settings.trading_fee_rate == 0.0005
        assert settings.trading_max_slots == 4
        assert settings.strategy_name == "VBOV1"

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_positive_float_negative_value(self) -> None:
        """Test validate_positive_float raises ValueError for negative values (covers line 78)."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(trading_fee_rate=-0.1)

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("trading_fee_rate",) and "positive" in str(error["msg"]).lower()
            for error in errors
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_positive_float_zero_value(self) -> None:
        """Test validate_positive_float accepts zero value."""
        settings = Settings(trading_fee_rate=0.0)
        assert settings.trading_fee_rate == 0.0

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_positive_int_zero_value(self) -> None:
        """Test validate_positive_int raises ValueError for zero or negative values (covers line 86)."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(trading_max_slots=0)

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("trading_max_slots",) and "positive" in str(error["msg"]).lower()
            for error in errors
        )

        with pytest.raises(ValidationError) as exc_info2:
            Settings(trading_max_slots=-1)

        errors2 = exc_info2.value.errors()
        assert any(
            error["loc"] == ("trading_max_slots",) and "positive" in str(error["msg"]).lower()
            for error in errors2
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_hour_invalid_range(self) -> None:
        """Test validate_hour raises ValueError for invalid hour range (covers line 94)."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(bot_daily_reset_hour=24)

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("bot_daily_reset_hour",) and "between 0 and 23" in str(error["msg"])
            for error in errors
        )

        with pytest.raises(ValidationError) as exc_info2:
            Settings(bot_daily_reset_hour=-1)

        errors2 = exc_info2.value.errors()
        assert any(
            error["loc"] == ("bot_daily_reset_hour",) and "between 0 and 23" in str(error["msg"])
            for error in errors2
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_minute_invalid_range(self) -> None:
        """Test validate_minute raises ValueError for invalid minute range (covers line 102)."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(bot_daily_reset_minute=60)

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("bot_daily_reset_minute",) and "between 0 and 59" in str(error["msg"])
            for error in errors
        )

        with pytest.raises(ValidationError) as exc_info2:
            Settings(bot_daily_reset_minute=-1)

        errors2 = exc_info2.value.errors()
        assert any(
            error["loc"] == ("bot_daily_reset_minute",) and "between 0 and 59" in str(error["msg"])
            for error in errors2
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_get_upbit_keys_missing_keys(self) -> None:
        """Test get_upbit_keys raises ValueError when keys are missing (covers lines 124-130)."""
        settings = Settings(upbit_access_key="", upbit_secret_key="")

        with pytest.raises(ValueError, match="Upbit API keys not configured"):
            settings.get_upbit_keys()

    @patch.dict(os.environ, {}, clear=True)
    def test_get_upbit_keys_partial_keys(self) -> None:
        """Test get_upbit_keys raises ValueError when only one key is provided."""
        settings = Settings(upbit_access_key="key", upbit_secret_key="")

        with pytest.raises(ValueError, match="Upbit API keys not configured"):
            settings.get_upbit_keys()

        settings2 = Settings(upbit_access_key="", upbit_secret_key="secret")

        with pytest.raises(ValueError, match="Upbit API keys not configured"):
            settings2.get_upbit_keys()

    @patch.dict(os.environ, {}, clear=True)
    def test_get_upbit_keys_valid(self) -> None:
        """Test get_upbit_keys returns keys when both are provided."""
        settings = Settings(upbit_access_key="test_key", upbit_secret_key="test_secret")

        access_key, secret_key = settings.get_upbit_keys()

        assert access_key == "test_key"
        assert secret_key == "test_secret"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_trading_tickers_list(self) -> None:
        """Test get_trading_tickers_list converts comma-separated string to list."""
        settings = Settings(trading_tickers="KRW-BTC,KRW-ETH,KRW-XRP")

        tickers = settings.get_trading_tickers_list()

        assert tickers == ["KRW-BTC", "KRW-ETH", "KRW-XRP"]

    @patch.dict(os.environ, {}, clear=True)
    def test_get_trading_tickers_list_with_spaces(self) -> None:
        """Test get_trading_tickers_list handles spaces correctly."""
        settings = Settings(trading_tickers="KRW-BTC, KRW-ETH , KRW-XRP")

        tickers = settings.get_trading_tickers_list()

        assert tickers == ["KRW-BTC", "KRW-ETH", "KRW-XRP"]

    @patch.dict(os.environ, {}, clear=True)
    def test_get_telegram_config(self) -> None:
        """Test get_telegram_config returns correct dictionary."""
        settings = Settings(
            telegram_token="test_token", telegram_chat_id="test_chat", telegram_enabled=False
        )

        config = settings.get_telegram_config()

        assert config["token"] == "test_token"
        assert config["chat_id"] == "test_chat"
        assert config["enabled"] is False

    @patch.dict(os.environ, {}, clear=True)
    def test_get_trading_config(self) -> None:
        """Test get_trading_config returns correct dictionary."""
        settings = Settings(
            trading_tickers="KRW-BTC,KRW-ETH",
            trading_fee_rate=0.001,
            trading_max_slots=2,
            trading_min_order_amount=10000.0,
        )

        config = settings.get_trading_config()

        assert config["tickers"] == ["KRW-BTC", "KRW-ETH"]
        assert config["fee_rate"] == 0.001
        assert config["max_slots"] == 2
        assert config["min_order_amount"] == 10000.0

    @patch.dict(os.environ, {}, clear=True)
    def test_get_strategy_config(self) -> None:
        """Test get_strategy_config returns correct dictionary."""
        settings = Settings(
            strategy_name="CustomVBO",
            strategy_sma_period=10,
            strategy_trend_sma_period=20,
            strategy_short_noise_period=5,
            strategy_long_noise_period=10,
            strategy_exclude_current=False,
        )

        config = settings.get_strategy_config()

        assert config["name"] == "CustomVBO"
        assert config["sma_period"] == 10
        assert config["trend_sma_period"] == 20
        assert config["short_noise_period"] == 5
        assert config["long_noise_period"] == 10
        assert config["exclude_current"] is False

    @patch.dict(os.environ, {}, clear=True)
    def test_get_bot_config(self) -> None:
        """Test get_bot_config returns correct dictionary."""
        settings = Settings(
            bot_daily_reset_hour=10,
            bot_daily_reset_minute=30,
            bot_websocket_reconnect_delay=5.0,
            bot_api_retry_delay=1.0,
        )

        config = settings.get_bot_config()

        assert config["daily_reset_hour"] == 10
        assert config["daily_reset_minute"] == 30
        assert config["websocket_reconnect_delay"] == 5.0
        assert config["api_retry_delay"] == 1.0


class TestGetSettings:
    """Test cases for get_settings function."""

    def test_get_settings_singleton(self) -> None:
        """Test get_settings returns singleton instance."""

        # Reset global settings
        import src.config.settings

        src.config.settings._settings = None

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
        assert isinstance(settings1, Settings)
