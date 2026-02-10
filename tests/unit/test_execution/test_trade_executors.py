"""Tests for trade executors edge cases.

Tests exchange error handling in buy/sell executors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.exchange import ExchangeError
from src.execution.trade_executors import BuyExecutor, SellExecutor


class TestBuyExecutorErrors:
    """Test BuyExecutor exception handling."""

    def _create_buy_executor(self) -> tuple[BuyExecutor, MagicMock, MagicMock]:
        """Create BuyExecutor with mock dependencies."""
        order_manager = MagicMock()
        position_manager = MagicMock()
        advanced_order_manager = MagicMock()
        telegram = MagicMock()
        executor = BuyExecutor(order_manager, position_manager, advanced_order_manager, telegram)
        return executor, order_manager, position_manager

    def test_exchange_error_returns_false(self) -> None:
        """ExchangeError during buy returns False."""
        executor, order_manager, _ = self._create_buy_executor()
        order_manager.place_buy_order.side_effect = ExchangeError("API timeout")

        result = executor.execute(
            ticker="KRW-BTC",
            current_price=50000.0,
            buy_amount=1000000.0,
            trading_config={},
            target_info={},
            min_amount=5000.0,
        )

        assert result is False

    def test_connection_error_returns_false(self) -> None:
        """ConnectionError during buy returns False."""
        executor, order_manager, _ = self._create_buy_executor()
        order_manager.place_buy_order.side_effect = ConnectionError("Network unreachable")

        result = executor.execute(
            ticker="KRW-BTC",
            current_price=50000.0,
            buy_amount=1000000.0,
            trading_config={},
            target_info={},
            min_amount=5000.0,
        )

        assert result is False

    def test_value_error_returns_false(self) -> None:
        """ValueError (e.g., min order size) during buy returns False."""
        executor, order_manager, _ = self._create_buy_executor()
        order_manager.place_buy_order.side_effect = ValueError("Minimum order size 5000 KRW")

        result = executor.execute(
            ticker="KRW-BTC",
            current_price=50000.0,
            buy_amount=1000.0,
            trading_config={},
            target_info={},
            min_amount=5000.0,
        )

        assert result is False

    def test_order_not_created_returns_false(self) -> None:
        """Returns False when order object is None."""
        executor, order_manager, _ = self._create_buy_executor()
        order_manager.place_buy_order.return_value = None

        result = executor.execute(
            ticker="KRW-BTC",
            current_price=50000.0,
            buy_amount=1000000.0,
            trading_config={},
            target_info={},
            min_amount=5000.0,
        )

        assert result is False

    def test_order_no_id_returns_false(self) -> None:
        """Returns False when order has no order_id."""
        executor, order_manager, _ = self._create_buy_executor()
        mock_order = MagicMock()
        mock_order.order_id = None
        order_manager.place_buy_order.return_value = mock_order

        result = executor.execute(
            ticker="KRW-BTC",
            current_price=50000.0,
            buy_amount=1000000.0,
            trading_config={},
            target_info={},
            min_amount=5000.0,
        )

        assert result is False


class TestSellExecutorErrors:
    """Test SellExecutor exception handling."""

    def _create_sell_executor(self) -> tuple[SellExecutor, MagicMock, MagicMock, MagicMock]:
        """Create SellExecutor with mock dependencies."""
        order_manager = MagicMock()
        position_manager = MagicMock()
        exchange = MagicMock()
        telegram = MagicMock()
        executor = SellExecutor(order_manager, position_manager, exchange, telegram)
        return executor, order_manager, position_manager, exchange

    def test_exchange_error_returns_false(self) -> None:
        """ExchangeError during sell returns False."""
        executor, order_manager, _, _ = self._create_sell_executor()
        order_manager.sell_all.side_effect = ExchangeError("API error")

        result = executor.execute(ticker="KRW-BTC", min_amount=0.0001)

        assert result is False

    def test_connection_error_returns_false(self) -> None:
        """ConnectionError during sell returns False."""
        executor, order_manager, _, _ = self._create_sell_executor()
        order_manager.sell_all.side_effect = ConnectionError("Timeout")

        result = executor.execute(ticker="KRW-BTC", min_amount=0.0001)

        assert result is False

    def test_successful_sell(self) -> None:
        """Successful sell removes position."""
        executor, order_manager, position_manager, _ = self._create_sell_executor()
        mock_order = MagicMock()
        mock_order.order_id = "order-123"
        order_manager.sell_all.return_value = mock_order

        result = executor.execute(ticker="KRW-BTC", min_amount=0.0001)

        assert result is True
        position_manager.remove_position.assert_called_once_with("KRW-BTC")

    def test_notification_failure_does_not_crash_sell(self) -> None:
        """Sell succeeds even if notification fails."""
        executor, order_manager, position_manager, exchange = self._create_sell_executor()
        mock_order = MagicMock()
        mock_order.order_id = "order-123"
        order_manager.sell_all.return_value = mock_order

        # Notification will fail, but sell should still succeed
        exchange.get_current_price.side_effect = ExchangeError("Price fetch failed")

        # Force _is_testing to return False so notification path is exercised
        with patch.object(SellExecutor, "_is_testing", return_value=False):
            result = executor.execute(ticker="KRW-BTC", min_amount=0.0001)

        assert result is True
        position_manager.remove_position.assert_called_once_with("KRW-BTC")
