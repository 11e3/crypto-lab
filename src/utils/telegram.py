"""
Telegram notification utilities.

Provides functions to send messages via Telegram bot.
"""

import os

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramNotifier:
    """Handles Telegram notifications."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize Telegram notifier.

        Args:
            token: Telegram bot token (defaults to TELEGRAM_TOKEN env var)
            chat_id: Telegram chat ID (defaults to TELEGRAM_CHAT_ID env var)
            enabled: Whether notifications are enabled
        """
        self.token = token or os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = enabled and bool(self.token) and bool(self.chat_id)

        if not self.enabled:
            logger.warning("Telegram notifications disabled (missing token or chat_id)")

    def send(self, message: str) -> bool:
        """
        Send a message to Telegram.

        Args:
            message: Message text to send

        Returns:
            True if message was sent successfully, False otherwise
        """
        # Safety check: prevent sending during tests (only if Telegram is real and enabled)
        # Allow mocks to work in tests
        import sys
        from unittest.mock import MagicMock

        is_testing = (
            "pytest" in sys.modules
            or "unittest" in sys.modules
            or "PYTEST_CURRENT_TEST" in os.environ
            or any("test" in arg.lower() for arg in sys.argv)
        )

        # Only block if:
        # 1. We're in a test environment
        # 2. Telegram is enabled
        # 3. Telegram has a real token (not placeholder, not test token)
        # 4. Telegram is NOT a mock (tests use mocks which should work)
        # 5. Token looks like a real production token (not "real_token", "test_token", etc.)
        is_test_token = (
            self.token
            and (
                self.token in ("real_token", "test_token", "test_chat_id")
                or self.token.startswith("test_")
                or len(self.token) < 20  # Real tokens are much longer
            )
        )
        
        if (
            is_testing
            and self.enabled
            and self.token
            and "YOUR_" not in self.token
            and not isinstance(self, MagicMock)
            and not is_test_token
        ):
            # Real Telegram in test environment - block it
            logger.debug(f"Telegram.send() blocked during testing: {message[:50]}...")
            return False

        if not self.enabled:
            return False

        if not self.token or "YOUR_" in self.token:
            logger.debug("Telegram token not configured")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {"chat_id": self.chat_id, "text": message}
            response = requests.post(url, data=data, timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}", exc_info=True)
            return False

    def send_trade_signal(
        self,
        action: str,
        ticker: str,
        price: float,
        **kwargs,
    ) -> bool:
        """
        Send formatted trade signal message.

        Args:
            action: Trade action (BUY, SELL, HOLD, EXIT)
            ticker: Trading pair ticker
            price: Current price
            **kwargs: Additional information (target, noise, etc.)

        Returns:
            True if message was sent successfully
        """
        emoji_map = {
            "BUY": "🔥",
            "SELL": "💰",
            "HOLD": "✊",
            "EXIT": "📉",
        }

        emoji = emoji_map.get(action, "📊")
        message = f"{emoji} [{action}] {ticker}\nPrice: {price:,.0f}"

        for key, value in kwargs.items():
            if isinstance(value, float):
                message += f"\n{key.capitalize()}: {value:,.2f}"
            else:
                message += f"\n{key.capitalize()}: {value}"

        return self.send(message)


# Global notifier instance
_notifier: TelegramNotifier | None = None


def get_notifier(
    token: str | None = None,
    chat_id: str | None = None,
    enabled: bool = True,
) -> TelegramNotifier:
    """
    Get or create global Telegram notifier instance.

    Args:
        token: Telegram bot token
        chat_id: Telegram chat ID
        enabled: Whether notifications are enabled

    Returns:
        TelegramNotifier instance
    """
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier(token=token, chat_id=chat_id, enabled=enabled)
    else:
        # Update existing notifier if new values provided
        if token is not None:
            _notifier.token = token
        if chat_id is not None:
            _notifier.chat_id = chat_id
        if enabled is not None:
            # Recalculate enabled based on new values
            _notifier.enabled = enabled and bool(_notifier.token) and bool(_notifier.chat_id)
    return _notifier


def send_message(message: str) -> bool:
    """
    Convenience function to send a Telegram message.

    Args:
        message: Message text

    Returns:
        True if sent successfully
    """
    return get_notifier().send(message)
