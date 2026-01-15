"""
Position manager for tracking and managing trading positions.
"""

from typing import TYPE_CHECKING

from src.execution.event_bus import EventBus, get_event_bus
from src.execution.events import EventType, PositionEvent
from src.execution.position import Position
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.exchange import Exchange

logger = get_logger(__name__)

# Re-export Position for backward compatibility
__all__ = ["Position", "PositionManager"]


class PositionManager:
    """
    Manages trading positions across multiple tickers.

    Tracks open positions, calculates PnL, and manages position lifecycle.
    """

    def __init__(
        self,
        exchange: "Exchange",
        publish_events: bool = True,
        event_bus: EventBus | None = None,
    ) -> None:
        """
        Initialize position manager.

        Args:
            exchange: Exchange instance for price queries
            publish_events: Whether to publish events (default: True)
            event_bus: Optional EventBus instance (uses global if not provided)
        """
        self.exchange = exchange
        self.positions: dict[str, Position] = {}
        self.publish_events = publish_events
        self.event_bus = event_bus if event_bus else (get_event_bus() if publish_events else None)

    def has_position(self, ticker: str) -> bool:
        """
        Check if a position exists for a ticker.

        Args:
            ticker: Trading pair symbol

        Returns:
            True if position exists
        """
        return ticker in self.positions

    def get_position(self, ticker: str) -> Position | None:
        """
        Get position for a ticker.

        Args:
            ticker: Trading pair symbol

        Returns:
            Position object or None
        """
        return self.positions.get(ticker)

    def add_position(
        self,
        ticker: str,
        entry_price: float,
        amount: float,
        entry_date: str | None = None,
    ) -> Position:
        """
        Add a new position.

        Args:
            ticker: Trading pair symbol
            entry_price: Entry price
            amount: Position size
            entry_date: Entry date (optional)

        Returns:
            Created Position object

        Raises:
            ValueError: If position already exists
        """
        if ticker in self.positions:
            raise ValueError(f"Position already exists for {ticker}")

        position = Position(ticker, entry_price, amount, entry_date)
        self.positions[ticker] = position
        logger.info(f"Added position: {position}")

        # Publish event
        if self.event_bus:
            current_price = self.get_current_price(ticker)
            pnl = self.calculate_pnl(ticker, current_price)
            pnl_pct = self.calculate_pnl_pct(ticker, current_price)

            event = PositionEvent(
                event_type=EventType.POSITION_OPENED,
                source="PositionManager",
                ticker=ticker,
                action="opened",
                entry_price=entry_price,
                amount=amount,
                current_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )
            self.event_bus.publish(event)

        return position

    def remove_position(self, ticker: str) -> Position | None:
        """
        Remove a position.

        Args:
            ticker: Trading pair symbol

        Returns:
            Removed Position object or None
        """
        position = self.positions.pop(ticker, None)
        if position:
            logger.info(f"Removed position: {position}")

            # Publish event
            if self.event_bus:
                current_price = self.get_current_price(ticker)
                pnl = self.calculate_pnl(ticker, current_price)
                pnl_pct = self.calculate_pnl_pct(ticker, current_price)

                event = PositionEvent(
                    event_type=EventType.POSITION_CLOSED,
                    source="PositionManager",
                    ticker=ticker,
                    action="closed",
                    entry_price=position.entry_price,
                    amount=position.amount,
                    current_price=current_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
                self.event_bus.publish(event)

        return position

    def get_current_price(self, ticker: str) -> float:
        """
        Get current market price for a ticker.

        Args:
            ticker: Trading pair symbol

        Returns:
            Current price, 0.0 on error
        """
        try:
            return self.exchange.get_current_price(ticker)
        except Exception as e:
            logger.error(f"Error getting price for {ticker}: {e}", exc_info=True)
            return 0.0

    def calculate_pnl(self, ticker: str, current_price: float | None = None) -> float:
        """
        Calculate unrealized PnL for a position.

        Args:
            ticker: Trading pair symbol
            current_price: Current market price (fetched if None)

        Returns:
            Unrealized PnL in quote currency
        """
        position = self.get_position(ticker)
        if not position:
            return 0.0

        if current_price is None:
            current_price = self.get_current_price(ticker)

        if current_price <= 0:
            return 0.0

        pnl = (current_price - position.entry_price) * position.amount
        return pnl

    def calculate_pnl_pct(self, ticker: str, current_price: float | None = None) -> float:
        """
        Calculate unrealized PnL percentage.

        Args:
            ticker: Trading pair symbol
            current_price: Current market price (fetched if None)

        Returns:
            Unrealized PnL percentage
        """
        position = self.get_position(ticker)
        if not position:
            return 0.0

        if current_price is None:
            current_price = self.get_current_price(ticker)

        if current_price <= 0 or position.entry_price <= 0:
            return 0.0

        return ((current_price / position.entry_price) - 1.0) * 100.0

    def get_all_positions(self) -> dict[str, Position]:
        """
        Get all open positions.

        Returns:
            Dictionary of ticker -> Position
        """
        return self.positions.copy()

    def get_position_count(self) -> int:
        """
        Get number of open positions.

        Returns:
            Number of positions
        """
        return len(self.positions)

    def clear_all(self) -> None:
        """Clear all positions."""
        count = len(self.positions)
        self.positions.clear()
        logger.info(f"Cleared all positions ({count} positions)")
