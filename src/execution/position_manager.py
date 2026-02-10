"""
Position manager for tracking and managing trading positions.

Focuses on position lifecycle management (SRP).
PnL calculations are delegated to PnLCalculator.
"""

from src.exchange import ExchangeError, PriceService
from src.execution.event_bus import EventBus, get_event_bus
from src.execution.events import EventType, PositionEvent
from src.execution.pnl_calculator import PnLCalculator
from src.execution.position import Position
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Re-export Position for backward compatibility
__all__ = ["Position", "PositionManager", "PnLCalculator"]


class PositionManager:
    """
    Manages trading positions across multiple tickers.

    Focuses on position lifecycle (add, remove, query).
    Delegates PnL calculations to PnLCalculator.
    """

    def __init__(
        self,
        exchange: PriceService,
        publish_events: bool = True,
        event_bus: EventBus | None = None,
        pnl_calculator: PnLCalculator | None = None,
    ) -> None:
        """
        Initialize position manager.

        Args:
            exchange: Service implementing PriceService protocol
            publish_events: Whether to publish events (default: True)
            event_bus: Optional EventBus instance (uses global if not provided)
            pnl_calculator: Optional PnLCalculator (creates default if not provided)
        """
        self.exchange = exchange
        self.positions: dict[str, Position] = {}
        self.publish_events = publish_events
        self.event_bus = event_bus if event_bus else (get_event_bus() if publish_events else None)
        self.pnl_calculator = pnl_calculator or PnLCalculator()

    def has_position(self, ticker: str) -> bool:
        """Check if a position exists for a ticker."""
        return ticker in self.positions

    def get_position(self, ticker: str) -> Position | None:
        """Get position for a ticker."""
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
        """Get current market price for a ticker."""
        try:
            return self.exchange.get_current_price(ticker)
        except (ExchangeError, ConnectionError, OSError) as e:
            logger.error(f"Error getting price for {ticker}: {e}", exc_info=True)
            return 0.0

    def calculate_pnl(self, ticker: str, current_price: float | None = None) -> float:
        """Calculate unrealized PnL for a position. Delegates to PnLCalculator."""
        position = self.get_position(ticker)
        if not position:
            return 0.0

        if current_price is None:
            current_price = self.get_current_price(ticker)

        return self.pnl_calculator.calculate_pnl(position, current_price)

    def calculate_pnl_pct(self, ticker: str, current_price: float | None = None) -> float:
        """Calculate unrealized PnL percentage. Delegates to PnLCalculator."""
        position = self.get_position(ticker)
        if not position:
            return 0.0

        if current_price is None:
            current_price = self.get_current_price(ticker)

        return self.pnl_calculator.calculate_pnl_pct(position, current_price)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all open positions."""
        return self.positions.copy()

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    def clear_all(self) -> None:
        """Clear all positions."""
        count = len(self.positions)
        self.positions.clear()
        logger.info(f"Cleared all positions ({count} positions)")
