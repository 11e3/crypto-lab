"""
PnL (Profit and Loss) calculation module.

Separates PnL calculation responsibility from position management (SRP).
"""

from src.execution.position import Position


class PnLCalculator:
    """
    Calculates Profit and Loss for trading positions.

    Focused on PnL calculations only, separate from position management.
    """

    @staticmethod
    def calculate_pnl(
        position: Position,
        current_price: float,
    ) -> float:
        """
        Calculate unrealized PnL for a position.

        Args:
            position: Position object
            current_price: Current market price

        Returns:
            Unrealized PnL in quote currency
        """
        if current_price <= 0:
            return 0.0

        return (current_price - position.entry_price) * position.amount

    @staticmethod
    def calculate_pnl_pct(
        position: Position,
        current_price: float,
    ) -> float:
        """
        Calculate unrealized PnL percentage.

        Args:
            position: Position object
            current_price: Current market price

        Returns:
            Unrealized PnL percentage
        """
        if current_price <= 0 or position.entry_price <= 0:
            return 0.0

        return ((current_price / position.entry_price) - 1.0) * 100.0

    @staticmethod
    def calculate_total_pnl(
        positions: dict[str, Position],
        prices: dict[str, float],
    ) -> float:
        """
        Calculate total unrealized PnL across all positions.

        Args:
            positions: Dictionary of ticker -> Position
            prices: Dictionary of ticker -> current price

        Returns:
            Total unrealized PnL
        """
        total = 0.0
        for ticker, position in positions.items():
            current_price = prices.get(ticker, 0.0)
            total += PnLCalculator.calculate_pnl(position, current_price)
        return total
