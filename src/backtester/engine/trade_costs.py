"""
Trade cost calculation module.

Centralizes all trade cost calculations (fees, slippage, PnL) following OCP.
"""

from dataclasses import dataclass


@dataclass
class TradeCosts:
    """Container for trade cost calculations."""

    revenue: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    net_amount: float  # Amount after fees


class TradeCostCalculator:
    """
    Calculator for trade costs and metrics.

    Centralizes cost calculations to eliminate duplication across engines.
    """

    def __init__(self, fee_rate: float = 0.0, slippage_rate: float = 0.0) -> None:
        """
        Initialize calculator.

        Args:
            fee_rate: Transaction fee rate (e.g., 0.0005 = 0.05%)
            slippage_rate: Slippage rate (e.g., 0.001 = 0.1%)
        """
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def calculate_buy_amount(self, invest_amount: float, buy_price: float) -> float:
        """
        Calculate amount of asset received after buying.

        Args:
            invest_amount: Amount to invest (in quote currency)
            buy_price: Buy price per unit

        Returns:
            Amount of asset received after fees
        """
        return (invest_amount / buy_price) * (1 - self.fee_rate)

    def calculate_exit_costs(
        self,
        entry_price: float,
        exit_price: float,
        amount: float,
    ) -> TradeCosts:
        """
        Calculate all costs for exiting a position.

        Args:
            entry_price: Entry price per unit
            exit_price: Exit price per unit
            amount: Position size

        Returns:
            TradeCosts with all calculated values
        """
        revenue = amount * exit_price * (1 - self.fee_rate)
        cost = amount * entry_price
        pnl = revenue - cost
        pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0.0

        commission = amount * (entry_price + exit_price) * self.fee_rate
        slippage = amount * (entry_price + exit_price) * self.slippage_rate

        return TradeCosts(
            revenue=revenue,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=slippage,
            net_amount=amount,
        )

    def calculate_whipsaw_costs(
        self,
        buy_price: float,
        sell_price: float,
        invest_amount: float,
    ) -> TradeCosts:
        """
        Calculate costs for a same-day entry and exit (whipsaw).

        Args:
            buy_price: Buy price per unit
            sell_price: Sell price per unit
            invest_amount: Initial investment amount

        Returns:
            TradeCosts with all calculated values
        """
        amount = self.calculate_buy_amount(invest_amount, buy_price)
        return_money = amount * sell_price * (1 - self.fee_rate)

        pnl = return_money - invest_amount
        pnl_pct = (sell_price / buy_price - 1) * 100 if buy_price > 0 else 0.0

        commission = amount * (buy_price + sell_price) * self.fee_rate
        slippage = amount * (buy_price + sell_price) * self.slippage_rate

        return TradeCosts(
            revenue=return_money,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=slippage,
            net_amount=amount,
        )

    def apply_slippage(self, price: float, is_buy: bool = True) -> float:
        """
        Apply slippage to a price.

        Args:
            price: Base price
            is_buy: True if buying (price increases), False if selling (price decreases)

        Returns:
            Price adjusted for slippage
        """
        if is_buy:
            return price * (1 + self.slippage_rate)
        return price * (1 - self.slippage_rate)
