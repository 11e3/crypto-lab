"""
Protocol definitions for exchange services.

Provides minimal interfaces for dependency injection following ISP (Interface Segregation Principle).
Each protocol defines only the methods needed by specific consumers.
"""

from typing import Protocol

import pandas as pd

from src.exchange.types import Balance, Order


class PriceService(Protocol):
    """Protocol for price queries.

    Used by: PositionManager
    """

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a trading pair."""
        ...


class MarketDataService(Protocol):
    """Protocol for market data access.

    Used by: SignalHandler
    """

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a trading pair."""
        ...

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "day",
        count: int = 200,
    ) -> pd.DataFrame | None:
        """Get OHLCV (candlestick) data."""
        ...


class OrderExecutionService(Protocol):
    """Protocol for order execution.

    Used by: OrderManager
    """

    def buy_market_order(self, symbol: str, amount: float) -> Order:
        """Place a market buy order."""
        ...

    def sell_market_order(self, symbol: str, amount: float) -> Order:
        """Place a market sell order."""
        ...

    def get_order_status(self, order_id: str) -> Order:
        """Get status of an existing order."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        ...

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a trading pair."""
        ...


class BalanceService(Protocol):
    """Protocol for balance queries.

    Used by: TradingBotFacade
    """

    def get_balance(self, currency: str) -> Balance:
        """Get balance for a currency."""
        ...
