"""
Exchange abstraction layer for trading operations.

Provides a unified interface for different cryptocurrency exchanges.
"""

from src.exceptions.exchange import (
    ExchangeAuthenticationError,
    ExchangeConnectionError,
    ExchangeError,
    ExchangeOrderError,
    InsufficientBalanceError,
)
from src.exchange.base import Exchange
from src.exchange.factory import ExchangeFactory, ExchangeName
from src.exchange.types import Balance, Order, OrderSide, OrderStatus, OrderType, Ticker
from src.exchange.upbit import UpbitExchange

__all__ = [
    "Exchange",
    "UpbitExchange",
    "ExchangeFactory",
    "ExchangeName",
    "Balance",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Ticker",
    "ExchangeError",
    "ExchangeConnectionError",
    "ExchangeAuthenticationError",
    "ExchangeOrderError",
    "InsufficientBalanceError",
]
