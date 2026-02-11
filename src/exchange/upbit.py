"""Upbit exchange implementation."""

from datetime import datetime
from typing import cast

import pandas as pd
import pyupbit
import requests

from src.exceptions.exchange import ExchangeConnectionError, ExchangeError
from src.exchange.base import Exchange
from src.exchange.types import Balance, Order, Ticker
from src.exchange.upbit_orders import (
    cancel_existing_order,
    execute_buy_market_order,
    execute_sell_market_order,
    fetch_order_status,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Common exception types for Upbit API calls
_UPBIT_ERRORS = (
    requests.RequestException,
    ConnectionError,
    OSError,
    ValueError,
    KeyError,
    TypeError,
)


class UpbitExchange(Exchange):
    """Upbit exchange implementation. Wraps pyupbit library."""

    def __init__(self, access_key: str, secret_key: str) -> None:
        """Initialize Upbit exchange client."""
        try:
            self.client = pyupbit.Upbit(access_key, secret_key)
            self.client.get_balances()
        except _UPBIT_ERRORS as e:
            logger.error(f"Failed to initialize Upbit client: {e}", exc_info=True)
            raise ExchangeConnectionError(f"Failed to connect to Upbit: {e}") from e

    def get_balance(self, currency: str) -> Balance:
        """Get balance for a currency."""
        try:
            balance = self.client.get_balance(currency)
            if balance is None:
                return Balance(currency=currency, balance=0.0, locked=0.0)

            locked = self._get_locked_amount(currency)
            return Balance(currency=currency, balance=float(balance), locked=locked)
        except _UPBIT_ERRORS as e:
            logger.error(f"Error getting balance for {currency}: {e}", exc_info=True)
            raise ExchangeError(f"Failed to get balance for {currency}: {e}") from e

    def _get_locked_amount(self, currency: str) -> float:
        """Get locked amount from pending orders."""
        try:
            orders = self.client.get_order(currency, state="wait")
            if orders and isinstance(orders, list):
                return sum(
                    float(order.get("locked", 0.0)) for order in orders if isinstance(order, dict)
                )
        except _UPBIT_ERRORS as e:
            logger.warning(f"Could not retrieve locked amount for {currency}: {e}")
        return 0.0

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a trading pair."""
        try:
            price = pyupbit.get_current_price(symbol)
            if price is None:
                raise ExchangeError(f"No price data available for {symbol}")
            return float(price)
        except ExchangeError:
            raise
        except _UPBIT_ERRORS as e:
            logger.error(f"Error getting price for {symbol}: {e}", exc_info=True)
            raise ExchangeError(f"Failed to get price for {symbol}: {e}") from e

    def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker information for a trading pair."""
        try:
            ticker_data = pyupbit.get_ticker(symbol)
            if ticker_data is None:
                raise ExchangeError(f"No ticker data available for {symbol}")

            price = float(ticker_data.get("trade_price", 0.0))
            volume = float(ticker_data.get("acc_trade_volume_24h", 0.0))

            return Ticker(
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=datetime.now(),
            )
        except ExchangeError:
            raise
        except _UPBIT_ERRORS as e:
            logger.error(f"Error getting ticker for {symbol}: {e}", exc_info=True)
            raise ExchangeError(f"Failed to get ticker for {symbol}: {e}") from e

    def buy_market_order(self, symbol: str, amount: float) -> Order:
        """Place a market buy order."""
        return execute_buy_market_order(self.client, symbol, amount)

    def sell_market_order(self, symbol: str, amount: float) -> Order:
        """Place a market sell order."""
        return execute_sell_market_order(self.client, symbol, amount)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "day",
        count: int = 200,
    ) -> pd.DataFrame | None:
        """Get OHLCV (candlestick) data."""
        try:
            df = pyupbit.get_ohlcv(symbol, interval=interval, count=count)
            if df is None or len(df) == 0:
                logger.warning(f"No OHLCV data for {symbol}")
                return None
            return cast(pd.DataFrame, df)
        except _UPBIT_ERRORS as e:
            logger.error(f"Error getting OHLCV for {symbol}: {e}", exc_info=True)
            return None

    def get_order_status(self, order_id: str) -> Order:
        """Get status of an existing order."""
        return fetch_order_status(self.client, order_id)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        return cancel_existing_order(self.client, order_id)
