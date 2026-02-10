"""Upbit order-related operations."""

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from src.exceptions.exchange import (
    ExchangeError,
    ExchangeOrderError,
    InsufficientBalanceError,
)
from src.exchange.types import Order, OrderSide, OrderStatus, OrderType


def _get_logger() -> logging.Logger:
    """Lazy import logger to avoid circular imports."""
    from src.utils.logger import get_logger

    return get_logger(__name__)


def execute_buy_market_order(client: Any, symbol: str, amount: float) -> Order:
    """
    Place a market buy order.

    Args:
        client: pyupbit.Upbit client instance
        symbol: Trading pair symbol (e.g., 'KRW-BTC')
        amount: Amount to buy (in KRW)

    Returns:
        Order object

    Raises:
        InsufficientBalanceError: If insufficient balance
        ExchangeOrderError: If order placement fails
    """
    try:
        result = client.buy_market_order(symbol, amount)
        if not result or "uuid" not in result:
            raise ExchangeOrderError(f"Buy order failed: {result}")

        order_id: str = result["uuid"]
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=amount,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            metadata=result,
        )
    except (ExchangeOrderError, InsufficientBalanceError):
        raise
    except (requests.RequestException, ConnectionError, OSError, ValueError, KeyError, TypeError) as e:
        _get_logger().error(f"Error placing buy order for {symbol}: {e}", exc_info=True)
        if "insufficient" in str(e).lower():
            raise InsufficientBalanceError(f"Insufficient balance: {e}") from e
        raise ExchangeOrderError(f"Failed to place buy order: {e}") from e


def execute_sell_market_order(client: Any, symbol: str, amount: float) -> Order:
    """
    Place a market sell order.

    Args:
        client: pyupbit.Upbit client instance
        symbol: Trading pair symbol (e.g., 'KRW-BTC')
        amount: Amount to sell (in base currency, e.g., BTC)

    Returns:
        Order object

    Raises:
        InsufficientBalanceError: If insufficient balance
        ExchangeOrderError: If order placement fails
    """
    try:
        result = client.sell_market_order(symbol, amount)
        if not result or "uuid" not in result:
            raise ExchangeOrderError(f"Sell order failed: {result}")

        order_id: str = result["uuid"]
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=amount,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            metadata=result,
        )
    except (ExchangeOrderError, InsufficientBalanceError):
        raise
    except (requests.RequestException, ConnectionError, OSError, ValueError, KeyError, TypeError) as e:
        _get_logger().error(f"Error placing sell order for {symbol}: {e}", exc_info=True)
        if "insufficient" in str(e).lower():
            raise InsufficientBalanceError(f"Insufficient balance: {e}") from e
        raise ExchangeOrderError(f"Failed to place sell order: {e}") from e


def fetch_order_status(client: Any, order_id: str) -> Order:
    """
    Get status of an existing order.

    Args:
        client: pyupbit.Upbit client instance
        order_id: Order identifier (UUID)

    Returns:
        Order object with current status

    Raises:
        ExchangeError: If API call fails
    """
    try:
        result = client.get_order(order_id)

        if not result:
            raise ExchangeError(f"Order {order_id} not found")

        # Normalize response: convert list to single item if needed
        if isinstance(result, list):
            if len(result) == 0:
                raise ExchangeError(f"Order {order_id} not found")
            result = result[0]

        return _parse_order_result(order_id, result)
    except ExchangeError:
        raise
    except (requests.RequestException, ConnectionError, OSError, ValueError, KeyError, TypeError) as e:
        _get_logger().error(f"Error getting order status for {order_id}: {e}", exc_info=True)
        raise ExchangeError(f"Failed to get order status: {e}") from e


def _parse_order_result(order_id: str, result: dict[str, Any]) -> Order:
    """Parse order result from Upbit API format."""
    state = result.get("state", "wait")
    status_map = {
        "wait": OrderStatus.PENDING,
        "done": OrderStatus.FILLED,
        "cancel": OrderStatus.CANCELLED,
    }
    status = status_map.get(state, OrderStatus.PENDING)

    side_str = result.get("side", "")
    side = OrderSide.BUY if side_str == "bid" else OrderSide.SELL

    ord_type = result.get("ord_type", "")
    order_type = OrderType.MARKET if ord_type == "price" else OrderType.LIMIT

    return Order(
        order_id=order_id,
        symbol=result.get("market", ""),
        side=side,
        order_type=order_type,
        amount=float(result.get("volume", 0.0)),
        price=float(result.get("price", 0.0)) if result.get("price") else None,
        status=status,
        filled_amount=float(result.get("executed_volume", 0.0)),
        filled_price=float(result.get("avg_price", 0.0)) if result.get("avg_price") else None,
        created_at=pd.to_datetime(str(result["created_at"])) if result.get("created_at") else None,
        metadata=result,
    )


def cancel_existing_order(client: Any, order_id: str) -> bool:
    """
    Cancel an existing order.

    Args:
        client: pyupbit.Upbit client instance
        order_id: Order identifier (UUID)

    Returns:
        True if cancellation successful

    Raises:
        ExchangeError: If cancellation fails
    """
    try:
        result = client.cancel_order(order_id)
        return result is not None
    except (requests.RequestException, ConnectionError, OSError, ValueError, KeyError, TypeError) as e:
        _get_logger().error(f"Error cancelling order {order_id}: {e}", exc_info=True)
        raise ExchangeError(f"Failed to cancel order: {e}") from e
