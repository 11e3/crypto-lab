"""
Multi-asset position sizing strategies.
"""

from collections.abc import Mapping

import pandas as pd

from src.risk.position_sizing import PositionSizingMethod
from src.risk.volatility import calculate_return_volatility

__all__ = ["calculate_multi_asset_position_sizes"]


def calculate_multi_asset_position_sizes(
    method: PositionSizingMethod,
    available_cash: float,
    tickers: list[str],
    current_prices: Mapping[str, float],
    historical_data: dict[str, pd.DataFrame],
    target_risk_pct: float = 0.02,
    lookback_period: int = 20,
) -> dict[str, float]:
    """
    Calculate position sizes for multiple assets simultaneously.

    This is used when multiple entry signals occur at the same time.

    Args:
        method: Position sizing method
        available_cash: Total available cash
        tickers: List of tickers to size positions for
        current_prices: Current prices for each ticker
        historical_data: Historical data for each ticker
        target_risk_pct: Target risk per position (for fixed-risk)
        lookback_period: Lookback period for volatility

    Returns:
        Dictionary mapping ticker to position size
    """
    if method == "equal":
        size_per_ticker = available_cash / len(tickers)
        return dict.fromkeys(tickers, size_per_ticker)

    if method == "volatility" or method == "inverse-volatility":
        return _inverse_volatility_multi(available_cash, tickers, historical_data, lookback_period)

    if method == "fixed-risk":
        return _fixed_risk_multi(
            available_cash,
            tickers,
            current_prices,
            historical_data,
            target_risk_pct,
            lookback_period,
        )

    # Fallback to equal
    size_per_ticker = available_cash / len(tickers)
    return dict.fromkeys(tickers, size_per_ticker)


def _inverse_volatility_multi(
    available_cash: float,
    tickers: list[str],
    historical_data: dict[str, pd.DataFrame],
    lookback_period: int,
) -> dict[str, float]:
    """Calculate inverse volatility weights for multiple assets."""
    weights: dict[str, float] = {}
    total_weight = 0.0

    for ticker in tickers:
        data = historical_data.get(ticker)
        volatility = (
            calculate_return_volatility(data, lookback_period) if data is not None else None
        )

        if volatility is None:
            weights[ticker] = 1.0
        else:
            weights[ticker] = 1.0 / volatility

        total_weight += weights[ticker]

    if total_weight > 0:
        return {ticker: (available_cash * weights[ticker] / total_weight) for ticker in tickers}
    return {ticker: available_cash / len(tickers) for ticker in tickers}


def _fixed_risk_multi(
    available_cash: float,
    tickers: list[str],
    current_prices: Mapping[str, float],
    historical_data: dict[str, pd.DataFrame],
    target_risk_pct: float,
    lookback_period: int,
) -> dict[str, float]:
    """Calculate fixed risk position sizes for multiple assets."""
    position_values: dict[str, float] = {}

    for ticker in tickers:
        if ticker not in current_prices or current_prices[ticker] <= 0:
            position_values[ticker] = 0.0
            continue

        data = historical_data.get(ticker)
        volatility = (
            calculate_return_volatility(data, lookback_period) if data is not None else None
        )

        if volatility is None:
            position_values[ticker] = available_cash / len(tickers)
        else:
            target_risk_amount = available_cash * target_risk_pct
            position_value = target_risk_amount / volatility
            position_values[ticker] = position_value

    # Normalize to ensure total doesn't exceed available cash
    total_value = sum(position_values.values())
    if total_value > available_cash:
        scale = available_cash / total_value
        position_values = {k: v * scale for k, v in position_values.items()}

    return position_values
