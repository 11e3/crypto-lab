"""
Dynamic position sizing strategies.

Provides various position sizing methods:
- Equal: Equal allocation (default)
- Volatility-based: Inverse volatility weighting
- Fixed-risk: Fixed risk per position
- Inverse-volatility: Inverse volatility weighting
"""

from typing import Literal

import pandas as pd

from src.risk.volatility import calculate_return_volatility
from src.utils.logger import get_logger

logger = get_logger(__name__)

PositionSizingMethod = Literal["equal", "volatility", "fixed-risk", "inverse-volatility"]


def calculate_position_size(
    method: PositionSizingMethod,
    available_cash: float,
    available_slots: int,
    ticker: str,
    current_price: float,
    historical_data: pd.DataFrame | None = None,
    target_risk_pct: float = 0.02,  # 2% risk per position for fixed-risk
    lookback_period: int = 20,  # For volatility calculation
) -> float:
    """
    Calculate position size based on selected method.

    Args:
        method: Position sizing method
        available_cash: Available cash for investment
        available_slots: Number of available position slots
        ticker: Ticker symbol (for logging)
        current_price: Current price of the asset
        historical_data: Historical OHLCV data (required for volatility-based methods)
        target_risk_pct: Target risk percentage per position (for fixed-risk method)
        lookback_period: Lookback period for volatility calculation

    Returns:
        Position size in base currency (KRW for KRW pairs)
    """
    if available_slots <= 0:
        return 0.0

    if method == "equal":
        size = _equal_sizing(available_cash, available_slots)
        return max(0.0, min(size, available_cash))

    if historical_data is None or len(historical_data) < lookback_period:
        logger.warning(
            f"Insufficient data for {method} sizing for {ticker}, falling back to equal sizing"
        )
        return _equal_sizing(available_cash, available_slots)

    if method == "volatility":
        size = _volatility_based_sizing(
            available_cash, available_slots, historical_data, lookback_period
        )
    elif method == "fixed-risk":
        size = _fixed_risk_sizing(
            available_cash,
            available_slots,
            current_price,
            historical_data,
            target_risk_pct,
            lookback_period,
        )
    elif method == "inverse-volatility":
        size = _inverse_volatility_sizing(
            available_cash, available_slots, historical_data, lookback_period
        )
    else:
        logger.warning(f"Unknown position sizing method: {method}, using equal sizing")
        size = _equal_sizing(available_cash, available_slots)

    # Ensure position size is never negative or exceeding available cash
    return max(0.0, min(size, available_cash))


def _equal_sizing(available_cash: float, available_slots: int) -> float:
    """Equal allocation among available slots."""
    if available_slots <= 0:
        return 0.0
    return available_cash / available_slots


def _volatility_based_sizing(
    available_cash: float,
    available_slots: int,
    historical_data: pd.DataFrame,
    lookback_period: int,
) -> float:
    """
    Volatility-based position sizing.

    Allocates more capital to less volatile assets.
    Uses inverse volatility weighting normalized against a baseline.
    """
    volatility = calculate_return_volatility(historical_data, lookback_period)
    if volatility is None:
        return _equal_sizing(available_cash, available_slots)

    base_size = available_cash / available_slots
    # Scale by inverse volatility relative to baseline (2% daily)
    baseline_volatility = 0.02
    weight = baseline_volatility / volatility
    return float(base_size * weight)


def _fixed_risk_sizing(
    available_cash: float,
    available_slots: int,
    current_price: float,
    historical_data: pd.DataFrame,
    target_risk_pct: float,
    lookback_period: int,
) -> float:
    """
    Fixed-risk position sizing.

    Allocates capital such that each position has the same risk (volatility).
    Position size = (target_risk * portfolio_value) / (volatility * price)
    """
    volatility = calculate_return_volatility(historical_data, lookback_period)
    if volatility is None or current_price <= 0:
        return _equal_sizing(available_cash, available_slots)

    # Calculate position size based on target risk
    # Risk = position_value * volatility
    # position_value = target_risk / volatility
    target_risk_amount = available_cash * target_risk_pct
    position_value = float(target_risk_amount / volatility)

    # Limit to available cash per slot
    max_per_slot = available_cash / available_slots
    position_value = min(position_value, max_per_slot)

    return float(position_value)


def _inverse_volatility_sizing(
    available_cash: float,
    available_slots: int,
    historical_data: pd.DataFrame,
    lookback_period: int,
) -> float:
    """
    Inverse volatility position sizing.

    Allocates more to less volatile assets.
    Similar to volatility-based but with different normalization.
    """
    volatility = calculate_return_volatility(historical_data, lookback_period)
    if volatility is None:
        return _equal_sizing(available_cash, available_slots)

    # Inverse volatility: lower volatility gets more capital
    # For single asset, scale by inverse volatility
    base_size: float = float(available_cash / available_slots)
    # Normalize by average volatility (assume 0.02 = 2% daily volatility as baseline)
    baseline_volatility = 0.02
    weight: float = float(baseline_volatility / volatility)

    return base_size * weight


# Re-export from position_sizing_multi for backward compatibility
from src.risk.position_sizing_multi import (  # noqa: E402
    calculate_multi_asset_position_sizes as calculate_multi_asset_position_sizes,
)

__all__ = [
    "PositionSizingMethod",
    "calculate_position_size",
    "calculate_multi_asset_position_sizes",
]
