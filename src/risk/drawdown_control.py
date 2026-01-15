"""
Drawdown control and risk management.

Provides:
- Dynamic position sizing based on drawdown
- Risk budgeting
- Automatic deleveraging
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskState(str, Enum):
    """Current risk state of portfolio."""

    NORMAL = "normal"  # Normal operations
    CAUTIOUS = "cautious"  # Slightly elevated risk
    DEFENSIVE = "defensive"  # High risk, reduce exposure
    CRITICAL = "critical"  # Emergency risk level


@dataclass
class DrawdownState:
    """Current drawdown state."""

    current_drawdown: float  # Current drawdown (negative)
    peak_value: float  # Peak portfolio value
    current_value: float  # Current portfolio value
    drawdown_start_date: datetime | None = None
    days_in_drawdown: int = 0
    risk_state: RiskState = RiskState.NORMAL


@dataclass
class RiskBudget:
    """Risk budget allocation."""

    total_risk_budget: float  # Total portfolio risk budget (e.g., 0.20 for 20%)
    asset_risk_budgets: dict[str, float] = field(default_factory=dict)  # Per-asset
    used_risk_budget: float = 0.0  # Currently used
    remaining_risk_budget: float = 0.0  # Available

    def __post_init__(self):
        self.remaining_risk_budget = self.total_risk_budget - self.used_risk_budget


class DrawdownController:
    """
    Dynamic drawdown control.

    Automatically adjusts portfolio exposure based on drawdown levels.

    Example:
        >>> controller = DrawdownController(
        ...     max_drawdown_limit=0.20,  # 20% max DD
        ...     deleveraging_start=0.10,  # Start reducing at 10%
        ... )
        >>> action = controller.get_action(current_drawdown=-0.15)
        >>> # action.exposure_adjustment = 0.5 (reduce to 50%)
    """

    def __init__(
        self,
        max_drawdown_limit: float = 0.20,
        deleveraging_start: float = 0.10,
        recovery_threshold: float = 0.05,
        min_exposure: float = 0.20,
        deleveraging_speed: Literal["gradual", "aggressive", "immediate"] = "gradual",
    ) -> None:
        """
        Initialize drawdown controller.

        Args:
            max_drawdown_limit: Maximum allowed drawdown
            deleveraging_start: Drawdown level to start deleveraging
            recovery_threshold: Drawdown level to return to normal
            min_exposure: Minimum exposure (even in critical state)
            deleveraging_speed: How quickly to reduce exposure
        """
        self.max_drawdown_limit = max_drawdown_limit
        self.deleveraging_start = deleveraging_start
        self.recovery_threshold = recovery_threshold
        self.min_exposure = min_exposure
        self.deleveraging_speed = deleveraging_speed

        # State tracking
        self._peak_value: float = 0.0
        self._current_state: RiskState = RiskState.NORMAL
        self._drawdown_start: datetime | None = None

    def update(
        self,
        portfolio_value: float,
        timestamp: datetime | None = None,
    ) -> DrawdownState:
        """
        Update controller with current portfolio value.

        Args:
            portfolio_value: Current portfolio value
            timestamp: Current timestamp

        Returns:
            Current DrawdownState
        """
        timestamp = timestamp or datetime.now()

        # Update peak
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            self._drawdown_start = None

        # Calculate drawdown
        if self._peak_value > 0:
            drawdown = (portfolio_value - self._peak_value) / self._peak_value
        else:
            drawdown = 0.0

        # Track drawdown duration
        days_in_dd = 0
        if drawdown < 0:
            if self._drawdown_start is None:
                self._drawdown_start = timestamp
            days_in_dd = (timestamp - self._drawdown_start).days

        # Update risk state
        self._current_state = self._determine_risk_state(drawdown)

        return DrawdownState(
            current_drawdown=drawdown,
            peak_value=self._peak_value,
            current_value=portfolio_value,
            drawdown_start_date=self._drawdown_start,
            days_in_drawdown=days_in_dd,
            risk_state=self._current_state,
        )

    def get_target_exposure(
        self,
        current_drawdown: float,
    ) -> float:
        """
        Calculate target exposure based on drawdown.

        Args:
            current_drawdown: Current drawdown (negative number)

        Returns:
            Target exposure multiplier (0 to 1)
        """
        dd = abs(current_drawdown)

        # No deleveraging needed
        if dd <= self.deleveraging_start:
            return 1.0

        # Full deleveraging
        if dd >= self.max_drawdown_limit:
            return self.min_exposure

        # Gradual deleveraging
        if self.deleveraging_speed == "immediate":
            return self.min_exposure

        elif self.deleveraging_speed == "aggressive":
            # Rapid reduction
            progress = (dd - self.deleveraging_start) / (
                self.max_drawdown_limit - self.deleveraging_start
            )
            return max(
                self.min_exposure,
                1.0 - (1.0 - self.min_exposure) * (progress ** 0.5),
            )

        else:  # gradual
            # Linear reduction
            progress = (dd - self.deleveraging_start) / (
                self.max_drawdown_limit - self.deleveraging_start
            )
            return max(
                self.min_exposure,
                1.0 - (1.0 - self.min_exposure) * progress,
            )

    def get_position_adjustment(
        self,
        current_positions: dict[str, float],
        current_drawdown: float,
    ) -> dict[str, float]:
        """
        Calculate position adjustments based on drawdown.

        Args:
            current_positions: Current position values
            current_drawdown: Current portfolio drawdown

        Returns:
            Target position values after adjustment
        """
        target_exposure = self.get_target_exposure(current_drawdown)

        return {
            asset: value * target_exposure
            for asset, value in current_positions.items()
        }

    def should_close_all(self, current_drawdown: float) -> bool:
        """Check if all positions should be closed."""
        return abs(current_drawdown) >= self.max_drawdown_limit * 1.5

    def can_add_risk(self, current_drawdown: float) -> bool:
        """Check if new risk can be added."""
        return abs(current_drawdown) < self.deleveraging_start

    def _determine_risk_state(self, drawdown: float) -> RiskState:
        """Determine current risk state based on drawdown."""
        dd = abs(drawdown)

        if dd < self.recovery_threshold:
            return RiskState.NORMAL
        elif dd < self.deleveraging_start:
            return RiskState.CAUTIOUS
        elif dd < self.max_drawdown_limit:
            return RiskState.DEFENSIVE
        else:
            return RiskState.CRITICAL

    def reset(self, initial_value: float = 0.0) -> None:
        """Reset controller state."""
        self._peak_value = initial_value
        self._current_state = RiskState.NORMAL
        self._drawdown_start = None


class RiskBudgetManager:
    """
    Risk budget allocation manager.

    Allocates and tracks risk budget across positions.

    Example:
        >>> manager = RiskBudgetManager(total_budget=0.20)
        >>> allocation = manager.allocate_risk(
        ...     assets=["BTC", "ETH", "SOL"],
        ...     volatilities={"BTC": 0.60, "ETH": 0.70, "SOL": 0.80},
        ...     method="inverse_vol",
        ... )
    """

    def __init__(
        self,
        total_budget: float = 0.20,
        max_single_asset: float = 0.05,
    ) -> None:
        """
        Initialize risk budget manager.

        Args:
            total_budget: Total portfolio risk budget
            max_single_asset: Maximum risk budget per asset
        """
        self.total_budget = total_budget
        self.max_single_asset = max_single_asset
        self._allocations: dict[str, float] = {}
        self._used: float = 0.0

    def allocate_risk(
        self,
        assets: list[str],
        volatilities: dict[str, float],
        method: Literal["equal", "inverse_vol", "proportional"] = "inverse_vol",
    ) -> dict[str, float]:
        """
        Allocate risk budget to assets.

        Args:
            assets: Assets to allocate to
            volatilities: Volatility for each asset
            method: Allocation method

        Returns:
            Risk budget per asset
        """
        if method == "equal":
            budget_per_asset = self.total_budget / len(assets)
            allocations = {
                asset: min(budget_per_asset, self.max_single_asset)
                for asset in assets
            }

        elif method == "inverse_vol":
            # Lower volatility gets more budget
            inv_vols = {
                asset: 1 / volatilities.get(asset, 1.0)
                for asset in assets
            }
            total_inv_vol = sum(inv_vols.values())

            allocations = {
                asset: min(
                    self.total_budget * (inv_vol / total_inv_vol),
                    self.max_single_asset,
                )
                for asset, inv_vol in inv_vols.items()
            }

        elif method == "proportional":
            # Proportional to volatility (risk parity-like)
            total_vol = sum(volatilities.get(a, 0.5) for a in assets)

            allocations = {
                asset: min(
                    self.total_budget * (volatilities.get(asset, 0.5) / total_vol),
                    self.max_single_asset,
                )
                for asset in assets
            }

        else:
            allocations = {asset: self.max_single_asset for asset in assets}

        self._allocations = allocations
        self._used = sum(allocations.values())

        return allocations

    def get_position_size(
        self,
        asset: str,
        volatility: float,
        portfolio_value: float,
    ) -> float:
        """
        Calculate position size from risk budget.

        Args:
            asset: Asset symbol
            volatility: Asset volatility (annualized)
            portfolio_value: Total portfolio value

        Returns:
            Position size in value terms
        """
        risk_budget = self._allocations.get(asset, self.max_single_asset)

        # Position size = (Risk Budget * Portfolio Value) / Volatility
        position_size = (risk_budget * portfolio_value) / volatility

        return position_size

    def check_budget(self, asset: str, additional_risk: float) -> bool:
        """
        Check if additional risk can be taken.

        Args:
            asset: Asset to check
            additional_risk: Additional risk to add

        Returns:
            True if within budget
        """
        current = self._allocations.get(asset, 0)
        if current + additional_risk > self.max_single_asset:
            return False

        if self._used + additional_risk > self.total_budget:
            return False

        return True

    def get_remaining_budget(self) -> float:
        """Get remaining risk budget."""
        return self.total_budget - self._used

    def get_budget(self) -> RiskBudget:
        """Get current risk budget state."""
        return RiskBudget(
            total_risk_budget=self.total_budget,
            asset_risk_budgets=self._allocations.copy(),
            used_risk_budget=self._used,
        )


__all__ = [
    "DrawdownController",
    "DrawdownState",
    "RiskState",
    "RiskBudgetManager",
    "RiskBudget",
]
