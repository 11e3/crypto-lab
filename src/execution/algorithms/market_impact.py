"""
Market impact modeling.

Provides models for estimating execution costs due to market impact.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class ImpactEstimate:
    """Estimated market impact."""

    temporary_impact: float  # Immediate price impact (reverts)
    permanent_impact: float  # Lasting price impact
    total_impact: float  # Total impact cost

    # Breakdown
    spread_cost: float = 0.0
    timing_cost: float = 0.0

    # Confidence
    confidence_low: float = 0.0
    confidence_high: float = 0.0


class MarketImpactModel:
    """
    Market impact estimation model.

    Based on Almgren-Chriss framework for optimal execution.

    The model estimates both:
    - Temporary impact: Immediate price movement that reverts
    - Permanent impact: Lasting effect on market price

    Example:
        >>> model = MarketImpactModel(
        ...     daily_volume=1_000_000,
        ...     daily_volatility=0.02,
        ...     spread=0.001,
        ... )
        >>> impact = model.estimate_impact(
        ...     order_size=50_000,
        ...     duration_hours=2,
        ... )
        >>> print(f"Total impact: {impact.total_impact:.4%}")
    """

    def __init__(
        self,
        daily_volume: float,
        daily_volatility: float,
        spread: float = 0.001,
        temporary_impact_coef: float = 0.1,
        permanent_impact_coef: float = 0.05,
    ) -> None:
        """
        Initialize market impact model.

        Args:
            daily_volume: Average daily volume
            daily_volatility: Daily volatility (e.g., 0.02 = 2%)
            spread: Bid-ask spread
            temporary_impact_coef: Coefficient for temporary impact
            permanent_impact_coef: Coefficient for permanent impact
        """
        self.daily_volume = daily_volume
        self.daily_volatility = daily_volatility
        self.spread = spread
        self.temporary_impact_coef = temporary_impact_coef
        self.permanent_impact_coef = permanent_impact_coef

    def estimate_impact(
        self,
        order_size: float,
        duration_hours: float = 1.0,
        urgency: float = 0.5,
    ) -> ImpactEstimate:
        """
        Estimate market impact for an order.

        Args:
            order_size: Order size in shares/units
            duration_hours: Execution duration in hours
            urgency: Execution urgency (0=patient, 1=aggressive)

        Returns:
            ImpactEstimate with breakdown
        """
        # Participation rate
        trading_hours = 8  # Assume 8 hour trading day
        duration_fraction = duration_hours / trading_hours
        expected_volume = self.daily_volume * duration_fraction
        participation = order_size / expected_volume if expected_volume > 0 else 1.0

        # Temporary impact (square root model)
        # Impact increases with sqrt of participation rate
        temp_impact = (
            self.temporary_impact_coef *
            self.daily_volatility *
            np.sqrt(participation) *
            (1 + urgency)  # Urgency increases impact
        )

        # Permanent impact (linear model)
        perm_impact = (
            self.permanent_impact_coef *
            self.daily_volatility *
            participation
        )

        # Spread cost (half spread for crossing)
        spread_cost = self.spread / 2

        # Timing cost (opportunity cost of not executing immediately)
        # Higher for patient execution
        timing_cost = (
            self.daily_volatility *
            np.sqrt(duration_fraction) *
            (1 - urgency)
        )

        # Total impact
        total = temp_impact + perm_impact + spread_cost

        # Confidence interval (rough estimate)
        std_error = self.daily_volatility * np.sqrt(participation) * 0.5
        conf_low = total - 2 * std_error
        conf_high = total + 2 * std_error

        return ImpactEstimate(
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            total_impact=total,
            spread_cost=spread_cost,
            timing_cost=timing_cost,
            confidence_low=max(0, conf_low),
            confidence_high=conf_high,
        )

    def optimal_duration(
        self,
        order_size: float,
        risk_aversion: float = 0.5,
    ) -> float:
        """
        Calculate optimal execution duration.

        Balances market impact (favors slow) vs timing risk (favors fast).

        Args:
            order_size: Order size
            risk_aversion: Risk aversion parameter (0-1)

        Returns:
            Optimal duration in hours
        """
        participation = order_size / self.daily_volume

        # Simple heuristic: larger orders need more time
        # Risk averse traders accept more timing risk for less impact
        base_duration = np.sqrt(participation) * 8  # Base on trading day

        # Adjust for risk aversion
        duration = base_duration * (1 + (1 - risk_aversion))

        # Cap at trading day
        return min(8, max(0.5, duration))

    def optimal_trajectory(
        self,
        order_size: float,
        duration_hours: float,
        num_periods: int = 10,
        risk_aversion: float = 0.5,
    ) -> list[float]:
        """
        Calculate optimal execution trajectory.

        Uses Almgren-Chriss optimal execution framework.

        Args:
            order_size: Total order size
            duration_hours: Total execution duration
            num_periods: Number of periods
            risk_aversion: Risk aversion parameter

        Returns:
            List of quantities per period
        """
        # Simplified Almgren-Chriss trajectory
        # More risk-averse = more front-loaded
        # Less risk-averse = more even

        trajectory = []
        remaining = order_size

        for i in range(num_periods):
            progress = i / num_periods

            # Exponential decay based on risk aversion
            decay_rate = 1 + risk_aversion * 2
            weight = np.exp(-decay_rate * progress)

            # Normalize weights
            total_weight = sum(
                np.exp(-decay_rate * j / num_periods)
                for j in range(i, num_periods)
            )

            quantity = remaining * weight / total_weight if total_weight > 0 else remaining / (num_periods - i)
            trajectory.append(quantity)
            remaining -= quantity

        return trajectory


class AlmgrenChrissModel(MarketImpactModel):
    """
    Full Almgren-Chriss optimal execution model.

    Provides more sophisticated impact estimation and optimal trajectories.
    """

    def __init__(
        self,
        daily_volume: float,
        daily_volatility: float,
        spread: float = 0.001,
        eta: float = 0.1,  # Temporary impact parameter
        gamma: float = 0.05,  # Permanent impact parameter
        risk_aversion: float = 1e-6,
    ) -> None:
        """
        Initialize Almgren-Chriss model.

        Args:
            eta: Temporary impact coefficient
            gamma: Permanent impact coefficient
            risk_aversion: Risk aversion parameter
        """
        super().__init__(
            daily_volume, daily_volatility, spread,
            eta, gamma
        )
        self.eta = eta
        self.gamma = gamma
        self.risk_aversion = risk_aversion

    def calculate_execution_cost(
        self,
        trajectory: list[float],
        period_length: float,
    ) -> dict[str, float]:
        """
        Calculate expected execution cost for a trajectory.

        Args:
            trajectory: List of quantities per period
            period_length: Length of each period in hours

        Returns:
            Dict with cost breakdown
        """
        n = len(trajectory)

        # Holdings at each time
        holdings = [sum(trajectory[i:]) for i in range(n)]
        holdings.append(0)  # Final position is 0

        # Permanent impact cost
        perm_cost = self.gamma * sum(trajectory) ** 2 / 2

        # Temporary impact cost
        temp_cost = self.eta * sum(q ** 2 for q in trajectory)

        # Timing risk (variance of cost)
        vol_per_period = self.daily_volatility * np.sqrt(period_length / 8)
        timing_risk = sum(
            h ** 2 * vol_per_period ** 2
            for h in holdings[:-1]
        )

        return {
            "permanent_cost": perm_cost,
            "temporary_cost": temp_cost,
            "timing_risk": timing_risk,
            "total_cost": perm_cost + temp_cost,
            "risk_adjusted_cost": perm_cost + temp_cost + self.risk_aversion * timing_risk,
        }


def estimate_crypto_impact(
    order_value: float,
    market_cap: float | None = None,
    daily_volume_value: float | None = None,
    volatility: float = 0.05,
) -> float:
    """
    Estimate market impact for crypto assets.

    Crypto markets often have higher impact due to lower liquidity.

    Args:
        order_value: Order value in currency
        market_cap: Asset market cap
        daily_volume_value: Daily trading volume value
        volatility: Asset volatility

    Returns:
        Estimated impact as fraction
    """
    if daily_volume_value is None:
        # Rough estimate based on market cap
        if market_cap:
            daily_volume_value = market_cap * 0.05  # Assume 5% turnover
        else:
            daily_volume_value = order_value * 10  # Very rough estimate

    participation = order_value / daily_volume_value

    # Crypto impact tends to be higher
    crypto_factor = 1.5

    # Square root impact model
    impact = crypto_factor * volatility * np.sqrt(participation)

    return min(0.10, impact)  # Cap at 10%


__all__ = [
    "MarketImpactModel",
    "AlmgrenChrissModel",
    "ImpactEstimate",
    "estimate_crypto_impact",
]
