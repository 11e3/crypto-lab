"""
VWAP (Volume-Weighted Average Price) execution algorithm.

Splits orders according to historical volume profile to achieve VWAP.
"""

from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from src.execution.algorithms.base import ExecutionAlgorithm, OrderSlice


class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price algorithm.

    Executes order proportionally to historical volume distribution.
    Achieves execution price close to volume-weighted average.

    Best for:
    - Orders that need to match VWAP benchmark
    - Stocks with predictable intraday volume patterns
    - Minimizing market impact

    Example:
        >>> # Historical volume by hour
        >>> volume_profile = {
        ...     9: 0.15, 10: 0.12, 11: 0.08, 12: 0.05,
        ...     13: 0.08, 14: 0.12, 15: 0.18, 16: 0.22,
        ... }
        >>> algo = VWAPAlgorithm(
        ...     ticker="AAPL",
        ...     side="buy",
        ...     total_quantity=1000,
        ...     arrival_price=150,
        ...     volume_profile=volume_profile,
        ... )
    """

    def __init__(
        self,
        ticker: str,
        side: Literal["buy", "sell"],
        total_quantity: float,
        arrival_price: float,
        volume_profile: dict[int, float] | None = None,
        duration_minutes: int = 60,
        num_slices: int = 10,
        start_time: datetime | None = None,
        participation_rate: float = 0.10,
    ) -> None:
        """
        Initialize VWAP algorithm.

        Args:
            ticker: Asset ticker
            side: Buy or sell
            total_quantity: Total quantity to execute
            arrival_price: Price at order arrival
            volume_profile: Volume distribution by hour (0-23) -> fraction
            duration_minutes: Execution duration
            num_slices: Number of slices
            start_time: Execution start time
            participation_rate: Maximum participation of market volume
        """
        super().__init__(ticker, side, total_quantity, arrival_price)

        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
        self.start_time = start_time or datetime.now()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)
        self.participation_rate = participation_rate

        # Default volume profile if not provided
        self.volume_profile = volume_profile or self._default_volume_profile()

        self._current_slice_index = 0
        self.slices = self.generate_schedule()

    def _default_volume_profile(self) -> dict[int, float]:
        """Generate default U-shaped volume profile."""
        # Typical U-shaped intraday volume pattern
        profile = {}
        hours = list(range(24))

        for h in hours:
            if 9 <= h <= 10:  # Morning high
                profile[h] = 0.15
            elif 10 < h <= 12:  # Mid-morning decline
                profile[h] = 0.08
            elif 12 < h <= 14:  # Lunch lull
                profile[h] = 0.05
            elif 14 < h <= 15:  # Afternoon pickup
                profile[h] = 0.10
            elif 15 < h <= 16:  # Closing surge
                profile[h] = 0.20
            else:
                profile[h] = 0.02  # Off hours

        # Normalize
        total = sum(profile.values())
        return {h: v / total for h, v in profile.items()}

    def generate_schedule(self) -> list[OrderSlice]:
        """Generate VWAP execution schedule based on volume profile."""
        slices = []
        interval = timedelta(minutes=self.duration_minutes / self.num_slices)

        # Calculate volume weights for each slice
        slice_weights = []
        for i in range(self.num_slices):
            target_time = self.start_time + interval * i
            hour = target_time.hour
            weight = self.volume_profile.get(hour, 0.05)
            slice_weights.append(weight)

        # Normalize weights
        total_weight = sum(slice_weights)
        if total_weight > 0:
            slice_weights = [w / total_weight for w in slice_weights]
        else:
            slice_weights = [1.0 / self.num_slices] * self.num_slices

        # Create slices
        for i in range(self.num_slices):
            target_time = self.start_time + interval * i
            quantity = self.total_quantity * slice_weights[i]

            slice = OrderSlice(
                quantity=quantity,
                target_time=target_time,
                urgency=0.5,
            )
            slices.append(slice)

        return slices

    def get_next_slice(self, current_time: datetime) -> OrderSlice | None:
        """Get next slice to execute."""
        if self._current_slice_index >= len(self.slices):
            return None

        next_slice = self.slices[self._current_slice_index]

        if current_time >= next_slice.target_time:
            self._current_slice_index += 1
            return next_slice

        return None

    def adjust_for_actual_volume(
        self,
        actual_volume: float,
        expected_volume: float,
    ) -> float:
        """
        Adjust next slice based on actual vs expected volume.

        Args:
            actual_volume: Actual market volume observed
            expected_volume: Expected volume from profile

        Returns:
            Adjustment factor for next slice
        """
        if expected_volume == 0:
            return 1.0

        ratio = actual_volume / expected_volume

        # If volume is higher than expected, we can execute more
        # If lower, execute less
        return min(2.0, max(0.5, ratio))

    def calculate_vwap_target(
        self,
        prices: pd.Series,
        volumes: pd.Series,
    ) -> float:
        """
        Calculate VWAP from price/volume data.

        Args:
            prices: Price series
            volumes: Volume series

        Returns:
            VWAP value
        """
        if len(prices) == 0 or volumes.sum() == 0:
            return self.arrival_price

        return (prices * volumes).sum() / volumes.sum()


class ParticipationVWAP(VWAPAlgorithm):
    """
    Participation-rate VWAP.

    Executes as a percentage of market volume to minimize impact.
    """

    def __init__(
        self,
        ticker: str,
        side: Literal["buy", "sell"],
        total_quantity: float,
        arrival_price: float,
        participation_rate: float = 0.10,
        max_participation: float = 0.20,
        **kwargs,
    ) -> None:
        """
        Initialize Participation VWAP.

        Args:
            participation_rate: Target % of market volume
            max_participation: Maximum % of market volume
        """
        super().__init__(
            ticker, side, total_quantity, arrival_price,
            participation_rate=participation_rate, **kwargs
        )
        self.max_participation = max_participation

    def calculate_slice_quantity(
        self,
        market_volume: float,
    ) -> float:
        """
        Calculate quantity based on market volume.

        Args:
            market_volume: Current market volume

        Returns:
            Quantity to execute this slice
        """
        # Target quantity based on participation rate
        target = market_volume * self.participation_rate

        # Cap at max participation
        max_qty = market_volume * self.max_participation

        # Don't exceed remaining quantity
        remaining = self.remaining_quantity

        return min(target, max_qty, remaining)


__all__ = ["VWAPAlgorithm", "ParticipationVWAP"]
