"""
TWAP (Time-Weighted Average Price) execution algorithm.

Splits orders evenly across time to achieve time-weighted average price.
"""

from datetime import datetime, timedelta
from typing import Literal
import random

from src.execution.algorithms.base import ExecutionAlgorithm, OrderSlice


class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price algorithm.

    Executes order in equal slices at regular intervals.
    Achieves execution price close to time-weighted average.

    Best for:
    - Low urgency orders
    - Markets with consistent liquidity
    - Minimizing timing risk

    Example:
        >>> algo = TWAPAlgorithm(
        ...     ticker="BTC",
        ...     side="buy",
        ...     total_quantity=10.0,
        ...     arrival_price=50000,
        ...     duration_minutes=60,
        ...     num_slices=12,
        ... )
        >>> schedule = algo.generate_schedule()
        >>> # Execute according to schedule
    """

    def __init__(
        self,
        ticker: str,
        side: Literal["buy", "sell"],
        total_quantity: float,
        arrival_price: float,
        duration_minutes: int = 60,
        num_slices: int = 10,
        randomize: bool = True,
        start_time: datetime | None = None,
    ) -> None:
        """
        Initialize TWAP algorithm.

        Args:
            ticker: Asset ticker
            side: Buy or sell
            total_quantity: Total quantity to execute
            arrival_price: Price at order arrival
            duration_minutes: Total execution duration
            num_slices: Number of slices to split order
            randomize: Add randomness to timing
            start_time: Execution start time
        """
        super().__init__(ticker, side, total_quantity, arrival_price)

        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
        self.randomize = randomize
        self.start_time = start_time or datetime.now()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)

        self._current_slice_index = 0
        self.slices = self.generate_schedule()

    def generate_schedule(self) -> list[OrderSlice]:
        """Generate TWAP execution schedule."""
        slices = []
        quantity_per_slice = self.total_quantity / self.num_slices
        interval = timedelta(minutes=self.duration_minutes / self.num_slices)

        for i in range(self.num_slices):
            target_time = self.start_time + interval * i

            # Add randomness to avoid predictability
            if self.randomize and i > 0:
                jitter_seconds = random.uniform(-30, 30)
                target_time += timedelta(seconds=jitter_seconds)

            slice = OrderSlice(
                quantity=quantity_per_slice,
                target_time=target_time,
                urgency=0.5,  # Moderate urgency for TWAP
            )
            slices.append(slice)

        return slices

    def get_next_slice(self, current_time: datetime) -> OrderSlice | None:
        """Get next slice to execute."""
        if self._current_slice_index >= len(self.slices):
            return None

        next_slice = self.slices[self._current_slice_index]

        # Check if it's time to execute
        if current_time >= next_slice.target_time:
            self._current_slice_index += 1
            return next_slice

        return None

    def get_slices_due(self, current_time: datetime) -> list[tuple[int, OrderSlice]]:
        """Get all slices that are due for execution."""
        due_slices = []
        for i, slice in enumerate(self.slices):
            if not slice.is_executed and current_time >= slice.target_time:
                due_slices.append((i, slice))
        return due_slices

    def adjust_remaining(self, market_conditions: dict) -> None:
        """
        Adjust remaining slices based on market conditions.

        Args:
            market_conditions: Dict with keys like 'volatility', 'spread', 'volume'
        """
        volatility = market_conditions.get("volatility", 1.0)
        spread = market_conditions.get("spread", 0.001)

        # Increase slice sizes in low volatility (faster execution)
        # Decrease in high volatility (more patient)
        for i in range(self._current_slice_index, len(self.slices)):
            slice = self.slices[i]
            if volatility > 1.5:
                slice.urgency = 0.3  # More passive
            elif volatility < 0.5:
                slice.urgency = 0.7  # More aggressive
            else:
                slice.urgency = 0.5


class AdaptiveTWAP(TWAPAlgorithm):
    """
    Adaptive TWAP that adjusts based on market conditions.

    Speeds up or slows down execution based on:
    - Price movement relative to arrival price
    - Current market volatility
    - Remaining time
    """

    def __init__(
        self,
        ticker: str,
        side: Literal["buy", "sell"],
        total_quantity: float,
        arrival_price: float,
        duration_minutes: int = 60,
        num_slices: int = 10,
        price_band: float = 0.02,  # 2% price band
        **kwargs,
    ) -> None:
        super().__init__(
            ticker, side, total_quantity, arrival_price,
            duration_minutes, num_slices, **kwargs
        )
        self.price_band = price_band

    def should_accelerate(self, current_price: float) -> bool:
        """Check if execution should accelerate."""
        if self.side == "buy":
            # Accelerate buying if price is below arrival (favorable)
            return current_price < self.arrival_price * (1 - self.price_band / 2)
        else:
            # Accelerate selling if price is above arrival (favorable)
            return current_price > self.arrival_price * (1 + self.price_band / 2)

    def should_slow_down(self, current_price: float) -> bool:
        """Check if execution should slow down."""
        if self.side == "buy":
            # Slow down buying if price is above arrival (unfavorable)
            return current_price > self.arrival_price * (1 + self.price_band / 2)
        else:
            # Slow down selling if price is below arrival (unfavorable)
            return current_price < self.arrival_price * (1 - self.price_band / 2)

    def get_adjusted_quantity(
        self,
        base_quantity: float,
        current_price: float,
    ) -> float:
        """Get adjusted quantity based on price."""
        if self.should_accelerate(current_price):
            return base_quantity * 1.5  # Execute 50% more
        elif self.should_slow_down(current_price):
            return base_quantity * 0.5  # Execute 50% less
        return base_quantity


__all__ = ["TWAPAlgorithm", "AdaptiveTWAP"]
