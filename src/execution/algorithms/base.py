"""
Base classes for execution algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np


@dataclass
class OrderSlice:
    """A single slice of an order for execution."""

    quantity: float
    target_time: datetime
    min_price: float | None = None
    max_price: float | None = None
    urgency: float = 0.5  # 0 = passive, 1 = aggressive

    # Execution result
    executed_quantity: float = 0.0
    executed_price: float = 0.0
    executed_time: datetime | None = None
    is_executed: bool = False


@dataclass
class ExecutionResult:
    """Result of order execution."""

    # Order info
    ticker: str
    side: Literal["buy", "sell"]
    total_quantity: float
    target_price: float  # Arrival price or VWAP target

    # Execution summary
    executed_quantity: float = 0.0
    average_price: float = 0.0
    total_cost: float = 0.0

    # Performance metrics
    slippage: float = 0.0  # Execution price vs target
    market_impact: float = 0.0  # Estimated price impact
    timing_cost: float = 0.0  # Cost of not executing immediately

    # Slices
    slices: list[OrderSlice] = field(default_factory=list)

    # Metadata
    algorithm: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    @property
    def implementation_shortfall(self) -> float:
        """Calculate implementation shortfall (total execution cost)."""
        return self.slippage + self.market_impact + self.timing_cost

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled."""
        if self.total_quantity == 0:
            return 0
        return self.executed_quantity / self.total_quantity

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Execution Result: {self.ticker} {self.side.upper()}\n"
            f"{'=' * 40}\n"
            f"Target Quantity: {self.total_quantity:,.2f}\n"
            f"Executed Quantity: {self.executed_quantity:,.2f} ({self.fill_rate:.1%})\n"
            f"Average Price: {self.average_price:,.2f}\n"
            f"Target Price: {self.target_price:,.2f}\n"
            f"Slippage: {self.slippage:.4%}\n"
            f"Market Impact: {self.market_impact:.4%}\n"
            f"Implementation Shortfall: {self.implementation_shortfall:.4%}\n"
            f"Algorithm: {self.algorithm}\n"
            f"Duration: {self.duration_seconds:.1f}s\n"
        )


class ExecutionAlgorithm(ABC):
    """
    Abstract base class for execution algorithms.

    Execution algorithms split large orders into smaller slices
    to minimize market impact and execution costs.

    Example:
        >>> algo = TWAPAlgorithm(
        ...     total_quantity=1000,
        ...     duration_minutes=60,
        ...     num_slices=12,
        ... )
        >>> while not algo.is_complete:
        ...     slice = algo.get_next_slice()
        ...     executed_price = execute_order(slice)
        ...     algo.record_execution(slice, executed_price)
    """

    def __init__(
        self,
        ticker: str,
        side: Literal["buy", "sell"],
        total_quantity: float,
        arrival_price: float,
    ) -> None:
        """
        Initialize execution algorithm.

        Args:
            ticker: Asset ticker
            side: Buy or sell
            total_quantity: Total quantity to execute
            arrival_price: Price at order arrival (benchmark)
        """
        self.ticker = ticker
        self.side = side
        self.total_quantity = total_quantity
        self.arrival_price = arrival_price

        self.slices: list[OrderSlice] = []
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

        self._executed_quantity: float = 0.0
        self._total_value: float = 0.0

    @abstractmethod
    def generate_schedule(self) -> list[OrderSlice]:
        """Generate execution schedule (list of slices)."""
        pass

    @abstractmethod
    def get_next_slice(self, current_time: datetime) -> OrderSlice | None:
        """Get next slice to execute."""
        pass

    def record_execution(
        self,
        slice_index: int,
        executed_quantity: float,
        executed_price: float,
        executed_time: datetime,
    ) -> None:
        """Record execution of a slice."""
        if slice_index < len(self.slices):
            slice = self.slices[slice_index]
            slice.executed_quantity = executed_quantity
            slice.executed_price = executed_price
            slice.executed_time = executed_time
            slice.is_executed = True

            self._executed_quantity += executed_quantity
            self._total_value += executed_quantity * executed_price

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self._executed_quantity >= self.total_quantity * 0.99

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to execute."""
        return max(0, self.total_quantity - self._executed_quantity)

    @property
    def average_price(self) -> float:
        """Calculate average execution price."""
        if self._executed_quantity == 0:
            return 0
        return self._total_value / self._executed_quantity

    def get_result(self) -> ExecutionResult:
        """Get execution result."""
        avg_price = self.average_price
        slippage = 0.0
        if self.arrival_price > 0:
            if self.side == "buy":
                slippage = (avg_price - self.arrival_price) / self.arrival_price
            else:
                slippage = (self.arrival_price - avg_price) / self.arrival_price

        duration = 0.0
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return ExecutionResult(
            ticker=self.ticker,
            side=self.side,
            total_quantity=self.total_quantity,
            target_price=self.arrival_price,
            executed_quantity=self._executed_quantity,
            average_price=avg_price,
            total_cost=self._total_value,
            slippage=slippage,
            slices=self.slices,
            algorithm=self.__class__.__name__,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=duration,
        )


__all__ = [
    "ExecutionAlgorithm",
    "OrderSlice",
    "ExecutionResult",
]
