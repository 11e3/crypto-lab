"""
Base condition classes for strategy abstraction.

Contains condition abstractions for composable entry/exit logic:
- Condition: ABC for atomic conditions
- Filter: Alias for Condition (backward compatibility)
- CompositeCondition: Combines conditions with AND/OR logic
"""

import logging
from abc import ABC, abstractmethod

import pandas as pd

from src.strategies.base_models import OHLCV

logger = logging.getLogger(__name__)


class Condition(ABC):
    """
    Abstract base class for entry/exit conditions.

    Conditions are atomic logic units that evaluate market data
    and return True/False for signal generation.
    """

    def __init__(self, name: str | None = None) -> None:
        """
        Initialize condition.

        Args:
            name: Human-readable condition name
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool | pd.Series:
        """
        Evaluate condition against market data.

        Args:
            current: Current bar OHLCV data
            history: Historical DataFrame with OHLCV + indicators
            indicators: Pre-calculated indicator values for current bar

        Returns:
            True if condition is met
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# Alias for backward compatibility
# Filters are now treated as Conditions in the unified architecture
Filter = Condition


class CompositeCondition(Condition):
    """Combines multiple conditions with AND/OR logic."""

    def __init__(
        self,
        conditions: list[Condition],
        operator: str = "AND",
        name: str | None = None,
    ) -> None:
        """
        Initialize composite condition.

        Args:
            conditions: List of conditions to combine
            operator: "AND" or "OR"
            name: Human-readable name
        """
        super().__init__(name)
        self.conditions = conditions
        self.operator = operator.upper()

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Evaluate all conditions with specified operator."""
        if not self.conditions:
            return True

        raw_results = [c.evaluate(current, history, indicators) for c in self.conditions]
        results: list[bool] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, pd.Series):
                # Use the latest value when vectorized signals are returned
                if len(r) > 1:
                    logger.debug(
                        f"Condition '{self.conditions[i].name}' returned Series of length {len(r)}; using last value"
                    )
                results.append(bool(r.iloc[-1]))
            else:
                results.append(bool(r))

        if self.operator == "AND":
            return all(results)
        elif self.operator == "OR":
            return any(results)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def add(self, condition: Condition) -> "CompositeCondition":
        """Add a condition and return self for chaining."""
        self.conditions.append(condition)
        return self

    def remove(self, condition: Condition) -> "CompositeCondition":
        """Remove a condition and return self for chaining."""
        self.conditions.remove(condition)
        return self
