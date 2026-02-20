"""Strategy registry — factory pattern for creating strategies by name.

Central registry that maps strategy names to their class implementations.
Used by the web layer to create strategy instances without hardcoding imports.

Usage::

    # Register a strategy (done in strategies/__init__.py)
    from src.strategies.registry import registry
    registry.register("VBO", VBOV1)

    # Create an instance by name
    strategy = registry.create("VBO", noise_ratio=0.5)

    # List all registered strategies
    names = registry.list_names()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.strategies.base import Strategy

logger = get_logger(__name__)

__all__ = ["StrategyFactory", "registry"]


class StrategyFactory:
    """Registry that maps strategy names to Strategy subclasses.

    Thread-safe for read operations. Registrations should happen at import time.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[Strategy]] = {}

    def register(self, name: str, cls: type[Strategy]) -> None:
        """Register a strategy class under the given name.

        Args:
            name: Canonical name (e.g., "VBO", "MeanReversion")
            cls: Strategy subclass to register
        """
        if name in self._registry:
            logger.warning(f"Strategy '{name}' already registered; overwriting")
        self._registry[name] = cls
        logger.debug(f"Registered strategy: {name} → {cls.__name__}")

    def create(self, strategy_name: str, **kwargs: Any) -> Strategy:
        """Create a strategy instance by name.

        Args:
            strategy_name: Registered strategy name (e.g. "VBO")
            **kwargs: Constructor arguments forwarded to the strategy class

        Returns:
            Strategy instance

        Raises:
            ValueError: If the name is not registered
        """
        if strategy_name not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(none)"
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
        cls = self._registry[strategy_name]
        return cls(**kwargs)

    def list_names(self) -> list[str]:
        """Return sorted list of registered strategy names."""
        return sorted(self._registry)

    def get_class(self, name: str) -> type[Strategy] | None:
        """Return the strategy class for the given name, or None if not found."""
        return self._registry.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Module-level singleton — populated in src/strategies/__init__.py
registry = StrategyFactory()
