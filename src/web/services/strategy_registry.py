"""Strategy registry service.

Discover automatically registered strategies and provide metadata.
Includes VBO strategy variants.
"""

import inspect
from importlib import import_module
from typing import Any

from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.web.services.parameter_models import ParameterSpec, StrategyInfo

logger = get_logger(__name__)

__all__ = [
    "StrategyRegistry",
    "is_vbo_strategy",
    "get_vbo_strategy_type",
    "map_strategy_to_internal_type",
    "create_analysis_strategy",
]

# VBO strategy name → (engine_type, display_name)
VBO_STRATEGY_MAPPING: dict[str, tuple[str, str]] = {
    "VBO": ("vbo", "VBO"),
}


def is_vbo_strategy(name: str) -> bool:
    """Check if strategy is a VBO variant (uses vectorized engine)."""
    return name in VBO_STRATEGY_MAPPING


def get_vbo_strategy_type(name: str) -> tuple[str, str]:
    """Return (engine_type, display_name) for a VBO strategy."""
    return VBO_STRATEGY_MAPPING.get(name, ("vbo", "VBO"))


def map_strategy_to_internal_type(strategy_name: str) -> str:
    """Map registered strategy name to internal engine type.

    Used by analysis page for Monte Carlo / Walk-Forward.
    Returns the lowercased canonical strategy name for internal routing.
    """
    from src.strategies.registry import registry

    canonical = strategy_name.upper()
    if canonical in registry:
        return canonical.lower()
    return strategy_name.lower()


def create_analysis_strategy(strategy_type: str, **kwargs: Any) -> Strategy:
    """Create strategy instance for Monte Carlo / Walk-Forward analysis.

    Looks up the strategy in the central registry by name.
    Falls back to VBO if the name is not recognised.

    Args:
        strategy_type: Strategy name (e.g. "VBO", "vbo", "vbov1")
        **kwargs: Optional constructor overrides (e.g. noise_ratio=0.3)
    """
    from src.strategies.registry import registry

    # Normalise: registry uses upper-case canonical names (e.g. "VBO")
    canonical = strategy_type.upper()
    if canonical in registry:
        return registry.create(canonical, **kwargs)

    # Legacy fallback — keeps existing behaviour for "vbov1" / "vbo" inputs
    logger.warning(f"Strategy '{strategy_type}' not in registry; defaulting to VBO")
    return registry.create("VBO", **kwargs)


class StrategyRegistry:
    """Strategy auto-detection and registry.

    Scan all strategy modules to discover Strategy subclasses,
    and generate metadata by extracting parameters from __init__ signature.

    Example:
        >>> registry = StrategyRegistry()
        >>> strategies = registry.list_strategies()
        >>> for info in strategies:
        ...     print(f"{info.name}: {info.description}")
        >>>
        >>> params = registry.get_parameters("VBOV1")
        >>> strategy_class = registry.get_strategy_class("VBOV1")
    """

    STRATEGY_MODULES = [
        "src.strategies.volatility_breakout",
    ]

    def __init__(self) -> None:
        """Initialize registry and discover strategies."""
        self._strategies: dict[str, StrategyInfo] = {}
        self._discover_strategies()
        self._register_vbo_strategies()

    def _discover_strategies(self) -> None:
        """Discover Strategy subclasses from all strategy modules."""
        for module_path in self.STRATEGY_MODULES:
            try:
                module = import_module(module_path)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_valid_strategy(obj):
                        self._register_strategy(name, obj, module_path)
            except ImportError as e:
                logger.warning(f"Failed to import module {module_path}: {e}")

    def _is_valid_strategy(self, cls: type) -> bool:
        """Check if it's a valid strategy class."""
        return issubclass(cls, Strategy) and cls is not Strategy and not inspect.isabstract(cls)

    def _register_strategy(self, name: str, cls: type, module_path: str) -> None:
        """Register strategy to the registry."""
        try:
            parameters = self._extract_parameters(cls)
            description = self._extract_description(cls)

            info = StrategyInfo(
                name=name,
                class_name=cls.__name__,
                module_path=module_path,
                strategy_class=cls,
                parameters=parameters,
                description=description,
            )

            self._strategies[name] = info
            logger.debug(f"Registered strategy: {name} with {len(parameters)} parameters")

        except Exception as e:
            logger.warning(f"Failed to register strategy {name}: {e}")

    def _extract_parameters(self, cls: type) -> dict[str, ParameterSpec]:
        """Extract parameters from __init__ signature."""
        sig = inspect.signature(cls.__init__)  # type: ignore[misc]
        params: dict[str, ParameterSpec] = {}

        for name, param in sig.parameters.items():
            if name in ("self", "name", "entry_conditions", "exit_conditions"):
                continue

            spec = self._create_parameter_spec(name, param)
            if spec:
                params[name] = spec

        return params

    def _create_parameter_spec(self, name: str, param: inspect.Parameter) -> ParameterSpec | None:
        """Create ParameterSpec from parameter information."""
        annotation = param.annotation
        default = param.default if param.default != inspect.Parameter.empty else None

        # Skip if default is None
        if default is None:
            return None

        # Infer type
        param_type = self._infer_type(annotation, default)
        if param_type is None:
            return None

        # Create spec by type with reasonable bounds based on default
        if param_type == "int":
            int_default = int(default)
            return ParameterSpec(
                name=name,
                type="int",
                default=default,
                min_value=1,
                max_value=max(100, int_default * 2),
                step=1,
                description=f"Integer parameter: {name}",
            )
        elif param_type == "float":
            float_default = float(default)
            return ParameterSpec(
                name=name,
                type="float",
                default=default,
                min_value=0.0,
                max_value=max(1.0, float_default * 2),
                step=0.01,
                description=f"Float parameter: {name}",
            )
        elif param_type == "bool":
            return ParameterSpec(
                name=name,
                type="bool",
                default=default,
                description=f"Boolean parameter: {name}",
            )

        return None

    def _infer_type(self, annotation: Any, default: Any) -> str | None:
        """Infer parameter type from type hint and default value."""
        # Check type hint
        if annotation != inspect.Parameter.empty:
            if annotation is int or annotation == "int":
                return "int"
            elif annotation is float or annotation == "float":
                return "float"
            elif annotation is bool or annotation == "bool":
                return "bool"

        # Infer from default value
        if isinstance(default, bool):
            return "bool"
        elif isinstance(default, int):
            return "int"
        elif isinstance(default, float):
            return "float"

        return None

    def _extract_description(self, cls: type) -> str:
        """Extract description from class docstring."""
        doc = inspect.getdoc(cls)
        if doc:
            # Use only first line
            return doc.split("\n")[0].strip()
        return f"{cls.__name__} strategy"

    def list_strategies(self) -> list[StrategyInfo]:
        """Return list of all registered strategies."""
        return list(self._strategies.values())

    def get_strategy(self, name: str) -> StrategyInfo | None:
        """Get StrategyInfo by strategy name."""
        return self._strategies.get(name)

    def get_strategy_class(self, name: str) -> type | None:
        """Get class by strategy name."""
        info = self._strategies.get(name)
        return info.strategy_class if info else None

    def get_parameters(self, name: str) -> dict[str, ParameterSpec]:
        """Get parameter spec by strategy name."""
        info = self._strategies.get(name)
        return info.parameters if info else {}

    def strategy_exists(self, name: str) -> bool:
        """Check if strategy is registered."""
        return name in self._strategies

    def _register_vbo_strategies(self) -> None:
        """Register VBO strategy variants.

        These use predefined parameter sets rather than auto-discovery.
        """
        vbo_strategy_defs: list[dict[str, Any]] = [
            {
                "name": "VBO",
                "module_path": "src.strategies.volatility_breakout",
                "description": "Volatility Breakout Strategy (BTC MA20 filter)",
                "parameters": {
                    "lookback": ParameterSpec(
                        name="lookback",
                        type="int",
                        default=5,
                        min_value=2,
                        max_value=20,
                        step=1,
                        description="Short-term MA period (lookback period)",
                    ),
                    "multiplier": ParameterSpec(
                        name="multiplier",
                        type="int",
                        default=2,
                        min_value=1,
                        max_value=5,
                        step=1,
                        description="Multiplier for long-term MA (Long MA = lookback * multiplier)",
                    ),
                },
            },
        ]

        for defn in vbo_strategy_defs:
            info = StrategyInfo(
                name=defn["name"],
                class_name=defn["name"],
                module_path=defn["module_path"],
                strategy_class=None,
                parameters=defn["parameters"],
                description=defn["description"],
            )
            self._strategies[defn["name"]] = info

        logger.info("Registered VBO strategies: " + ", ".join(d["name"] for d in vbo_strategy_defs))
