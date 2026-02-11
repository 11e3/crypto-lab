"""Strategy registry service.

Discover automatically registered strategies and provide metadata.
Includes bt library strategies integration.
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
    "is_bt_strategy",
    "get_bt_strategy_type",
    "map_strategy_to_internal_type",
    "create_analysis_strategy",
]

# bt strategy name → (engine_type, display_name)
BT_STRATEGY_MAPPING: dict[str, tuple[str, str]] = {
    "bt_VBO": ("vbo", "bt VBO"),
    "bt_VBO_Regime": ("vbo_regime", "bt VBO Regime"),
    "bt_Momentum": ("momentum", "bt Momentum"),
    "bt_BuyAndHold": ("buy_and_hold", "bt Buy & Hold"),
    "bt_VBO_SingleCoin": ("vbo_single_coin", "bt VBO Single Coin"),
    "bt_VBO_Portfolio": ("vbo_portfolio", "bt VBO Portfolio"),
}


def is_bt_strategy(name: str) -> bool:
    """Check if strategy is from bt library."""
    return name.startswith("bt_")


def get_bt_strategy_type(name: str) -> tuple[str, str]:
    """Return (bt_engine_type, display_name) for a bt strategy."""
    return BT_STRATEGY_MAPPING.get(name, ("vbo", "bt VBO"))


def map_strategy_to_internal_type(strategy_name: str) -> str:
    """Map registered strategy name to internal engine type.

    Used by analysis page for Monte Carlo / Walk-Forward.
    """
    name_lower = strategy_name.lower()

    if "vanilla" in name_lower or "vbo" in name_lower:
        if "legacy" in name_lower:
            return "legacy"
        elif "minimal" in name_lower:
            return "minimal"
        return "vanilla"
    elif "momentum" in name_lower:
        return "momentum"
    elif "mean" in name_lower and "reversion" in name_lower:
        return "mean-reversion"
    return "vanilla"


def create_analysis_strategy(strategy_type: str) -> Any:
    """Create strategy instance for Monte Carlo / Walk-Forward analysis."""
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.strategies.momentum import MomentumStrategy
    from src.strategies.volatility_breakout import create_vbo_strategy

    if strategy_type in ("vanilla", "minimal"):
        return create_vbo_strategy(
            name="VanillaVBO",
            use_trend_filter=False,
            use_noise_filter=False,
        )
    elif strategy_type == "legacy":
        return create_vbo_strategy(
            name="LegacyVBO",
            use_trend_filter=True,
            use_noise_filter=True,
        )
    elif strategy_type == "momentum":
        return MomentumStrategy(name="Momentum")
    elif strategy_type == "mean-reversion":
        return MeanReversionStrategy(name="MeanReversion")
    return create_vbo_strategy(name="DefaultVBO")


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
        >>> params = registry.get_parameters("VanillaVBO")
        >>> strategy_class = registry.get_strategy_class("VanillaVBO")
    """

    STRATEGY_MODULES = [
        "src.strategies.volatility_breakout",
        "src.strategies.momentum",
        "src.strategies.mean_reversion",
        "src.strategies.opening_range_breakout",
    ]

    def __init__(self) -> None:
        """Initialize registry and discover strategies."""
        self._strategies: dict[str, StrategyInfo] = {}
        self._discover_strategies()
        self._register_bt_strategies()

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

    def _register_bt_strategies(self) -> None:
        """Register bt library strategies.

        Note: We always register bt strategies without checking availability.
        The actual availability check happens when running a backtest.
        This avoids slow bt library import during app startup.
        """
        # Shared parameter specs reused across strategies
        _ma_short = ParameterSpec(
            name="ma_short", type="int", default=5,
            min_value=2, max_value=20, step=1,
            description="Short-term MA period",
        )
        _btc_ma = ParameterSpec(
            name="btc_ma", type="int", default=20,
            min_value=5, max_value=60, step=5,
            description="BTC MA period for market filter",
        )
        _noise_ratio = ParameterSpec(
            name="noise_ratio", type="float", default=0.5,
            min_value=0.1, max_value=1.0, step=0.1,
            description="Volatility breakout multiplier (k factor)",
        )

        bt_strategy_defs: list[dict[str, Any]] = [
            {
                "name": "bt_VBO",
                "module_path": "bt.strategies.vbo",
                "description": "[bt] Volatility Breakout Strategy (BTC MA20 filter)",
                "parameters": {
                    "lookback": ParameterSpec(
                        name="lookback", type="int", default=5,
                        min_value=2, max_value=20, step=1,
                        description="Short-term MA period (lookback period)",
                    ),
                    "multiplier": ParameterSpec(
                        name="multiplier", type="int", default=2,
                        min_value=1, max_value=5, step=1,
                        description="Multiplier for long-term MA (Long MA = lookback * multiplier)",
                    ),
                },
            },
            {
                "name": "bt_VBO_Regime",
                "module_path": "bt.strategies.vbo_regime",
                "description": "[bt] Volatility Breakout Strategy (ML Regime filter)",
                "parameters": {"ma_short": _ma_short, "noise_ratio": _noise_ratio},
            },
            {
                "name": "bt_Momentum",
                "module_path": "bt.strategies.momentum",
                "description": "[bt] Pure Momentum Strategy (equal-weight allocation)",
                "parameters": {
                    "lookback": ParameterSpec(
                        name="lookback", type="int", default=20,
                        min_value=5, max_value=60, step=5,
                        description="Momentum lookback period",
                    ),
                },
            },
            {
                "name": "bt_BuyAndHold",
                "module_path": "bt.strategies.buy_and_hold",
                "description": "[bt] Simple Buy and Hold Strategy",
                "parameters": {},
            },
            {
                "name": "bt_VBO_SingleCoin",
                "module_path": "bt.strategies.vbo_single_coin",
                "description": "[bt] Single-asset VBO Strategy (BTC MA filter, all-in allocation)",
                "parameters": {"ma_short": _ma_short, "btc_ma": _btc_ma, "noise_ratio": _noise_ratio},
            },
            {
                "name": "bt_VBO_Portfolio",
                "module_path": "bt.strategies.vbo_portfolio",
                "description": "[bt] Multi-asset VBO Strategy (BTC MA filter, 1/N allocation)",
                "parameters": {"ma_short": _ma_short, "btc_ma": _btc_ma, "noise_ratio": _noise_ratio},
            },
        ]

        try:
            for defn in bt_strategy_defs:
                info = StrategyInfo(
                    name=defn["name"],
                    class_name=defn["name"],
                    module_path=defn["module_path"],
                    strategy_class=None,
                    parameters=defn["parameters"],
                    description=defn["description"],
                )
                self._strategies[defn["name"]] = info

            logger.info(
                "Registered bt library strategies: "
                + ", ".join(d["name"] for d in bt_strategy_defs)
            )
        except Exception as e:
            logger.warning(f"Failed to register bt strategies: {e}")
