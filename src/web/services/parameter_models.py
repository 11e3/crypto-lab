"""Parameter and strategy metadata models."""

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "ParameterSpec",
    "StrategyInfo",
]


@dataclass(frozen=True)
class ParameterSpec:
    """Strategy parameter specification.

    Attributes:
        name: Parameter name
        type: Parameter type (int, float, bool, choice)
        default: Default value
        min_value: Minimum value (for int, float types)
        max_value: Maximum value (for int, float types)
        step: Step size (used for slider)
        choices: List of available choices (for choice type)
        description: Parameter description
    """

    name: str
    type: Literal["int", "float", "bool", "choice"]
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    choices: list[Any] | None = None
    description: str = ""


@dataclass(frozen=True)
class StrategyInfo:
    """Strategy metadata.

    Attributes:
        name: Strategy name (name to display in UI)
        class_name: Strategy class name
        module_path: Module path (e.g., src.strategies.volatility_breakout)
        strategy_class: Strategy class type
        parameters: Parameter specification dictionary {param_name: ParameterSpec}
        description: Strategy description
    """

    name: str
    class_name: str
    module_path: str
    strategy_class: type | None
    parameters: dict[str, ParameterSpec]
    description: str
