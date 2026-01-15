"""Strategy registry service.

자동으로 등록된 전략들을 탐색하고 메타데이터를 제공합니다.
"""

import inspect
from importlib import import_module
from typing import Any

from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.web.services.parameter_models import ParameterSpec, StrategyInfo

logger = get_logger(__name__)

__all__ = ["StrategyRegistry"]


class StrategyRegistry:
    """전략 자동 감지 및 레지스트리.

    모든 전략 모듈을 스캔하여 Strategy 서브클래스를 탐색하고,
    __init__ 시그니처에서 파라미터를 추출하여 메타데이터를 생성합니다.

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
        "src.strategies.pair_trading",
        "src.strategies.opening_range_breakout",
    ]

    def __init__(self) -> None:
        """Initialize registry and discover strategies."""
        self._strategies: dict[str, StrategyInfo] = {}
        self._discover_strategies()

    def _discover_strategies(self) -> None:
        """모든 전략 모듈에서 Strategy 서브클래스 탐색."""
        for module_path in self.STRATEGY_MODULES:
            try:
                module = import_module(module_path)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_valid_strategy(obj):
                        self._register_strategy(name, obj, module_path)
            except ImportError as e:
                logger.warning(f"Failed to import module {module_path}: {e}")

    def _is_valid_strategy(self, cls: type) -> bool:
        """유효한 전략 클래스인지 확인."""
        return issubclass(cls, Strategy) and cls is not Strategy and not inspect.isabstract(cls)

    def _register_strategy(self, name: str, cls: type, module_path: str) -> None:
        """전략을 레지스트리에 등록."""
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
        """__init__ 시그니처에서 파라미터 추출."""
        sig = inspect.signature(cls.__init__)
        params: dict[str, ParameterSpec] = {}

        for name, param in sig.parameters.items():
            if name in ("self", "name", "entry_conditions", "exit_conditions"):
                continue

            spec = self._create_parameter_spec(name, param)
            if spec:
                params[name] = spec

        return params

    def _create_parameter_spec(self, name: str, param: inspect.Parameter) -> ParameterSpec | None:
        """파라미터 정보에서 ParameterSpec 생성."""
        annotation = param.annotation
        default = param.default if param.default != inspect.Parameter.empty else None

        # 기본값이 None이면 스킵
        if default is None:
            return None

        # 타입 추론
        param_type = self._infer_type(annotation, default)
        if param_type is None:
            return None

        # 타입별 스펙 생성
        if param_type == "int":
            return ParameterSpec(
                name=name,
                type="int",
                default=default,
                min_value=1,
                max_value=100,
                step=1,
                description=f"Integer parameter: {name}",
            )
        elif param_type == "float":
            return ParameterSpec(
                name=name,
                type="float",
                default=default,
                min_value=0.0,
                max_value=1.0,
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
        """타입 힌트와 기본값에서 파라미터 타입 추론."""
        # 타입 힌트 체크
        if annotation != inspect.Parameter.empty:
            if annotation is int or annotation == "int":
                return "int"
            elif annotation is float or annotation == "float":
                return "float"
            elif annotation is bool or annotation == "bool":
                return "bool"

        # 기본값으로 추론
        if isinstance(default, bool):
            return "bool"
        elif isinstance(default, int):
            return "int"
        elif isinstance(default, float):
            return "float"

        return None

    def _extract_description(self, cls: type) -> str:
        """클래스 docstring에서 설명 추출."""
        doc = inspect.getdoc(cls)
        if doc:
            # 첫 번째 줄만 사용
            return doc.split("\n")[0].strip()
        return f"{cls.__name__} strategy"

    def list_strategies(self) -> list[StrategyInfo]:
        """등록된 모든 전략 목록 반환."""
        return list(self._strategies.values())

    def get_strategy(self, name: str) -> StrategyInfo | None:
        """전략 이름으로 StrategyInfo 조회."""
        return self._strategies.get(name)

    def get_strategy_class(self, name: str) -> type | None:
        """전략 이름으로 클래스 조회."""
        info = self._strategies.get(name)
        return info.strategy_class if info else None

    def get_parameters(self, name: str) -> dict[str, ParameterSpec]:
        """전략 이름으로 파라미터 스펙 조회."""
        info = self._strategies.get(name)
        return info.parameters if info else {}

    def strategy_exists(self, name: str) -> bool:
        """전략이 등록되어 있는지 확인."""
        return name in self._strategies
