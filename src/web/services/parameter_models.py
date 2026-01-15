"""Parameter and strategy metadata models."""

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "ParameterSpec",
    "StrategyInfo",
]


@dataclass(frozen=True)
class ParameterSpec:
    """전략 파라미터 명세.

    Attributes:
        name: 파라미터 이름
        type: 파라미터 타입 (int, float, bool, choice)
        default: 기본값
        min_value: 최소값 (int, float 타입일 때)
        max_value: 최대값 (int, float 타입일 때)
        step: 스텝 크기 (슬라이더에 사용)
        choices: 선택 가능한 값 목록 (choice 타입일 때)
        description: 파라미터 설명
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
    """전략 메타데이터.

    Attributes:
        name: 전략 이름 (UI에 표시될 이름)
        class_name: 전략 클래스 이름
        module_path: 모듈 경로 (예: src.strategies.volatility_breakout)
        strategy_class: 전략 클래스 타입
        parameters: 파라미터 명세 딕셔너리 {파라미터명: ParameterSpec}
        description: 전략 설명
    """

    name: str
    class_name: str
    module_path: str
    strategy_class: type
    parameters: dict[str, ParameterSpec]
    description: str
