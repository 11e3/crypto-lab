"""Strategy selector component.

전략 선택 및 동적 파라미터 편집 UI 컴포넌트.
"""

from typing import Any

import streamlit as st

from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.web.services import ParameterSpec, StrategyRegistry

logger = get_logger(__name__)

__all__ = ["render_strategy_selector", "create_strategy_instance"]


@st.cache_resource
def get_cached_registry() -> StrategyRegistry:
    """캐시된 전략 레지스트리 반환."""
    return StrategyRegistry()


def render_strategy_selector() -> tuple[str, dict[str, Any]]:
    """전략 선택 및 파라미터 동적 렌더링.
    
    Returns:
        (strategy_name, parameters) 튜플
    """
    st.subheader("📈 전략 선택")

    registry = get_cached_registry()
    strategies = registry.list_strategies()

    if not strategies:
        st.error("⚠️ 등록된 전략이 없습니다.")
        return "", {}

    # 전략 이름 목록
    strategy_names = [s.name for s in strategies]

    # 전략 선택
    selected_name = st.selectbox(
        "전략",
        options=strategy_names,
        help="백테스트에 사용할 전략 선택",
    )

    # 선택된 전략 정보
    selected_strategy = registry.get_strategy(selected_name)
    if not selected_strategy:
        st.error(f"⚠️ 전략을 찾을 수 없습니다: {selected_name}")
        return "", {}

    # 전략 설명 표시
    if selected_strategy.description:
        st.caption(f"ℹ️ {selected_strategy.description}")

    st.markdown("---")

    # 파라미터 편집
    st.subheader("🎛️ 전략 파라미터")

    parameters = selected_strategy.parameters

    if not parameters:
        st.info("📌 이 전략은 설정 가능한 파라미터가 없습니다.")
        return selected_name, {}

    param_values: dict[str, Any] = {}

    # 파라미터 타입별로 UI 렌더링
    for name, spec in parameters.items():
        param_values[name] = _render_parameter_input(name, spec)

    return selected_name, param_values


def _render_parameter_input(name: str, spec: ParameterSpec) -> Any:
    """파라미터 타입에 따른 입력 UI 렌더링.
    
    Args:
        name: 파라미터 이름
        spec: 파라미터 스펙
        
    Returns:
        사용자 입력값
    """
    # 이름 포맷팅 (snake_case -> Title Case)
    label = name.replace("_", " ").title()

    match spec.type:
        case "int":
            return st.slider(
                label,
                min_value=int(spec.min_value or 1),
                max_value=int(spec.max_value or 100),
                value=int(spec.default),
                step=int(spec.step or 1),
                help=spec.description or f"정수형 파라미터: {name}",
            )

        case "float":
            return st.number_input(
                label,
                min_value=float(spec.min_value or 0.0),
                max_value=float(spec.max_value or 1.0),
                value=float(spec.default),
                step=float(spec.step or 0.01),
                format="%.4f",
                help=spec.description or f"실수형 파라미터: {name}",
            )

        case "bool":
            return st.checkbox(
                label,
                value=bool(spec.default),
                help=spec.description or f"불리언 파라미터: {name}",
            )

        case "choice":
            choices = spec.choices or []
            default_index = (
                choices.index(spec.default) if spec.default in choices else 0
            )
            return st.selectbox(
                label,
                options=choices,
                index=default_index,
                help=spec.description or f"선택 파라미터: {name}",
            )

        case _:
            logger.warning(f"Unknown parameter type: {spec.type}")
            return spec.default


def create_strategy_instance(
    strategy_name: str, parameters: dict[str, Any]
) -> Strategy | None:
    """전략 인스턴스 생성.
    
    Args:
        strategy_name: 전략 이름
        parameters: 파라미터 딕셔너리
        
    Returns:
        Strategy 인스턴스 또는 None (실패 시)
    """
    try:
        registry = get_cached_registry()
        strategy_class = registry.get_strategy_class(strategy_name)

        if not strategy_class:
            logger.error(f"Strategy class not found: {strategy_name}")
            return None

        # 전략 인스턴스 생성
        strategy = strategy_class(**parameters)
        logger.info(f"Created strategy: {strategy_name} with params: {parameters}")
        return strategy

    except Exception as e:
        logger.exception(f"Failed to create strategy {strategy_name}: {e}")
        return None
