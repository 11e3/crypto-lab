"""Tests for parameter and strategy metadata models."""

from __future__ import annotations

import pytest

from src.web.services.parameter_models import ParameterSpec, StrategyInfo


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_int_param(self) -> None:
        spec = ParameterSpec(
            name="period",
            type="int",
            default=14,
            min_value=5,
            max_value=50,
            step=1,
            description="SMA period",
        )
        assert spec.name == "period"
        assert spec.type == "int"
        assert spec.default == 14
        assert spec.min_value == 5
        assert spec.max_value == 50

    def test_float_param(self) -> None:
        spec = ParameterSpec(
            name="k_value",
            type="float",
            default=0.5,
            min_value=0.1,
            max_value=2.0,
            step=0.1,
        )
        assert spec.type == "float"
        assert spec.default == 0.5

    def test_bool_param(self) -> None:
        spec = ParameterSpec(name="use_adaptive", type="bool", default=True)
        assert spec.type == "bool"
        assert spec.default is True
        assert spec.min_value is None

    def test_choice_param(self) -> None:
        spec = ParameterSpec(
            name="method",
            type="choice",
            default="sma",
            choices=["sma", "ema", "wma"],
        )
        assert spec.type == "choice"
        assert spec.choices == ["sma", "ema", "wma"]

    def test_frozen(self) -> None:
        spec = ParameterSpec(name="x", type="int", default=10)
        with pytest.raises(AttributeError):
            spec.name = "y"  # type: ignore[misc]

    def test_default_description_empty(self) -> None:
        spec = ParameterSpec(name="x", type="int", default=0)
        assert spec.description == ""


class TestStrategyInfo:
    """Tests for StrategyInfo dataclass."""

    def test_basic_creation(self) -> None:
        params = {
            "period": ParameterSpec(name="period", type="int", default=14),
        }
        info = StrategyInfo(
            name="VBO Strategy",
            class_name="VBOStrategy",
            module_path="src.strategies.vbo",
            strategy_class=None,
            parameters=params,
            description="Volatility breakout strategy.",
        )
        assert info.name == "VBO Strategy"
        assert info.class_name == "VBOStrategy"
        assert "period" in info.parameters

    def test_frozen(self) -> None:
        info = StrategyInfo(
            name="Test",
            class_name="Test",
            module_path="test",
            strategy_class=None,
            parameters={},
            description="",
        )
        with pytest.raises(AttributeError):
            info.name = "Modified"  # type: ignore[misc]

    def test_empty_parameters(self) -> None:
        info = StrategyInfo(
            name="Simple",
            class_name="SimpleStrategy",
            module_path="src.strategies.simple",
            strategy_class=None,
            parameters={},
            description="No params strategy.",
        )
        assert info.parameters == {}
