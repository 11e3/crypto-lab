"""Tests for strategy selector component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.web.components.sidebar.strategy_selector import (
    _render_parameter_input,
    create_strategy_instance,
)
from src.web.services.parameter_models import ParameterSpec

# =========================================================================
# _render_parameter_input
# =========================================================================


class TestRenderParameterInput:
    """Tests for _render_parameter_input."""

    @patch("src.web.components.sidebar.strategy_selector.st")
    def test_int_parameter(self, mock_st: MagicMock) -> None:
        mock_st.slider.return_value = 20
        spec = ParameterSpec(
            name="window",
            type="int",
            default=20,
            min_value=1,
            max_value=100,
            step=1,
            description="Window size",
        )
        result = _render_parameter_input("window", spec)
        assert result == 20
        mock_st.slider.assert_called_once()

    @patch("src.web.components.sidebar.strategy_selector.st")
    def test_int_parameter_no_max(self, mock_st: MagicMock) -> None:
        """When max_value is None, should auto-calculate."""
        mock_st.slider.return_value = 50
        spec = ParameterSpec(
            name="window",
            type="int",
            default=50,
            min_value=1,
            max_value=None,
            step=1,
        )
        result = _render_parameter_input("window", spec)
        assert result == 50

    @patch("src.web.components.sidebar.strategy_selector.st")
    def test_float_parameter(self, mock_st: MagicMock) -> None:
        mock_st.number_input.return_value = 0.02
        spec = ParameterSpec(
            name="fee_rate",
            type="float",
            default=0.02,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Fee rate",
        )
        result = _render_parameter_input("fee_rate", spec)
        assert result == 0.02
        mock_st.number_input.assert_called_once()

    @patch("src.web.components.sidebar.strategy_selector.st")
    def test_float_parameter_no_max(self, mock_st: MagicMock) -> None:
        mock_st.number_input.return_value = 0.5
        spec = ParameterSpec(
            name="threshold",
            type="float",
            default=0.5,
            min_value=0.0,
            max_value=None,
            step=0.01,
        )
        result = _render_parameter_input("threshold", spec)
        assert result == 0.5

    @patch("src.web.components.sidebar.strategy_selector.st")
    def test_bool_parameter(self, mock_st: MagicMock) -> None:
        mock_st.checkbox.return_value = True
        spec = ParameterSpec(
            name="enable_filter",
            type="bool",
            default=True,
            description="Enable filter",
        )
        result = _render_parameter_input("enable_filter", spec)
        assert result is True
        mock_st.checkbox.assert_called_once()

    @patch("src.web.components.sidebar.strategy_selector.st")
    def test_choice_parameter(self, mock_st: MagicMock) -> None:
        mock_st.selectbox.return_value = "sma"
        spec = ParameterSpec(
            name="ma_type",
            type="choice",
            default="sma",
            choices=["sma", "ema", "wma"],
            description="Moving average type",
        )
        result = _render_parameter_input("ma_type", spec)
        assert result == "sma"
        mock_st.selectbox.assert_called_once()

    @patch("src.web.components.sidebar.strategy_selector.st")
    def test_unknown_type_returns_default(self, mock_st: MagicMock) -> None:
        """Unknown parameter types fall back to returning the default value."""
        # Use MagicMock to bypass frozen dataclass Literal restriction
        spec = MagicMock()
        spec.type = "unknown_type"
        spec.default = 42
        result = _render_parameter_input("mystery", spec)
        assert result == 42


# =========================================================================
# create_strategy_instance
# =========================================================================


class TestCreateStrategyInstance:
    """Tests for create_strategy_instance."""

    @patch("src.web.components.sidebar.strategy_selector.get_cached_registry")
    def test_creates_instance(self, mock_registry_fn: MagicMock) -> None:
        mock_registry = MagicMock()
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        mock_registry.get_strategy_class.return_value = mock_class
        mock_registry_fn.return_value = mock_registry

        result = create_strategy_instance("VBO", {"window": 20})
        assert result is not None
        mock_class.assert_called_once_with(window=20)

    @patch("src.web.components.sidebar.strategy_selector.get_cached_registry")
    def test_returns_none_when_class_not_found(self, mock_registry_fn: MagicMock) -> None:
        mock_registry = MagicMock()
        mock_registry.get_strategy_class.return_value = None
        mock_registry_fn.return_value = mock_registry

        result = create_strategy_instance("NonExistent", {})
        assert result is None

    @patch("src.web.components.sidebar.strategy_selector.get_cached_registry")
    def test_returns_none_on_exception(self, mock_registry_fn: MagicMock) -> None:
        mock_registry = MagicMock()
        mock_class = MagicMock(side_effect=TypeError("bad args"))
        mock_registry.get_strategy_class.return_value = mock_class
        mock_registry_fn.return_value = mock_registry

        result = create_strategy_instance("VBO", {"bad_param": 999})
        assert result is None
