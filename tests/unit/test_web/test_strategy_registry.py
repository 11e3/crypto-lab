"""Tests for strategy registry service.

Tests:
- is_vbo_strategy / get_vbo_strategy_type helper functions
- map_strategy_to_internal_type mapping logic
- create_analysis_strategy factory
- StrategyRegistry discovery, registration, and query
"""

from __future__ import annotations

import pytest

from src.web.services.parameter_models import ParameterSpec, StrategyInfo
from src.web.services.strategy_registry import (
    StrategyRegistry,
    create_analysis_strategy,
    get_vbo_strategy_type,
    is_vbo_strategy,
    map_strategy_to_internal_type,
)

# ── Helper functions ──


class TestIsVboStrategy:
    """Test VBO strategy name detection."""

    def test_vbo_name(self) -> None:
        assert is_vbo_strategy("VBO") is True

    def test_non_vbo(self) -> None:
        assert is_vbo_strategy("VBOV1") is False
        assert is_vbo_strategy("SomeStrategy") is False

    def test_empty_string(self) -> None:
        assert is_vbo_strategy("") is False


class TestGetVboStrategyType:
    """Test VBO strategy type mapping."""

    def test_known_strategies(self) -> None:
        engine, display = get_vbo_strategy_type("VBO")
        assert engine == "vbo"
        assert display == "VBO"

    def test_unknown_defaults_to_vbo(self) -> None:
        engine, display = get_vbo_strategy_type("Unknown")
        assert engine == "vbo"
        assert display == "VBO"


class TestMapStrategyToInternalType:
    """Test strategy name to internal type mapping."""

    def test_registered_strategy_maps_to_lowercase(self) -> None:
        # VBO is in the registry → returns "vbo"
        assert map_strategy_to_internal_type("VBO") == "vbo"

    def test_unregistered_strategy_maps_to_lowercase_input(self) -> None:
        # Unknown names → lowercased input (not hardcoded "vbov1" any more)
        assert map_strategy_to_internal_type("SomeOther") == "someother"
        assert map_strategy_to_internal_type("VBOV1") == "vbov1"


# ── create_analysis_strategy ──


class TestCreateAnalysisStrategy:
    """Test analysis strategy factory."""

    def test_any_type_creates_vbov1(self) -> None:
        strategy = create_analysis_strategy("vanilla")
        assert strategy is not None
        assert strategy.name == "VBOV1"

    def test_unknown_creates_vbov1(self) -> None:
        strategy = create_analysis_strategy("unknown_type")
        assert strategy is not None
        assert strategy.name == "VBOV1"


# ── StrategyRegistry ──


class TestStrategyRegistry:
    """Test strategy registry auto-discovery."""

    @pytest.fixture
    def registry(self) -> StrategyRegistry:
        return StrategyRegistry()

    def test_discovers_strategies(self, registry: StrategyRegistry) -> None:
        strategies = registry.list_strategies()
        assert len(strategies) > 0

    def test_contains_vbov1(self, registry: StrategyRegistry) -> None:
        names = [s.name for s in registry.list_strategies()]
        assert "VBOV1" in names

    def test_contains_vbo_strategies(self, registry: StrategyRegistry) -> None:
        names = [s.name for s in registry.list_strategies()]
        assert "VBO" in names

    def test_get_strategy(self, registry: StrategyRegistry) -> None:
        info = registry.get_strategy("VBO")
        assert info is not None
        assert info.name == "VBO"
        assert info.description != ""

    def test_get_strategy_nonexistent(self, registry: StrategyRegistry) -> None:
        assert registry.get_strategy("nonexistent") is None

    def test_get_strategy_class_vbo(self, registry: StrategyRegistry) -> None:
        # VBO strategies registered manually have None strategy_class
        cls = registry.get_strategy_class("VBO")
        assert cls is None

    def test_get_parameters(self, registry: StrategyRegistry) -> None:
        params = registry.get_parameters("VBO")
        assert "lookback" in params
        assert "multiplier" in params
        assert isinstance(params["lookback"], ParameterSpec)

    def test_get_parameters_empty(self, registry: StrategyRegistry) -> None:
        params = registry.get_parameters("nonexistent")
        assert params == {}

    def test_strategy_exists(self, registry: StrategyRegistry) -> None:
        assert registry.strategy_exists("VBO") is True
        assert registry.strategy_exists("nonexistent") is False

    def test_strategy_info_structure(self, registry: StrategyRegistry) -> None:
        info = registry.get_strategy("VBO")
        assert info is not None
        assert isinstance(info, StrategyInfo)
        assert isinstance(info.parameters, dict)
        assert isinstance(info.description, str)

    def test_native_strategy_has_class(self, registry: StrategyRegistry) -> None:
        """Native strategies should have a strategy_class."""
        strategies = registry.list_strategies()
        native = [s for s in strategies if s.strategy_class is not None]
        assert len(native) >= 1

    def test_parameter_extraction_from_native(self, registry: StrategyRegistry) -> None:
        """Native strategies should have auto-extracted parameters."""
        strategies = registry.list_strategies()
        native_with_params = [
            s for s in strategies if s.strategy_class is not None and len(s.parameters) > 0
        ]
        if native_with_params:
            info = native_with_params[0]
            for _param_name, spec in info.parameters.items():
                assert spec.type in ("int", "float", "bool", "choice")
                assert spec.default is not None
