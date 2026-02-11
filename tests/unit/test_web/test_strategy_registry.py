"""Tests for strategy registry service.

Tests:
- is_bt_strategy / get_bt_strategy_type helper functions
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
    get_bt_strategy_type,
    is_bt_strategy,
    map_strategy_to_internal_type,
)

# ── Helper functions ──


class TestIsBtStrategy:
    """Test bt strategy name detection."""

    def test_bt_prefix(self) -> None:
        assert is_bt_strategy("bt_VBO") is True
        assert is_bt_strategy("bt_Momentum") is True

    def test_non_bt(self) -> None:
        assert is_bt_strategy("VanillaVBO") is False
        assert is_bt_strategy("MomentumStrategy") is False

    def test_empty_string(self) -> None:
        assert is_bt_strategy("") is False


class TestGetBtStrategyType:
    """Test bt strategy type mapping."""

    def test_known_strategies(self) -> None:
        engine, display = get_bt_strategy_type("bt_VBO")
        assert engine == "vbo"
        assert display == "bt VBO"

    def test_regime_strategy(self) -> None:
        engine, display = get_bt_strategy_type("bt_VBO_Regime")
        assert engine == "vbo_regime"

    def test_unknown_defaults_to_vbo(self) -> None:
        engine, display = get_bt_strategy_type("bt_Unknown")
        assert engine == "vbo"
        assert display == "bt VBO"


class TestMapStrategyToInternalType:
    """Test strategy name to internal type mapping."""

    def test_vanilla_vbo(self) -> None:
        assert map_strategy_to_internal_type("VanillaVBO") == "vanilla"

    def test_legacy_vbo(self) -> None:
        assert map_strategy_to_internal_type("LegacyVBO") == "legacy"

    def test_minimal_vbo(self) -> None:
        assert map_strategy_to_internal_type("MinimalVBO") == "minimal"

    def test_momentum(self) -> None:
        assert map_strategy_to_internal_type("MomentumStrategy") == "momentum"

    def test_mean_reversion(self) -> None:
        assert map_strategy_to_internal_type("MeanReversionStrategy") == "mean-reversion"

    def test_unknown_defaults_to_vanilla(self) -> None:
        assert map_strategy_to_internal_type("SomeRandomName") == "vanilla"

    def test_case_insensitive(self) -> None:
        assert map_strategy_to_internal_type("vanillavbo") == "vanilla"
        assert map_strategy_to_internal_type("MOMENTUM") == "momentum"


# ── create_analysis_strategy ──


class TestCreateAnalysisStrategy:
    """Test analysis strategy factory."""

    def test_vanilla_creates_vbo(self) -> None:
        strategy = create_analysis_strategy("vanilla")
        assert strategy is not None
        assert strategy.name == "VanillaVBO"

    def test_minimal_creates_vbo(self) -> None:
        strategy = create_analysis_strategy("minimal")
        assert strategy is not None
        assert strategy.name == "VanillaVBO"

    def test_legacy_creates_vbo_with_filters(self) -> None:
        strategy = create_analysis_strategy("legacy")
        assert strategy is not None
        assert strategy.name == "LegacyVBO"

    def test_momentum_creates_momentum(self) -> None:
        strategy = create_analysis_strategy("momentum")
        assert strategy is not None
        assert strategy.name == "Momentum"

    def test_mean_reversion_creates_mr(self) -> None:
        strategy = create_analysis_strategy("mean-reversion")
        assert strategy is not None
        assert strategy.name == "MeanReversion"

    def test_unknown_creates_default_vbo(self) -> None:
        strategy = create_analysis_strategy("unknown_type")
        assert strategy is not None
        assert strategy.name == "DefaultVBO"


# ── StrategyRegistry ──


class TestStrategyRegistry:
    """Test strategy registry auto-discovery."""

    @pytest.fixture
    def registry(self) -> StrategyRegistry:
        return StrategyRegistry()

    def test_discovers_strategies(self, registry: StrategyRegistry) -> None:
        strategies = registry.list_strategies()
        assert len(strategies) > 0

    def test_contains_vbo_strategies(self, registry: StrategyRegistry) -> None:
        names = [s.name for s in registry.list_strategies()]
        # Should find at least VBO and Momentum strategies
        assert any("VBO" in n or "Vbo" in n or "vbo" in n.lower() for n in names)

    def test_contains_bt_strategies(self, registry: StrategyRegistry) -> None:
        names = [s.name for s in registry.list_strategies()]
        bt_names = [n for n in names if n.startswith("bt_")]
        assert len(bt_names) >= 4  # At least 4 bt strategies

    def test_get_strategy(self, registry: StrategyRegistry) -> None:
        info = registry.get_strategy("bt_VBO")
        assert info is not None
        assert info.name == "bt_VBO"
        assert info.description != ""

    def test_get_strategy_nonexistent(self, registry: StrategyRegistry) -> None:
        assert registry.get_strategy("nonexistent") is None

    def test_get_strategy_class_bt(self, registry: StrategyRegistry) -> None:
        # bt strategies have None strategy_class
        cls = registry.get_strategy_class("bt_VBO")
        assert cls is None

    def test_get_parameters(self, registry: StrategyRegistry) -> None:
        params = registry.get_parameters("bt_VBO")
        assert "lookback" in params
        assert "multiplier" in params
        assert isinstance(params["lookback"], ParameterSpec)

    def test_get_parameters_empty(self, registry: StrategyRegistry) -> None:
        params = registry.get_parameters("nonexistent")
        assert params == {}

    def test_strategy_exists(self, registry: StrategyRegistry) -> None:
        assert registry.strategy_exists("bt_VBO") is True
        assert registry.strategy_exists("nonexistent") is False

    def test_strategy_info_structure(self, registry: StrategyRegistry) -> None:
        info = registry.get_strategy("bt_VBO")
        assert info is not None
        assert isinstance(info, StrategyInfo)
        assert isinstance(info.parameters, dict)
        assert isinstance(info.description, str)

    def test_native_strategy_has_class(self, registry: StrategyRegistry) -> None:
        """Native (non-bt) strategies should have a strategy_class."""
        strategies = registry.list_strategies()
        native = [s for s in strategies if not s.name.startswith("bt_")]
        if native:
            # At least one native strategy should have a class
            has_class = any(s.strategy_class is not None for s in native)
            assert has_class

    def test_parameter_extraction_from_native(self, registry: StrategyRegistry) -> None:
        """Native strategies should have auto-extracted parameters."""
        strategies = registry.list_strategies()
        native_with_params = [
            s for s in strategies if not s.name.startswith("bt_") and len(s.parameters) > 0
        ]
        if native_with_params:
            info = native_with_params[0]
            for _param_name, spec in info.parameters.items():
                assert spec.type in ("int", "float", "bool", "choice")
                assert spec.default is not None
