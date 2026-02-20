"""Tests for src/strategies/registry.py — StrategyFactory."""

from __future__ import annotations

import pytest

from src.strategies.registry import StrategyFactory

# ---------------------------------------------------------------------------
# Minimal concrete strategy for testing
# ---------------------------------------------------------------------------


class _ConcreteStrategy:
    """Minimal strategy stub (no ABC requirement for registry)."""

    def __init__(self, name: str = "stub", **kwargs: object) -> None:
        self.name = name
        self.kwargs = kwargs


class _AnotherStrategy:
    def __init__(self, name: str = "other", **kwargs: object) -> None:
        self.name = name


# ---------------------------------------------------------------------------
# StrategyFactory tests
# ---------------------------------------------------------------------------


class TestStrategyFactory:
    """Tests for StrategyFactory."""

    def _factory(self) -> StrategyFactory:
        return StrategyFactory()

    # ---- register ----

    def test_register_stores_class(self) -> None:
        factory = self._factory()
        factory.register("STUB", _ConcreteStrategy)  # type: ignore[arg-type]
        assert factory.get_class("STUB") is _ConcreteStrategy

    def test_register_overwrites_existing(self, caplog: pytest.LogCaptureFixture) -> None:
        factory = self._factory()
        factory.register("STUB", _ConcreteStrategy)  # type: ignore[arg-type]
        factory.register("STUB", _AnotherStrategy)  # type: ignore[arg-type]
        assert factory.get_class("STUB") is _AnotherStrategy

    # ---- create ----

    def test_create_returns_instance(self) -> None:
        factory = self._factory()
        factory.register("STUB", _ConcreteStrategy)  # type: ignore[arg-type]
        instance = factory.create("STUB")
        assert isinstance(instance, _ConcreteStrategy)

    def test_create_passes_kwargs(self) -> None:
        factory = self._factory()
        factory.register("STUB", _ConcreteStrategy)  # type: ignore[arg-type]
        # "name" is passed as a kwarg to the strategy constructor (not to create())
        instance = factory.create("STUB", extra_key="custom")
        assert instance.kwargs == {"extra_key": "custom"}  # type: ignore[union-attr]

    def test_create_unknown_raises(self) -> None:
        factory = self._factory()
        with pytest.raises(ValueError, match="Unknown strategy"):
            factory.create("NONEXISTENT")

    def test_create_error_message_lists_available(self) -> None:
        factory = self._factory()
        factory.register("STUB", _ConcreteStrategy)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="STUB"):
            factory.create("MISSING")

    # ---- list_names ----

    def test_list_names_empty(self) -> None:
        factory = self._factory()
        assert factory.list_names() == []

    def test_list_names_sorted(self) -> None:
        factory = self._factory()
        factory.register("ZZZ", _ConcreteStrategy)  # type: ignore[arg-type]
        factory.register("AAA", _AnotherStrategy)  # type: ignore[arg-type]
        assert factory.list_names() == ["AAA", "ZZZ"]

    # ---- get_class ----

    def test_get_class_missing_returns_none(self) -> None:
        factory = self._factory()
        assert factory.get_class("MISSING") is None

    # ---- __contains__ ----

    def test_contains_registered(self) -> None:
        factory = self._factory()
        factory.register("STUB", _ConcreteStrategy)  # type: ignore[arg-type]
        assert "STUB" in factory

    def test_not_contains_unregistered(self) -> None:
        factory = self._factory()
        assert "NOPE" not in factory


# ---------------------------------------------------------------------------
# Module-level registry tests
# ---------------------------------------------------------------------------


class TestModuleRegistry:
    """Tests for the module-level registry singleton (with VBOV1 registered)."""

    def test_vbo_is_registered(self) -> None:
        from src.strategies import registry

        assert "VBO" in registry

    def test_vbo_create_returns_vbov1(self) -> None:
        from src.strategies import registry
        from src.strategies.volatility_breakout.vbo_v1 import VBOV1

        strategy = registry.create("VBO")
        assert isinstance(strategy, VBOV1)

    def test_vbo_list_names_includes_vbo(self) -> None:
        from src.strategies import registry

        assert "VBO" in registry.list_names()
