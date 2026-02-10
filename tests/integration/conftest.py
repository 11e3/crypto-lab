"""
Integration test fixtures for execution module.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from src.execution.event_bus import set_event_bus


@pytest.fixture(autouse=True)
def reset_event_bus() -> Generator[None, None, None]:
    """Reset EventBus singleton before and after each test for isolation."""
    set_event_bus(None)
    yield
    set_event_bus(None)
