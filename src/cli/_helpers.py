"""Shared CLI helpers: date parsing, BacktestConfig factory, parameter grid builder."""

from __future__ import annotations

import argparse
from datetime import date
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from src.backtester.models import BacktestConfig


def parse_date(value: str | None) -> date | None:
    """Parse an ISO-8601 date string, or return None."""
    if value is None:
        return None
    return date.fromisoformat(value)


def build_config(args: argparse.Namespace) -> BacktestConfig:
    """Build BacktestConfig from common CLI arguments (capital, slots, fee)."""
    from src.backtester.models import BacktestConfig

    return BacktestConfig(
        initial_capital=float(args.capital),
        max_slots=int(args.slots),
        fee_rate=float(args.fee),
    )


def build_param_grid(schema: dict[str, object]) -> dict[str, list[Any]]:
    """Build parameter grid from a strategy's parameter_schema() return value.

    Supports float ranges (min/max/step) and int ranges (min/max/step).
    """
    grid: dict[str, list[Any]] = {}
    for name, raw_spec in schema.items():
        spec = cast(dict[str, Any], raw_spec)
        kind: str = spec.get("type", "")
        if kind == "float":
            lo, hi, step = float(spec["min"]), float(spec["max"]), float(spec["step"])
            values: list[Any] = []
            v = lo
            while v <= hi + 1e-9:
                values.append(round(v, 8))
                v += step
            grid[name] = values
        elif kind == "int":
            i_lo = int(spec["min"])
            i_hi = int(spec["max"])
            i_step = int(spec["step"])
            grid[name] = list(range(i_lo, i_hi + 1, i_step))
    return grid
