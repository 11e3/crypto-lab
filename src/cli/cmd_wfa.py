"""CLI subcommand: wfa (walk-forward analysis)."""

from __future__ import annotations

import argparse
from typing import Any

from src.cli._helpers import build_config, build_param_grid, parse_date


def register(subparsers: Any) -> None:
    """Register wfa subcommand."""
    p = subparsers.add_parser("wfa", help="Run walk-forward analysis")
    p.add_argument("--tickers", nargs="+", required=True, metavar="TICKER")
    p.add_argument("--strategy", required=True, metavar="NAME")
    p.add_argument("--start", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--capital", type=float, default=1_000_000, metavar="KRW")
    p.add_argument("--slots", type=int, default=5, metavar="N")
    p.add_argument("--fee", type=float, default=0.0005, metavar="RATE")
    p.add_argument("--interval", default="day", metavar="INTERVAL")
    p.add_argument("--opt-days", type=int, default=365, dest="opt_days", metavar="N")
    p.add_argument("--test-days", type=int, default=90, dest="test_days", metavar="N")
    p.add_argument("--step-days", type=int, default=90, dest="step_days", metavar="N")
    p.add_argument("--metric", default="sharpe_ratio", metavar="METRIC")
    p.add_argument("--workers", type=int, default=None, metavar="N")
    p.set_defaults(func=_run_wfa)


def _run_wfa(args: argparse.Namespace) -> None:
    from src.backtester.wfa import run_walk_forward_analysis
    from src.strategies.registry import registry

    cls = registry.get_class(args.strategy)
    if cls is None:
        raise SystemExit(f"Unknown strategy: {args.strategy!r}")

    param_grid = build_param_grid(cls.parameter_schema())
    if not param_grid:
        raise SystemExit(f"Strategy {args.strategy!r} has no tunable parameters.")

    config = build_config(args)
    name = args.strategy

    def _factory(params: dict[str, Any]) -> Any:
        return registry.create(name, **params)

    result = run_walk_forward_analysis(
        strategy_factory=_factory,
        param_grid=param_grid,
        tickers=args.tickers,
        interval=args.interval,
        config=config,
        optimization_days=args.opt_days,
        test_days=args.test_days,
        step_days=args.step_days,
        metric=args.metric,
        start_date=parse_date(args.start),
        end_date=parse_date(args.end),
        n_workers=args.workers,
    )

    _print_wfa_result(result, args.metric)


def _print_wfa_result(result: Any, metric: str) -> None:
    print("\n=== Walk-Forward Analysis ===")
    print(f"  Periods       : {result.total_periods}")
    print(f"  Positive      : {result.positive_periods}/{result.total_periods}")
    print(f"  Consistency   : {result.consistency_rate:.1f}%")
    print(f"  Avg CAGR      : {result.avg_test_cagr:.2f}%")
    print(f"  Avg Sharpe    : {result.avg_test_sharpe:.2f}")
    print(f"  Avg MDD       : {result.avg_test_mdd:.2f}%")
    print()
    for period in result.periods:
        score = getattr(period.test_result, metric, 0.0) if period.test_result is not None else 0.0
        print(f"  {period.test_start} → {period.test_end} | {metric}={score:.4f}")


__all__ = ["register"]
