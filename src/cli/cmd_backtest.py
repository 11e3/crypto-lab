"""CLI subcommands: backtest, optimize."""

from __future__ import annotations

import argparse
from typing import Any

from src.cli._helpers import build_config, build_param_grid, parse_date
from src.cli._output import save_output


def register(subparsers: Any) -> None:
    """Register backtest and optimize subcommands."""
    _register_backtest(subparsers)
    _register_optimize(subparsers)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tickers", nargs="+", required=True, metavar="TICKER")
    parser.add_argument("--strategy", required=True, metavar="NAME")
    parser.add_argument("--start", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=1_000_000, metavar="KRW")
    parser.add_argument("--slots", type=int, default=5, metavar="N")
    parser.add_argument("--fee", type=float, default=0.0005, metavar="RATE")
    parser.add_argument("--interval", default="day", metavar="INTERVAL")


def _register_backtest(subparsers: Any) -> None:
    p = subparsers.add_parser("backtest", help="Run strategy backtest")
    _add_common_args(p)
    p.add_argument("--output", default=None, metavar="PATH", help="Save HTML report to PATH")
    p.add_argument("--no-save", action="store_true", dest="no_save", help="Skip saving result to results/")
    p.set_defaults(func=_run_backtest)


def _register_optimize(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "optimize", help="Optimize strategy parameters via grid/random search"
    )
    _add_common_args(p)
    p.add_argument("--metric", default="sharpe_ratio", metavar="METRIC")
    p.add_argument("--method", default="grid", choices=["grid", "random"])
    p.add_argument("--n-iter", type=int, default=100, dest="n_iter", metavar="N")
    p.add_argument("--workers", type=int, default=None, metavar="N")
    p.add_argument("--no-save", action="store_true", dest="no_save", help="Skip saving result to results/")
    p.set_defaults(func=_run_optimize)


def _run_backtest(args: argparse.Namespace) -> None:
    from src.backtester.engine import run_backtest
    from src.backtester.report_pkg.report import generate_report
    from src.strategies.registry import registry

    strategy = registry.create(args.strategy)
    config = build_config(args)

    result = run_backtest(
        strategy=strategy,
        tickers=args.tickers,
        interval=args.interval,
        config=config,
        start_date=parse_date(args.start),
        end_date=parse_date(args.end),
    )

    with save_output("backtest", args.strategy, not args.no_save):
        print(f"\n=== Backtest: {args.strategy} ===")
        print(f"  Total Return : {result.total_return:.2f}%")
        print(f"  CAGR         : {result.cagr:.2f}%")
        print(f"  MDD          : -{result.mdd:.2f}%")
        print(f"  Sharpe       : {result.sharpe_ratio:.2f}")
        print(f"  Win Rate     : {result.win_rate:.1f}%")
        print(f"  Total Trades : {result.total_trades}")

    if args.output:
        from pathlib import Path

        generate_report(
            result,
            strategy_name=args.strategy,
            save_path=Path(args.output),
            show=False,
        )
        print(f"  Report saved : {args.output}")


def _run_optimize(args: argparse.Namespace) -> None:
    from src.backtester.optimization import optimize_strategy_parameters
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

    result = optimize_strategy_parameters(
        strategy_factory=_factory,
        param_grid=param_grid,
        tickers=args.tickers,
        interval=args.interval,
        config=config,
        metric=args.metric,
        method=args.method,
        n_iter=args.n_iter,
        n_workers=args.workers,
    )

    with save_output("optimize", args.strategy, not args.no_save):
        print(f"\n=== Optimize: {args.strategy} ===")
        print(f"  Best Score ({args.metric}): {result.best_score:.4f}")
        print("  Best Params:")
        for k, v in result.best_params.items():
            print(f"    {k}: {v}")


__all__ = ["register"]
