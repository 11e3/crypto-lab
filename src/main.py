"""crypto-lab CLI entry point.

Commands:
    backtest  Run a strategy backtest on historical data
    optimize  Optimize strategy parameters via grid or random search
    collect   Collect OHLCV data from an exchange
    wfa       Walk-forward analysis to detect overfitting
    list      List all registered strategies

Example usage:
    crypto-lab list
    crypto-lab backtest --tickers KRW-BTC --strategy VBOV1 --start 2023-01-01
    crypto-lab optimize --tickers KRW-BTC KRW-ETH --strategy VBODayExit --metric sharpe_ratio
    crypto-lab collect --tickers KRW-BTC --interval day --source upbit
    crypto-lab wfa --tickers KRW-BTC --strategy VBOV1 --opt-days 365 --test-days 90
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto-lab",
        description="Crypto backtesting and strategy research toolkit",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging verbosity (default: WARNING)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # Lazy imports keep startup fast when heavy deps (pyupbit, ccxt) are not needed
    from src.cli import cmd_backtest, cmd_data, cmd_wfa

    cmd_backtest.register(subparsers)
    cmd_data.register(subparsers)
    cmd_wfa.register(subparsers)

    list_p = subparsers.add_parser("list", help="List registered strategies")
    list_p.set_defaults(func=_run_list)

    return parser


def _run_list(_args: argparse.Namespace) -> None:
    from src.strategies.registry import registry

    names = registry.list_names()
    if not names:
        print("No strategies registered.")
        return
    print("Registered strategies:")
    for name in sorted(names):
        print(f"  {name}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        sys.exit(1)

    try:
        func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
