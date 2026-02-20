"""CLI subcommand: collect."""

from __future__ import annotations

import argparse
from typing import Any


def register(subparsers: Any) -> None:
    """Register collect subcommand."""
    p = subparsers.add_parser("collect", help="Collect OHLCV data from exchange")
    p.add_argument("--tickers", nargs="+", required=True, metavar="TICKER")
    p.add_argument("--interval", default="day", metavar="INTERVAL")
    p.add_argument(
        "--source",
        default="upbit",
        choices=["upbit", "binance"],
        metavar="EXCHANGE",
    )
    p.add_argument(
        "--full-refresh", action="store_true", help="Re-fetch all data ignoring existing"
    )
    p.set_defaults(func=_run_collect)


def _run_collect(args: argparse.Namespace) -> None:
    from typing import cast

    from src.data.collector_factory import DataCollectorFactory, ExchangeName

    collector = DataCollectorFactory.create(exchange_name=cast(ExchangeName, args.source))

    print(f"Collecting {args.interval} candles from {args.source} ...")
    for ticker in args.tickers:
        count = collector.collect(ticker, args.interval, full_refresh=args.full_refresh)
        print(f"  {ticker}: +{count} new candles")


__all__ = ["register"]
