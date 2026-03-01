"""CLI subcommand: collect."""

from __future__ import annotations

import argparse
from typing import Any


def register(subparsers: Any) -> None:
    """Register collect subcommand."""
    p = subparsers.add_parser("collect", help="Collect OHLCV data from exchange")
    p.add_argument("--tickers", nargs="+", default=None, metavar="TICKER")
    p.add_argument("--interval", default="day", metavar="INTERVAL")
    p.add_argument(
        "--source",
        default="upbit",
        choices=["upbit", "binance"],
        metavar="EXCHANGE",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Incrementally update all existing data files (no --tickers needed)",
    )
    p.add_argument(
        "--full-refresh", action="store_true", help="Re-fetch all data ignoring existing"
    )
    p.set_defaults(func=_run_collect)


def _run_collect(args: argparse.Namespace) -> None:
    from typing import cast

    from src.data.collector_factory import DataCollectorFactory, ExchangeName

    if not args.update and not args.tickers:
        raise SystemExit("Error: --tickers is required unless --update is specified")

    collector = DataCollectorFactory.create(exchange_name=cast(ExchangeName, args.source))

    if args.update and not args.tickers:
        # Auto-detect all existing parquet files for this source
        pairs = _find_existing_data(args.source)
        if not pairs:
            print(f"No existing data found for {args.source}. Run with --tickers first.")
            return
        print(f"Updating {len(pairs)} file(s) from {args.source} ...")
        for ticker, interval in pairs:
            count = collector.collect(ticker, interval, full_refresh=args.full_refresh)
            status = f"+{count} new candles" if count > 0 else "already up-to-date"
            print(f"  {ticker} ({interval}): {status}")
    else:
        # Normal collect with explicit tickers
        print(f"Collecting {args.interval} candles from {args.source} ...")
        for ticker in args.tickers:
            count = collector.collect(ticker, args.interval, full_refresh=args.full_refresh)
            status = f"+{count} new candles" if count > 0 else "already up-to-date"
            print(f"  {ticker}: {status}")


def _find_existing_data(source: str) -> list[tuple[str, str]]:
    """Scan data directory and return (ticker, interval) for each existing parquet file."""
    from src.config import BINANCE_DATA_DIR, UPBIT_DATA_DIR

    data_dir = UPBIT_DATA_DIR if source == "upbit" else BINANCE_DATA_DIR
    if not data_dir.exists():
        return []

    pairs: list[tuple[str, str]] = []
    for f in sorted(data_dir.glob("*.parquet")):
        # Filename format: {TICKER}_{INTERVAL}.parquet  e.g. KRW-BTC_day.parquet
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


__all__ = ["register"]
