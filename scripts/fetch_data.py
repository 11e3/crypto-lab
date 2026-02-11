#!/usr/bin/env python3
"""
Data Fetcher CLI - Download OHLCV data from exchanges.

Part of the Crypto Lab Ecosystem.

Usage:
    # Download specific symbols
    python scripts/fetch_data.py --symbols BTC,ETH,XRP --interval day

    # Update existing data (incremental)
    python scripts/fetch_data.py --update

    # Full refresh (re-download all)
    python scripts/fetch_data.py --symbols BTC --interval day --full-refresh

    # Download multiple intervals
    python scripts/fetch_data.py --symbols BTC,ETH --interval day,minute240,minute30

    # List available data
    python scripts/fetch_data.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import RAW_DATA_DIR
from src.data.collector import UpbitDataCollector
from src.data.collector_fetch import Interval
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Default symbols for Korean market
DEFAULT_SYMBOLS = ["BTC", "ETH", "XRP", "TRX", "SOL", "ADA", "DOGE", "MATIC"]

# Map short names to Upbit tickers
SYMBOL_MAP = {
    "BTC": "KRW-BTC",
    "ETH": "KRW-ETH",
    "XRP": "KRW-XRP",
    "TRX": "KRW-TRX",
    "SOL": "KRW-SOL",
    "ADA": "KRW-ADA",
    "DOGE": "KRW-DOGE",
    "MATIC": "KRW-MATIC",
    "EOS": "KRW-EOS",
    "LINK": "KRW-LINK",
    "DOT": "KRW-DOT",
    "ATOM": "KRW-ATOM",
    "AVAX": "KRW-AVAX",
    "SHIB": "KRW-SHIB",
}

# Valid intervals
VALID_INTERVALS: list[Interval] = [
    "minute1",
    "minute3",
    "minute5",
    "minute10",
    "minute15",
    "minute30",
    "minute60",
    "minute240",
    "day",
    "week",
    "month",
]


def parse_symbols(symbols_str: str) -> list[str]:
    """Parse comma-separated symbols to Upbit tickers."""
    symbols = [s.strip().upper() for s in symbols_str.split(",")]
    tickers = []
    for symbol in symbols:
        if symbol.startswith("KRW-"):
            tickers.append(symbol)
        elif symbol in SYMBOL_MAP:
            tickers.append(SYMBOL_MAP[symbol])
        else:
            # Try as-is with KRW prefix
            tickers.append(f"KRW-{symbol}")
    return tickers


def parse_intervals(intervals_str: str) -> list[Interval]:
    """Parse comma-separated intervals."""
    intervals = [i.strip().lower() for i in intervals_str.split(",")]
    valid: list[Interval] = []
    for interval in intervals:
        # Map common aliases
        if interval == "1d":
            interval = "day"
        elif interval == "1w":
            interval = "week"
        elif interval == "1m":
            interval = "minute1"
        elif interval == "4h":
            interval = "minute240"
        elif interval == "1h":
            interval = "minute60"
        elif interval == "30m":
            interval = "minute30"
        elif interval == "15m":
            interval = "minute15"
        elif interval == "5m":
            interval = "minute5"

        if interval in VALID_INTERVALS:
            valid.append(interval)  # type: ignore[arg-type]
        else:
            logger.warning(f"Invalid interval: {interval}. Skipping.")
    return valid


def list_available_data(data_dir: Path) -> None:
    """List all available data files."""
    print(f"\nAvailable data in: {data_dir}")
    print("-" * 60)

    if not data_dir.exists():
        print("  No data directory found.")
        return

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print("  No data files found.")
        return

    print(f"{'Ticker':<15} {'Interval':<15} {'Size (KB)':<12} {'Modified'}")
    print("-" * 60)

    for f in files:
        name_parts = f.stem.rsplit("_", 1)
        if len(name_parts) == 2:
            ticker, interval = name_parts
        else:
            ticker, interval = f.stem, "unknown"

        size_kb = f.stat().st_size / 1024
        modified = f.stat().st_mtime

        from datetime import datetime

        mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")

        print(f"{ticker:<15} {interval:<15} {size_kb:<12.1f} {mod_time}")

    print(f"\nTotal: {len(files)} files")


def download_data(
    tickers: list[str],
    intervals: list[Interval],
    full_refresh: bool = False,
) -> dict[str, int]:
    """Download data for specified tickers and intervals."""
    collector = UpbitDataCollector()

    print("\nDownloading data...")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Intervals: {', '.join(intervals)}")
    print(f"  Mode: {'Full Refresh' if full_refresh else 'Incremental Update'}")
    print("-" * 60)

    results = collector.collect_multiple(
        tickers=tickers,
        intervals=intervals,
        full_refresh=full_refresh,
    )

    # Print summary
    print("\nDownload Summary:")
    print("-" * 60)
    print(f"{'Ticker-Interval':<30} {'New Candles'}")
    print("-" * 60)

    total_new = 0
    errors = 0
    for key, count in results.items():
        if count < 0:
            print(f"{key:<30} ERROR")
            errors += 1
        else:
            print(f"{key:<30} {count:,}")
            total_new += count

    print("-" * 60)
    print(f"Total new candles: {total_new:,}")
    if errors > 0:
        print(f"Errors: {errors}")

    return results


def update_all_data() -> dict[str, int]:
    """Update all existing data files."""
    data_dir = RAW_DATA_DIR

    if not data_dir.exists():
        print("No existing data to update.")
        return {}

    files = list(data_dir.glob("*.parquet"))
    if not files:
        print("No existing data to update.")
        return {}

    # Extract unique tickers and intervals
    tickers_set: set[str] = set()
    intervals_set: set[Interval] = set()

    for f in files:
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            ticker, interval_str = parts
            tickers_set.add(ticker)
            if interval_str in VALID_INTERVALS:
                intervals_set.add(interval_str)  # type: ignore[arg-type]

    if not tickers_set or not intervals_set:
        print("Could not parse existing data files.")
        return {}

    print(f"Found {len(files)} existing data files.")
    return download_data(
        tickers=sorted(tickers_set),
        intervals=sorted(intervals_set),
        full_refresh=False,
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download OHLCV data from exchanges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --symbols BTC,ETH --interval day
  %(prog)s --update
  %(prog)s --symbols BTC --interval day,minute240 --full-refresh
  %(prog)s --list

Available Intervals:
  minute1, minute3, minute5, minute10, minute15, minute30,
  minute60, minute240, day, week, month

  Aliases: 1d=day, 1w=week, 4h=minute240, 1h=minute60, 30m=minute30
        """,
    )

    parser.add_argument(
        "--symbols",
        "-s",
        type=str,
        help="Comma-separated symbols (e.g., BTC,ETH,XRP)",
    )

    parser.add_argument(
        "--interval",
        "-i",
        type=str,
        default="day",
        help="Comma-separated intervals (default: day)",
    )

    parser.add_argument(
        "--update",
        "-u",
        action="store_true",
        help="Update all existing data (incremental)",
    )

    parser.add_argument(
        "--full-refresh",
        "-f",
        action="store_true",
        help="Force full refresh (ignore existing data)",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available data files",
    )

    parser.add_argument(
        "--all-default",
        "-a",
        action="store_true",
        help="Download all default symbols",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        list_available_data(RAW_DATA_DIR)
        return 0

    # Update mode
    if args.update:
        results = update_all_data()
        return 0 if all(v >= 0 for v in results.values()) else 1

    # Download mode
    if args.symbols:
        tickers = parse_symbols(args.symbols)
    elif args.all_default:
        tickers = [SYMBOL_MAP[s] for s in DEFAULT_SYMBOLS if s in SYMBOL_MAP]
    else:
        parser.print_help()
        print("\nError: Please specify --symbols, --update, or --all-default")
        return 1

    intervals = parse_intervals(args.interval)
    if not intervals:
        print("Error: No valid intervals specified.")
        return 1

    results = download_data(
        tickers=tickers,
        intervals=intervals,
        full_refresh=args.full_refresh,
    )

    return 0 if all(v >= 0 for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
