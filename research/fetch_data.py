"""바이낸스 선물 OHLCV 데이터 수집 유틸리티.

Usage:
    python research/fetch_data.py                    # 기본 (BTC, ETH, 1d, 3년)
    python research/fetch_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --interval 4h --days 365
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 바이낸스 선물 (USDT-M)
exchange = ccxt.binanceusdm({"enableRateLimit": True})
exchange.load_markets()

DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
]


def fetch_ohlcv(
    symbol: str,
    interval: str = "1d",
    days: int | None = 1095,
    limit_per_call: int = 1500,
) -> pd.DataFrame:
    """바이낸스 선물 OHLCV를 페이지네이션으로 수집."""
    if days is None:
        since = 0  # 상장일부터
    else:
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    all_rows: list[list] = []

    while True:
        rows = exchange.fetch_ohlcv(symbol, interval, since=since, limit=limit_per_call)
        if not rows:
            break
        all_rows.extend(rows)
        since = rows[-1][0] + 1  # 마지막 타임스탬프 다음부터
        if len(rows) < limit_per_call:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def save_ohlcv(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    """parquet으로 저장."""
    clean = symbol.replace("/", "").replace(":USDT", "")
    fname = f"{clean}_{interval}.parquet"
    path = DATA_DIR / fname
    df.to_parquet(path)
    print(f"  saved {path} ({len(df)} rows, {df.index[0]} ~ {df.index[-1]})")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance futures OHLCV data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols (e.g., BTCUSDT ETHUSDT or BTC/USDT:USDT)",
    )
    parser.add_argument("--interval", default="1d", help="Candle interval (1d, 4h, 1h)")
    parser.add_argument("--days", type=int, default=1095, help="Days of history")
    parser.add_argument("--all", action="store_true", help="Fetch from listing date")
    args = parser.parse_args()

    # 사용자 입력 형태 → ccxt 형태 변환 (e.g., BTCUSDT → BTC/USDT:USDT)
    symbols = []
    for s in args.symbols:
        if ":USDT" in s:
            symbols.append(s)  # 이미 ccxt 형태
        elif "/" in s:
            symbols.append(f"{s}:USDT")
        else:
            base = s.replace("USDT", "")
            symbols.append(f"{base}/USDT:USDT")

    days = None if args.all else args.days
    print(f"Fetching {len(symbols)} symbols, interval={args.interval}, days={'all' if days is None else days}")

    for symbol in symbols:
        print(f"\n[{symbol}]")
        try:
            df = fetch_ohlcv(symbol, args.interval, days)
            save_ohlcv(df, symbol, args.interval)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
