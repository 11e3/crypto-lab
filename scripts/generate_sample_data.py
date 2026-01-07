import csv
import math
import os
import random
from datetime import datetime, timedelta

RAW_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(RAW_DIR, "sample_KRW-BTC.csv")

# Deterministic seed for reproducibility
random.seed(42)

# Geometric Brownian Motion parameters
START_PRICE = 50000000.0  # 50M KRW
MU = 0.0  # drift
SIGMA = 0.04  # daily volatility
DAYS = 180
VOLUME_BASE = 50.0


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)


def generate_series():
    dt = 1.0
    prices = [START_PRICE]
    for _ in range(DAYS - 1):
        z = random.gauss(0.0, 1.0)
        next_price = prices[-1] * math.exp((MU - 0.5 * SIGMA**2) * dt + SIGMA * math.sqrt(dt) * z)
        prices.append(max(1.0, next_price))
    return prices


def to_ohlcv(prices):
    ohlcv = []
    start = datetime.utcnow() - timedelta(days=DAYS)
    for i, p in enumerate(prices):
        # Simple synthetic OHLCV around close price
        close = p
        high = p * (1.0 + abs(random.gauss(0, 0.01)))
        low = p * (1.0 - abs(random.gauss(0, 0.01)))
        open_ = (high + low) / 2.0
        volume = VOLUME_BASE + abs(random.gauss(0, 10))
        ts = (start + timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        ohlcv.append(
            {
                "timestamp": ts.isoformat(),
                "open": round(open_, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": round(volume, 3),
            }
        )
    return ohlcv


def write_csv(rows):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    ensure_dirs()
    prices = generate_series()
    rows = to_ohlcv(prices)
    write_csv(rows)
    print(f"Wrote sample OHLCV to {OUTPUT_FILE}")
