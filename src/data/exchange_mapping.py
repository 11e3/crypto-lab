"""Exchange-agnostic interval and symbol mapping utilities."""

from typing import Literal

__all__ = [
    "BinanceInterval",
    "ExchangeName",
    "UPBIT_TO_CANONICAL",
    "BINANCE_TO_CANONICAL",
    "CANONICAL_TO_UPBIT",
    "CANONICAL_TO_BINANCE",
    "get_exchange_intervals",
]

ExchangeName = Literal["upbit", "binance"]

BinanceInterval = Literal[
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
]

# Upbit interval -> canonical (Binance-style) interval
UPBIT_TO_CANONICAL: dict[str, str] = {
    "minute1": "1m",
    "minute3": "3m",
    "minute5": "5m",
    "minute10": "10m",
    "minute15": "15m",
    "minute30": "30m",
    "minute60": "1h",
    "minute240": "4h",
    "day": "1d",
    "week": "1w",
    "month": "1M",
}

# Binance interval -> canonical (identity mapping)
BINANCE_TO_CANONICAL: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}

# Reverse mappings
CANONICAL_TO_UPBIT: dict[str, str] = {v: k for k, v in UPBIT_TO_CANONICAL.items()}
CANONICAL_TO_BINANCE: dict[str, str] = {v: k for k, v in BINANCE_TO_CANONICAL.items()}


def get_exchange_intervals(exchange: ExchangeName) -> list[tuple[str, str]]:
    """Get available intervals for an exchange as (code, display_name) tuples.

    Args:
        exchange: Exchange name

    Returns:
        List of (interval_code, display_name) tuples
    """
    if exchange == "upbit":
        return [
            ("minute1", "1 min"),
            ("minute3", "3 min"),
            ("minute5", "5 min"),
            ("minute10", "10 min"),
            ("minute15", "15 min"),
            ("minute30", "30 min"),
            ("minute60", "1 hour"),
            ("minute240", "4 hours"),
            ("day", "Daily"),
            ("week", "Weekly"),
            ("month", "Monthly"),
        ]
    elif exchange == "binance":
        return [
            ("1m", "1 min"),
            ("3m", "3 min"),
            ("5m", "5 min"),
            ("15m", "15 min"),
            ("30m", "30 min"),
            ("1h", "1 hour"),
            ("2h", "2 hours"),
            ("4h", "4 hours"),
            ("6h", "6 hours"),
            ("8h", "8 hours"),
            ("12h", "12 hours"),
            ("1d", "Daily"),
            ("3d", "3 days"),
            ("1w", "Weekly"),
            ("1M", "Monthly"),
        ]
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")
