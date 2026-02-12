"""Funding Rate Mean Reversion 리서치 스크립트 (바이낸스 선물).

펀딩비 극단값에서 역방향 진입 — 선물 고유의 구조적 엣지.
- 펀딩비 > 상위 임계값 → 숏 (과열)
- 펀딩비 < 하위 임계값 → 롱 (과매도)
- 홀딩: 펀딩비 정상화까지

결과: ❌ FAIL
- Daily: best Sharpe 0.098 — 시그널 자체가 약함
- 1h: best Sharpe 0.49 (entry_z=2.5, exit_z=0.5, lb=336, vol=0.25) — 개선됐으나 미달
- 원인: 펀딩비 극단값에서 가격 반전이 일관적이지 않음, 거래 비용이 alpha 잠식

Usage:
    python research/funding_rate_mr.py --fetch        # 펀딩비 데이터 수집
    python research/funding_rate_mr.py                # 백테스트
    python research/funding_rate_mr.py --sweep        # 파라미터 스윕
    python research/funding_rate_mr.py --interval 1h --sweep  # 1h 타임프레임
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- 수수료/슬리피지 (per leg) ---
TAKER_FEE = 0.0005
SLIPPAGE = 0.0003
COST_PER_LEG = TAKER_FEE + SLIPPAGE

SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "DOT/USDT:USDT",
]

BARS_PER_YEAR = {"1h": 8760, "4h": 2190, "1d": 365}


# === 펀딩비 수집 ===


def fetch_funding_rates(days: int = 2400) -> None:
    """바이낸스 선물 펀딩비 수집. 8시간 간격."""
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    exchange.load_markets()

    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000

    for symbol in SYMBOLS:
        clean = symbol.replace("/", "").replace(":USDT", "")
        print(f"\n[{clean}] Fetching funding rates...")

        all_rows = []
        current_since = since

        while True:
            try:
                rates = exchange.fetch_funding_rate_history(symbol, since=current_since, limit=1000)
            except Exception as e:
                print(f"  ERROR: {e}")
                break

            if not rates:
                break

            all_rows.extend(rates)
            current_since = rates[-1]["timestamp"] + 1

            if len(rates) < 1000:
                break
            time.sleep(exchange.rateLimit / 1000)

        if not all_rows:
            continue

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")[["fundingRate"]].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df.columns = ["funding_rate"]

        path = DATA_DIR / f"{clean}_funding.parquet"
        df.to_parquet(path)
        print(f"  saved {path} ({len(df)} rows, {df.index[0]} ~ {df.index[-1]})")


# === 데이터 로드 ===


def load_data(interval: str = "1d") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(closes, volumes, funding_rates) 반환.

    interval에 따라 펀딩비 리샘플링:
    - 1d: 일별 합산 (8h × 3)
    - 1h/4h: 8h 펀딩비를 OHLCV 인덱스에 ffill (각 펀딩 시점에 값, 나머지 0)
    """
    # OHLCV
    files = sorted(DATA_DIR.glob(f"*_{interval}.parquet"))
    if not files:
        raise FileNotFoundError(f"No OHLCV parquet files in {DATA_DIR}")

    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for f in files:
        symbol = f.stem.replace(f"_{interval}", "")
        df = pd.read_parquet(f)
        closes[symbol] = df["close"]
        volumes[symbol] = df["close"] * df["volume"]

    close_df = pd.DataFrame(closes).dropna(how="all").ffill()
    vol_df = pd.DataFrame(volumes).reindex(close_df.index).fillna(0)

    # 펀딩비 — 네이티브 8h 빈도 유지 (시그널 생성용)
    funding_native: dict[str, pd.Series] = {}
    for symbol in close_df.columns:
        fpath = DATA_DIR / f"{symbol}_funding.parquet"
        if fpath.exists():
            fdf = pd.read_parquet(fpath)
            if interval == "1d":
                funding_native[symbol] = fdf["funding_rate"].resample("1D").sum()
            else:
                # 네이티브 8h 빈도 그대로 (시그널은 이 빈도에서 계산)
                funding_native[symbol] = fdf["funding_rate"]

    fund_native_df = pd.DataFrame(funding_native)
    if interval == "1d":
        fund_native_df = fund_native_df.reindex(close_df.index).fillna(0)

    available = [c for c in close_df.columns if c in fund_native_df.columns and fund_native_df[c].abs().sum() > 0]
    print(f"Loaded {len(available)} symbols with funding data, {len(close_df)} rows ({interval})")

    return close_df[available], vol_df[available], fund_native_df[available]


# === 시그널 생성 ===


def generate_signals(
    funding: pd.DataFrame,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    lookback: int = 30,
) -> pd.DataFrame:
    """펀딩비 z-score 기반 평균회귀 시그널.

    - z > entry_z → 숏 (펀딩비 과열, 롱이 많다)
    - z < -entry_z → 롱 (펀딩비 과매도)
    - |z| < exit_z → 청산
    """
    signals = pd.DataFrame(0.0, index=funding.index, columns=funding.columns)

    for col in funding.columns:
        fr = funding[col]
        rolling_mean = fr.rolling(lookback).mean()
        rolling_std = fr.rolling(lookback).std().clip(lower=1e-8)
        z = (fr - rolling_mean) / rolling_std

        col_signals = np.zeros(len(z))
        position = 0.0
        for i in range(len(z)):
            zval = z.iloc[i]
            if np.isnan(zval):
                col_signals[i] = 0.0
                continue

            if position == 0:
                if zval > entry_z:
                    position = -1.0  # 숏
                elif zval < -entry_z:
                    position = 1.0  # 롱
            elif position > 0:  # 롱 포지션
                if zval > -exit_z:  # z가 정상화
                    position = 0.0
            elif position < 0:  # 숏 포지션
                if zval < exit_z:  # z가 정상화
                    position = 0.0

            col_signals[i] = position

        signals[col] = col_signals

    return signals


# === 백테스트 ===


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: int
    metrics: dict[str, float]


def backtest(
    closes: pd.DataFrame,
    signals: pd.DataFrame,
    vol_target: float | None = 0.15,
    vol_lookback: int = 20,
    ann_factor: int = 365,
) -> BacktestResult:
    bar_returns = closes.pct_change().fillna(0)

    # 균등 배분
    n_positions = signals.abs().sum(axis=1).replace(0, 1)
    weights = signals.div(n_positions, axis=0)

    # 변동성 타겟팅
    if vol_target is not None:
        port_ret = (weights.shift(1) * bar_returns).sum(axis=1)
        realized_vol = port_ret.rolling(vol_lookback).std() * np.sqrt(ann_factor)
        vol_scalar = (vol_target / realized_vol.clip(lower=0.01)).clip(upper=3.0)
        weights = weights.multiply(vol_scalar, axis=0)

    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    cost = turnover * COST_PER_LEG

    port_returns = (weights.shift(1) * bar_returns).sum(axis=1) - cost
    port_returns = port_returns.iloc[1:]
    equity = (1 + port_returns).cumprod()
    trades = int((weights.diff().abs() > 0).sum().sum())

    return BacktestResult(
        equity=equity,
        returns=port_returns,
        positions=weights,
        trades=trades,
        metrics=compute_metrics(port_returns, equity, ann_factor),
    )


# === 메트릭 ===


def compute_metrics(returns: pd.Series, equity: pd.Series, ann_factor: int = 365) -> dict[str, float]:
    n_bars = len(returns)
    years = n_bars / ann_factor

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1
    vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (returns.mean() * ann_factor) / max(vol, 1e-8)

    peak = equity.cummax()
    mdd = ((equity - peak) / peak).min()
    calmar = cagr / max(abs(mdd), 1e-8)

    win_rate = (returns > 0).sum() / max((returns != 0).sum(), 1)
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / max(gross_loss, 1e-8)

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MDD": mdd,
        "Calmar": calmar,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Total Return": total_return,
        "Bars": n_bars,
        "Years": round(years, 2),
    }


# === 시각화 ===


def plot_equity(result: BacktestResult, title: str = "Funding Rate MR") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    ax = axes[0]
    ax.plot(result.equity.index, result.equity.values, linewidth=1.2)
    ax.set_title(f"{title} — Equity Curve", fontsize=13)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    peak = result.equity.cummax()
    dd = (result.equity - peak) / peak
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    n_pos = result.positions.abs().gt(0).sum(axis=1)
    ax.bar(n_pos.index, n_pos.values, width=1.5, alpha=0.5, color="steelblue")
    ax.set_ylabel("# Positions")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "funding_rate_mr_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: research/funding_rate_mr_result.png")


def print_metrics(result: BacktestResult, params: dict) -> None:
    print("\n" + "=" * 60)
    print("FUNDING RATE MEAN REVERSION BACKTEST RESULT")
    print("=" * 60)
    print(f"Parameters: {params}")
    print(f"Trades: {result.trades}")
    print("-" * 40)
    for k, v in result.metrics.items():
        if k in ("Bars",):
            print(f"  {k:20s}: {int(v)}")
        elif isinstance(v, float):
            if k in ("CAGR", "Volatility", "MDD", "Win Rate", "Total Return"):
                print(f"  {k:20s}: {v:+.2%}")
            else:
                print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
    print("=" * 60)

    sharpe = result.metrics["Sharpe"]
    mdd = result.metrics["MDD"]
    print(f"\n  Sharpe >= 1.0 ? {'PASS' if sharpe >= 1.0 else 'FAIL'} ({sharpe:.2f})")
    print(f"  MDD >= -30%   ? {'PASS' if mdd >= -0.30 else 'FAIL'} ({mdd:.2%})")


# === 파라미터 스윕 ===


def parameter_sweep(
    closes: pd.DataFrame, funding: pd.DataFrame, interval: str = "1d"
) -> pd.DataFrame:
    ann = BARS_PER_YEAR.get(interval, 365)
    results = []
    entry_zs = [1.0, 1.5, 2.0, 2.5]
    exit_zs = [0.0, 0.3, 0.5]

    if interval == "1h":
        # 1h: lookback in hours. 펀딩비는 8h 간격이므로 ~3일(72h), ~7일(168h), ~14일(336h)
        lookbacks = [72, 168, 336, 720]
        vol_lookbacks = [168]  # ~7일
    elif interval == "4h":
        lookbacks = [18, 42, 84, 180]
        vol_lookbacks = [42]
    else:
        lookbacks = [14, 30, 60]
        vol_lookbacks = [20]

    vol_targets = [None, 0.10, 0.15, 0.25]

    total = len(entry_zs) * len(exit_zs) * len(lookbacks) * len(vol_targets)
    print(f"\nRunning parameter sweep: {total} combinations ({interval}, ann={ann})")

    for ez in entry_zs:
        for xz in exit_zs:
            for lb in lookbacks:
                sigs_native = generate_signals(funding, ez, xz, lb)
                # 1h/4h: 시그널을 OHLCV 인덱스에 reindex
                if interval != "1d":
                    sigs = sigs_native.reindex(closes.index, method="ffill").fillna(0)
                else:
                    sigs = sigs_native
                for vt in vol_targets:
                    res = backtest(
                        closes, sigs, vol_target=vt,
                        vol_lookback=vol_lookbacks[0], ann_factor=ann,
                    )
                    results.append(
                        {
                            "entry_z": ez,
                            "exit_z": xz,
                            "lookback": lb,
                            "vol_target": vt or 0,
                            "Sharpe": res.metrics["Sharpe"],
                            "CAGR": res.metrics["CAGR"],
                            "MDD": res.metrics["MDD"],
                            "Calmar": res.metrics["Calmar"],
                            "Trades": res.trades,
                        }
                    )

    df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
    print("\n--- Top 15 by Sharpe ---")
    print(df.head(15).to_string(index=False, float_format="{:.4f}".format))
    print("\n--- Top 15 by Calmar (Sharpe >= 0.5) ---")
    filtered = df[df["Sharpe"] >= 0.5].sort_values("Calmar", ascending=False)
    print(filtered.head(15).to_string(index=False, float_format="{:.4f}".format))
    return df


# === 메인 ===


def main() -> None:
    parser = argparse.ArgumentParser(description="Funding Rate Mean Reversion backtest")
    parser.add_argument("--fetch", action="store_true", help="Fetch funding rate data first")
    parser.add_argument("--fetch-days", type=int, default=2400, help="Days of funding data to fetch")
    parser.add_argument("--entry-z", type=float, default=1.5, help="Entry z-score threshold")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Exit z-score threshold")
    parser.add_argument("--lookback", type=int, default=30, help="Z-score lookback period")
    parser.add_argument("--vol-target", type=float, default=0.15, help="Vol target (0 to disable)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--interval", default="1d", help="OHLCV interval")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot")
    args = parser.parse_args()

    if args.fetch:
        fetch_funding_rates(args.fetch_days)
        print("\nFunding data fetched. Run again without --fetch to backtest.")
        return

    closes, volumes, funding = load_data(args.interval)
    ann = BARS_PER_YEAR.get(args.interval, 365)

    if funding.abs().sum().sum() == 0:
        print("No funding rate data found. Run with --fetch first.")
        return

    vt = args.vol_target if args.vol_target > 0 else None

    if args.sweep:
        parameter_sweep(closes, funding, interval=args.interval)
        return

    params = {
        "entry_z": args.entry_z,
        "exit_z": args.exit_z,
        "lookback": args.lookback,
        "vol_target": vt,
        "interval": args.interval,
    }

    # 시그널은 펀딩비 네이티브 빈도(8h or 1d)에서 생성
    signals_native = generate_signals(funding, args.entry_z, args.exit_z, args.lookback)

    # 1h/4h: 시그널을 OHLCV 인덱스에 reindex (ffill로 포지션 유지)
    if args.interval != "1d":
        signals = signals_native.reindex(closes.index, method="ffill").fillna(0)
    else:
        signals = signals_native

    result = backtest(closes, signals, vol_target=vt, ann_factor=ann)
    print_metrics(result, params)

    if not args.no_plot:
        plot_equity(result, f"Funding Rate MR (z={args.entry_z}/{args.exit_z}, lb={args.lookback})")


if __name__ == "__main__":
    main()
