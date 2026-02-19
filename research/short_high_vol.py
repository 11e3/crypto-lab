"""Short High-Volatility 전략 (바이낸스 선물).

Factor screen 기반 단일 factor 전략:
- Factor: NATR(14) — IC IR = -0.252 (전체 factor 중 최강)
- Short: NATR 상위 자산 (고변동성 → 이후 하락)
- Long: NATR 하위 자산 (저변동성 → 이후 상승)

Low-volatility anomaly: 변동성 높은 자산이 risk-adjusted 기준 underperform.
크립토에서도 유효 — 급등 후 변동성 폭발한 자산이 평균회귀.

결과: [PENDING]

Usage:
    python research/fetch_data.py
    python research/short_high_vol.py
    python research/short_high_vol.py --sweep
    python research/short_high_vol.py --direction both --top-n 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

TAKER_FEE = 0.0005
SLIPPAGE = 0.0003
COST_PER_LEG = TAKER_FEE + SLIPPAGE


# === 데이터 로드 ===


def load_data(interval: str = "1d") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """research/data/ 에서 parquet 로드 -> (closes, highs, lows, volumes)."""
    files = sorted(DATA_DIR.glob(f"*_{interval}.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR} for interval={interval}")

    closes_d: dict[str, pd.Series] = {}
    highs_d: dict[str, pd.Series] = {}
    lows_d: dict[str, pd.Series] = {}
    volumes_d: dict[str, pd.Series] = {}
    for f in files:
        symbol = f.stem.replace(f"_{interval}", "")
        df = pd.read_parquet(f)
        closes_d[symbol] = df["close"]
        highs_d[symbol] = df["high"]
        lows_d[symbol] = df["low"]
        volumes_d[symbol] = df["close"] * df["volume"]

    close_df = pd.DataFrame(closes_d).dropna(how="all").ffill()
    high_df = pd.DataFrame(highs_d).reindex(close_df.index).ffill()
    low_df = pd.DataFrame(lows_d).reindex(close_df.index).ffill()
    vol_df = pd.DataFrame(volumes_d).reindex(close_df.index).fillna(0)

    print(f"Loaded {len(close_df.columns)} symbols, {len(close_df)} rows "
          f"({close_df.index[0]} ~ {close_df.index[-1]})")
    return close_df, high_df, low_df, vol_df


# === 시그널 생성 ===


def generate_signals(
    closes: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    atr_period: int = 14,
    vol_lookback: int = 20,
    vol_metric: str = "natr",
    top_n: int = 2,
    rebal_days: int = 7,
    direction: str = "both",
) -> pd.DataFrame:
    """Short High-Vol 시그널.

    Args:
        atr_period: ATR 계산 기간
        vol_lookback: realized vol 계산 기간 (vol_metric="realized" 시)
        vol_metric: "natr" (NATR 사용) 또는 "realized" (실현변동성)
        top_n: 롱/숏 자산 수
        rebal_days: 리밸런싱 주기
        direction: "both" (롱+숏), "long" (저vol 롱만), "short" (고vol 숏만)

    Returns:
        signals: +1 (롱), -1 (숏), 0 (플랫)
    """
    if vol_metric == "natr":
        # NATR: ATR / close * 100
        tr = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)
        for sym in closes.columns:
            hl = highs[sym] - lows[sym]
            hc = (highs[sym] - closes[sym].shift(1)).abs()
            lc = (lows[sym] - closes[sym].shift(1)).abs()
            tr[sym] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        vol_score = tr.rolling(atr_period).mean() / closes * 100
    else:
        # Realized volatility (annualized)
        returns = closes.pct_change()
        vol_score = returns.rolling(vol_lookback).std() * np.sqrt(365)

    signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    rebal_dates = closes.index[::rebal_days]

    for date in rebal_dates:
        row = vol_score.loc[date].dropna()
        if len(row) < 3:
            continue

        # Short: 고변동성 자산 (NATR 높은 쪽)
        if direction in ("short", "both"):
            high_vol = row.nlargest(min(top_n, len(row))).index
            signals.loc[date, high_vol] = -1.0

        # Long: 저변동성 자산 (NATR 낮은 쪽)
        if direction in ("long", "both"):
            low_vol = row.nsmallest(min(top_n, len(row))).index
            signals.loc[date, low_vol] = 1.0

    # 리밸런싱 사이 이전 시그널 유지
    signals = signals.replace(0, np.nan)
    for date in rebal_dates:
        signals.loc[date] = signals.loc[date].fillna(0)
    signals = signals.ffill().fillna(0)

    return signals


# === 백테스트 엔진 ===


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
    leverage: float = 1.0,
    vol_target: float | None = None,
    vol_lookback: int = 20,
) -> BacktestResult:
    """시그널 기반 포트폴리오 백테스트."""
    daily_returns = closes.pct_change().fillna(0)

    n_positions = signals.abs().sum(axis=1).replace(0, 1)
    weights = signals.div(n_positions, axis=0) * leverage

    if vol_target is not None:
        port_ret = (weights.shift(1) * daily_returns).sum(axis=1)
        realized_vol = port_ret.rolling(vol_lookback).std() * np.sqrt(365)
        vol_scalar = (vol_target / realized_vol.clip(lower=0.01)).clip(upper=3.0)
        weights = weights.multiply(vol_scalar, axis=0)

    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    cost = turnover * COST_PER_LEG

    port_returns = (weights.shift(1) * daily_returns).sum(axis=1) - cost
    port_returns = port_returns.iloc[1:]

    equity = (1 + port_returns).cumprod()
    trades = int((weights.diff().abs() > 0).sum().sum())
    metrics = compute_metrics(port_returns, equity)

    return BacktestResult(
        equity=equity,
        returns=port_returns,
        positions=weights,
        trades=trades,
        metrics=metrics,
    )


# === 메트릭 ===


def compute_metrics(returns: pd.Series, equity: pd.Series) -> dict[str, float]:
    """핵심 성과 지표."""
    days = len(returns)
    years = days / 365

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    growth = max(1 + total_return, 1e-8)
    cagr = growth ** (1 / max(years, 0.01)) - 1

    vol = returns.std() * np.sqrt(365)
    sharpe = (returns.mean() * 365) / max(vol, 1e-8)

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = drawdown.min()

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
        "Days": days,
        "Annualized Days": 365,
    }


# === 시각화 ===


SCRIPT_NAME = Path(__file__).stem


def plot_equity(result: BacktestResult, title: str = "Strategy") -> None:
    """에퀴티 커브 + 드로다운."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    ax = axes[0]
    ax.plot(result.equity.index, result.equity.values, linewidth=1.2)
    ax.set_title(f"{title} - Equity Curve", fontsize=13)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    peak = result.equity.cummax()
    dd = (result.equity - peak) / peak
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    n_pos = result.positions.abs().gt(0).sum(axis=1)
    ax.bar(n_pos.index, n_pos.values, width=1.5, alpha=0.5, color="steelblue")
    ax.set_ylabel("# Positions")
    ax.set_title("Active Positions", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / f"{SCRIPT_NAME}_result.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_metrics(result: BacktestResult, params: dict) -> None:
    """결과 출력."""
    m = result.metrics
    sharpe = m["Sharpe"]
    mdd = m["MDD"]

    print("\n" + "=" * 60)
    print(f"{SCRIPT_NAME.upper()} BACKTEST RESULT")
    print("=" * 60)
    print(f"Parameters: {params}")
    print("-" * 40)
    print(f"  CAGR   : {m['CAGR']:+.2%}")
    print(f"  Sharpe : {sharpe:.4f}  {'PASS' if sharpe >= 1.0 else 'FAIL'}")
    print(f"  MDD    : {mdd:+.2%}    {'PASS' if mdd >= -0.30 else 'FAIL'}")
    print(f"  Calmar : {m['Calmar']:.4f}")
    print(f"  Vol    : {m['Volatility']:.2%}")
    print(f"  Trades : {result.trades}")

    yearly = result.returns.groupby(result.returns.index.year).apply(  # type: ignore[attr-defined]
        lambda r: (1 + r).prod() - 1
    )
    print("\n--- Yearly Returns ---")
    for year, ret in yearly.items():
        print(f"  {year}: {ret:+.2%}")

    print("=" * 60)


# === 파라미터 스윕 ===


def parameter_sweep(
    closes: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
) -> pd.DataFrame:
    """주요 파라미터 조합 스윕."""
    results = []

    atr_periods = [10, 14, 20]
    vol_lookbacks = [10, 20, 40]
    vol_metrics = ["natr", "realized"]
    top_ns = [1, 2, 3]
    rebal_days_list = [3, 7, 14]
    directions = ["both", "long", "short"]
    vol_targets = [None, 0.15, 0.25]

    combos = list(product(atr_periods, vol_lookbacks, vol_metrics, top_ns,
                          rebal_days_list, directions))
    total = len(combos) * len(vol_targets)
    print(f"\nRunning parameter sweep: {total} combinations")

    for atr_p, vlb, vm, tn, rd, d in combos:
        sigs = generate_signals(closes, highs, lows, atr_p, vlb, vm, tn, rd, d)
        for vt in vol_targets:
            res = backtest(closes, sigs, vol_target=vt)
            results.append({
                "atr": atr_p,
                "vol_lb": vlb,
                "metric": vm,
                "top_n": tn,
                "rebal": rd,
                "dir": d,
                "vol_tgt": vt or 0,
                "Sharpe": res.metrics["Sharpe"],
                "CAGR": res.metrics["CAGR"],
                "MDD": res.metrics["MDD"],
                "Calmar": res.metrics["Calmar"],
                "Trades": res.trades,
            })

    df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
    print(f"\n--- Top 10 by Sharpe ---")
    print(df.head(10).to_string(index=False, float_format="{:.4f}".format))
    print(f"\n--- Top 10 by Calmar (Sharpe >= 0.8) ---")
    filtered = df[df["Sharpe"] >= 0.8].sort_values("Calmar", ascending=False)
    print(filtered.head(10).to_string(index=False, float_format="{:.4f}".format))

    # MDD pass filter
    mdd_pass = df[df["MDD"] >= -0.30].sort_values("Sharpe", ascending=False)
    if not mdd_pass.empty:
        print(f"\n--- Top 10 MDD <= 30% ---")
        print(mdd_pass.head(10).to_string(index=False, float_format="{:.4f}".format))

    out_csv = Path(__file__).parent / f"{SCRIPT_NAME}_sweep.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")
    return df


# === 메인 ===


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{SCRIPT_NAME} backtest")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--vol-metric", choices=["natr", "realized"], default="natr")
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--rebal-days", type=int, default=7)
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--vol-target", type=float, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--exclude", nargs="+", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    closes, highs, lows, volumes = load_data(args.interval)

    if args.symbols:
        valid = [s for s in args.symbols if s in closes.columns]
        if not valid:
            print(f"No matching symbols. Available: {list(closes.columns)}")
            return
        closes = pd.DataFrame(closes[valid])
        highs = pd.DataFrame(highs[valid])
        lows = pd.DataFrame(lows[valid])
    if args.exclude:
        keep = [c for c in closes.columns if c not in args.exclude]
        closes = pd.DataFrame(closes[keep])
        highs = pd.DataFrame(highs[keep])
        lows = pd.DataFrame(lows[keep])
    print(f"Universe: {list(closes.columns)}")

    if args.sweep:
        parameter_sweep(closes, highs, lows)
        return

    params = {
        "atr_period": args.atr_period,
        "vol_lookback": args.vol_lookback,
        "vol_metric": args.vol_metric,
        "top_n": args.top_n,
        "rebal_days": args.rebal_days,
        "direction": args.direction,
        "vol_target": args.vol_target,
    }

    signals = generate_signals(
        closes, highs, lows,
        args.atr_period, args.vol_lookback, args.vol_metric,
        args.top_n, args.rebal_days, args.direction,
    )
    result = backtest(closes, signals, args.leverage, vol_target=args.vol_target)
    print_metrics(result, params)

    if not args.no_plot:
        plot_equity(result, f"Short High-Vol ({args.vol_metric}, atr={args.atr_period}, "
                    f"n={args.top_n}, {args.direction})")


if __name__ == "__main__":
    main()
