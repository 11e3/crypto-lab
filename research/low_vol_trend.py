"""Low-Volatility Trend 전략 (바이낸스 선물).

Factor screen 기반 복합 전략:
- Factor 1: -NATR(14) — 저변동성 자산 선호 (IC IR = -0.25, 반전 사용)
- Factor 2: dist_from_50d_high — 고점 근처 자산 선호 (IC IR = +0.23)
- Composite score = rank(-natr) + rank(dist_50d_high)
- Long: 상위 top_n, Short: 하위 top_n (optional)

논리: 변동성 낮고 + 고점 근처 = 안정적 상승 추세 자산
      변동성 높고 + 저점 근처 = 불안정 하락 추세 자산 (숏)

결과: [PENDING]

Usage:
    python research/fetch_data.py
    python research/low_vol_trend.py
    python research/low_vol_trend.py --sweep
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
    high_lookback: int = 50,
    top_n: int = 2,
    rebal_days: int = 7,
    direction: str = "long",
    vol_weight: float = 0.5,
) -> pd.DataFrame:
    """Low-Vol Trend 복합 시그널.

    Args:
        atr_period: ATR/NATR 계산 기간
        high_lookback: N일 고점 대비 거리 계산 기간
        top_n: 롱/숏 자산 수
        rebal_days: 리밸런싱 주기 (일)
        direction: "long", "short", "both"
        vol_weight: 변동성 factor 가중치 (0~1). 1-vol_weight = trend 가중치

    Returns:
        signals: +1 (롱), -1 (숏), 0 (플랫)
    """
    # Factor 1: NATR (normalized ATR) — 낮을수록 좋음
    tr = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)
    for sym in closes.columns:
        hl = highs[sym] - lows[sym]
        hc = (highs[sym] - closes[sym].shift(1)).abs()
        lc = (lows[sym] - closes[sym].shift(1)).abs()
        tr[sym] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    natr = tr.rolling(atr_period).mean() / closes * 100

    # Factor 2: distance from N-day high — 높을수록 좋음 (0 = at high)
    high_nd = closes.rolling(high_lookback).max()
    dist_high = closes / high_nd - 1  # [-1, 0] range

    # Composite score: rank(-natr) * vol_weight + rank(dist_high) * (1-vol_weight)
    # 리밸런싱 날짜에서만 계산
    signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    rebal_dates = closes.index[::rebal_days]

    for date in rebal_dates:
        natr_row = natr.loc[date].dropna()
        dist_row = dist_high.loc[date].dropna()

        common = natr_row.index.intersection(dist_row.index)
        if len(common) < 3:
            continue

        # Cross-sectional rank (pct): 0=worst, 1=best
        vol_rank = (-natr_row[common]).rank(pct=True)  # 낮은 NATR = 높은 rank
        trend_rank = dist_row[common].rank(pct=True)   # 고점 근처 = 높은 rank
        composite = vol_weight * vol_rank + (1 - vol_weight) * trend_rank

        # Long: top_n
        if direction in ("long", "both"):
            top = composite.nlargest(min(top_n, len(composite))).index
            signals.loc[date, top] = 1.0

        # Short: bottom_n
        if direction in ("short", "both"):
            bottom = composite.nsmallest(min(top_n, len(composite))).index
            signals.loc[date, bottom] = -1.0

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
    high_lookbacks = [30, 50, 80]
    top_ns = [1, 2, 3]
    rebal_days_list = [3, 7, 14]
    vol_weights = [0.3, 0.5, 0.7]
    directions = ["long"]
    vol_targets = [None, 0.25]

    combos = list(product(atr_periods, high_lookbacks, top_ns, rebal_days_list, vol_weights, directions))
    total = len(combos) * len(vol_targets)
    print(f"\nRunning parameter sweep: {total} combinations")

    for atr_p, hl, tn, rd, vw, d in combos:
        sigs = generate_signals(closes, highs, lows, atr_p, hl, tn, rd, d, vw)
        for vt in vol_targets:
            res = backtest(closes, sigs, vol_target=vt)
            results.append({
                "atr": atr_p,
                "high_lb": hl,
                "top_n": tn,
                "rebal": rd,
                "vol_w": vw,
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

    out_csv = Path(__file__).parent / f"{SCRIPT_NAME}_sweep.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")
    return df


# === 메인 ===


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{SCRIPT_NAME} backtest")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--high-lookback", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--rebal-days", type=int, default=7)
    parser.add_argument("--vol-weight", type=float, default=0.5, help="Vol factor weight (0-1)")
    parser.add_argument("--direction", choices=["long", "short", "both"], default="long")
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
        volumes = pd.DataFrame(volumes[valid])
    if args.exclude:
        keep = [c for c in closes.columns if c not in args.exclude]
        closes = pd.DataFrame(closes[keep])
        highs = pd.DataFrame(highs[keep])
        lows = pd.DataFrame(lows[keep])
        volumes = pd.DataFrame(volumes[keep])
    print(f"Universe: {list(closes.columns)}")

    if args.sweep:
        parameter_sweep(closes, highs, lows)
        return

    params = {
        "atr_period": args.atr_period,
        "high_lookback": args.high_lookback,
        "top_n": args.top_n,
        "rebal_days": args.rebal_days,
        "vol_weight": args.vol_weight,
        "direction": args.direction,
        "vol_target": args.vol_target,
    }

    signals = generate_signals(
        closes, highs, lows,
        args.atr_period, args.high_lookback, args.top_n,
        args.rebal_days, args.direction, args.vol_weight,
    )
    result = backtest(closes, signals, args.leverage, vol_target=args.vol_target)
    print_metrics(result, params)

    if not args.no_plot:
        plot_equity(result, f"Low-Vol Trend (atr={args.atr_period}, high={args.high_lookback}, "
                    f"n={args.top_n}, w={args.vol_weight})")


if __name__ == "__main__":
    main()
