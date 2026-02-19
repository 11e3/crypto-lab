"""Volume-Momentum 전략 (바이낸스 선물).

Factor screen 기반 복합 전략:
- Factor 1: relative_volume_rank — 거래량 상위 자산 (IC IR = +0.19)
- Factor 2: momentum (10d/20d) — 모멘텀 상위 자산 (IC IR = +0.17)
- 필터: RSI(14) 과매수/과매도 구간 제외 (optional)

논리: 거래량 증가 + 가격 상승 모멘텀 = 기관/스마트머니 유입 신호
      거래량 없는 모멘텀 = 지속력 약한 노이즈

결과: [PENDING]

Usage:
    python research/fetch_data.py
    python research/volume_momentum.py
    python research/volume_momentum.py --sweep
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


def load_data(interval: str = "1d") -> tuple[pd.DataFrame, pd.DataFrame]:
    """research/data/ 에서 parquet 로드 -> (closes, volumes)."""
    files = sorted(DATA_DIR.glob(f"*_{interval}.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR} for interval={interval}")

    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for f in files:
        symbol = f.stem.replace(f"_{interval}", "")
        df = pd.read_parquet(f)
        closes[symbol] = df["close"]
        volumes[symbol] = df["close"] * df["volume"]

    close_df = pd.DataFrame(closes).dropna(how="all").ffill()
    vol_df = pd.DataFrame(volumes).reindex(close_df.index).fillna(0)
    print(f"Loaded {len(close_df.columns)} symbols, {len(close_df)} rows "
          f"({close_df.index[0]} ~ {close_df.index[-1]})")
    return close_df, vol_df


# === 시그널 생성 ===


def generate_signals(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    mom_lookback: int = 10,
    vol_ma_short: int = 5,
    vol_ma_long: int = 20,
    top_n: int = 2,
    rebal_days: int = 7,
    direction: str = "long",
    mom_weight: float = 0.5,
    rsi_filter: bool = False,
    rsi_upper: int = 80,
    rsi_lower: int = 20,
) -> pd.DataFrame:
    """Volume-Momentum 복합 시그널.

    Args:
        mom_lookback: 모멘텀 계산 기간
        vol_ma_short: 단기 거래량 이평
        vol_ma_long: 장기 거래량 이평
        top_n: 롱/숏 자산 수
        rebal_days: 리밸런싱 주기 (일)
        direction: "long", "short", "both"
        mom_weight: 모멘텀 factor 가중치 (0~1). 1-mom_weight = volume 가중치
        rsi_filter: RSI 필터 사용 여부
        rsi_upper/rsi_lower: RSI 과매수/과매도 threshold

    Returns:
        signals: +1 (롱), -1 (숏), 0 (플랫)
    """
    # Factor 1: Volume ratio (short MA / long MA) — 높을수록 거래량 급증
    vol_short = volumes.rolling(vol_ma_short).mean()
    vol_long = volumes.rolling(vol_ma_long).mean()
    vol_ratio = vol_short / vol_long.replace(0, np.nan)

    # Factor 2: Momentum
    momentum = closes.pct_change(mom_lookback)

    # Optional: RSI filter
    rsi = None
    if rsi_filter:
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

    signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    rebal_dates = closes.index[::rebal_days]

    for date in rebal_dates:
        mom_row = momentum.loc[date].dropna()
        vol_row = vol_ratio.loc[date].dropna()

        common = mom_row.index.intersection(vol_row.index)
        if len(common) < 3:
            continue

        # RSI filter: exclude overbought/oversold
        if rsi is not None and date in rsi.index:
            rsi_row = rsi.loc[date]
            valid = common[
                (rsi_row[common] < rsi_upper) & (rsi_row[common] > rsi_lower)
            ]
            if len(valid) < 3:
                valid = common  # fallback
            common = valid

        # Cross-sectional rank
        mom_rank = momentum.loc[date][common].rank(pct=True)
        vol_rank = vol_ratio.loc[date][common].rank(pct=True)
        composite = mom_weight * mom_rank + (1 - mom_weight) * vol_rank

        # Long: top_n (highest composite = strong momentum + high volume)
        if direction in ("long", "both"):
            # 추가 필터: 모멘텀이 양수인 자산만 롱
            long_candidates = composite[momentum.loc[date][common] > 0]
            if len(long_candidates) > 0:
                top = long_candidates.nlargest(min(top_n, len(long_candidates))).index
                signals.loc[date, top] = 1.0

        # Short: bottom_n (weak momentum + low volume)
        if direction in ("short", "both"):
            short_candidates = composite[momentum.loc[date][common] < 0]
            if len(short_candidates) > 0:
                bottom = short_candidates.nsmallest(min(top_n, len(short_candidates))).index
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
    volumes: pd.DataFrame,
) -> pd.DataFrame:
    """주요 파라미터 조합 스윕."""
    results = []

    mom_lookbacks = [5, 10, 20]
    vol_ma_shorts = [3, 5, 10]
    vol_ma_longs = [20, 40]
    top_ns = [1, 2, 3]
    rebal_days_list = [3, 7, 14]
    mom_weights = [0.3, 0.5, 0.7]
    directions = ["long"]
    vol_targets = [None, 0.25]

    combos = list(product(mom_lookbacks, vol_ma_shorts, vol_ma_longs, top_ns,
                          rebal_days_list, mom_weights, directions))
    # Filter: short MA < long MA
    combos = [(ml, vs, vl, tn, rd, mw, d) for ml, vs, vl, tn, rd, mw, d in combos if vs < vl]
    total = len(combos) * len(vol_targets)
    print(f"\nRunning parameter sweep: {total} combinations")

    for ml, vs, vl, tn, rd, mw, d in combos:
        sigs = generate_signals(closes, volumes, ml, vs, vl, tn, rd, d, mw)
        for vt in vol_targets:
            res = backtest(closes, sigs, vol_target=vt)
            results.append({
                "mom_lb": ml,
                "vol_s": vs,
                "vol_l": vl,
                "top_n": tn,
                "rebal": rd,
                "mom_w": mw,
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
    parser.add_argument("--mom-lookback", type=int, default=10)
    parser.add_argument("--vol-ma-short", type=int, default=5)
    parser.add_argument("--vol-ma-long", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--rebal-days", type=int, default=7)
    parser.add_argument("--mom-weight", type=float, default=0.5)
    parser.add_argument("--direction", choices=["long", "short", "both"], default="long")
    parser.add_argument("--rsi-filter", action="store_true")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--vol-target", type=float, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--exclude", nargs="+", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    closes, volumes = load_data(args.interval)

    if args.symbols:
        valid = [s for s in args.symbols if s in closes.columns]
        if not valid:
            print(f"No matching symbols. Available: {list(closes.columns)}")
            return
        closes = pd.DataFrame(closes[valid])
        volumes = pd.DataFrame(volumes[valid])
    if args.exclude:
        keep = [c for c in closes.columns if c not in args.exclude]
        closes = pd.DataFrame(closes[keep])
        volumes = pd.DataFrame(volumes[keep])
    print(f"Universe: {list(closes.columns)}")

    if args.sweep:
        parameter_sweep(closes, volumes)
        return

    params = {
        "mom_lookback": args.mom_lookback,
        "vol_ma_short": args.vol_ma_short,
        "vol_ma_long": args.vol_ma_long,
        "top_n": args.top_n,
        "rebal_days": args.rebal_days,
        "mom_weight": args.mom_weight,
        "direction": args.direction,
        "rsi_filter": args.rsi_filter,
        "vol_target": args.vol_target,
    }

    signals = generate_signals(
        closes, volumes,
        args.mom_lookback, args.vol_ma_short, args.vol_ma_long,
        args.top_n, args.rebal_days, args.direction, args.mom_weight,
        args.rsi_filter,
    )
    result = backtest(closes, signals, args.leverage, vol_target=args.vol_target)
    print_metrics(result, params)

    if not args.no_plot:
        plot_equity(result, f"Volume-Momentum (mom={args.mom_lookback}, vol={args.vol_ma_short}/{args.vol_ma_long}, "
                    f"n={args.top_n})")


if __name__ == "__main__":
    main()
