"""Cross-Sectional Momentum 리서치 스크립트 (바이낸스 선물).

순수 횡단면 모멘텀: N일 수익률 기준 상위 롱, 하위 숏.
- 듀얼 모멘텀과 달리 절대 모멘텀 필터 없음 (항상 롱+숏)
- 시장 중립에 가까움 (동수 롱/숏)
- 리밸런싱 주기가 핵심 파라미터

결과: ⚠️ 조건부 PASS
- Best (Sharpe+MDD 동시): Sharpe 1.06, CAGR +18%, MDD -19%
  Params: lb=20, top_n=4, rebal=7d, vol_target=0.15
- Best (Sharpe만): Sharpe 1.13, CAGR +41%, MDD -55%
  Params: lb=10, top_n=3, rebal=14d, vol_target=0.30
- 듀얼 모멘텀과 수익원 겹침 (모멘텀 계열)

Usage:
    python research/fetch_data.py                          # 먼저 데이터 수집
    python research/cross_sectional_momentum.py            # 기본 파라미터
    python research/cross_sectional_momentum.py --sweep    # 파라미터 스윕
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# --- 수수료/슬리피지 (per leg) ---
TAKER_FEE = 0.0005
SLIPPAGE = 0.0003
COST_PER_LEG = TAKER_FEE + SLIPPAGE


# === 데이터 로드 ===


def load_data(interval: str = "1d") -> tuple[pd.DataFrame, pd.DataFrame]:
    """research/data/ 에서 모든 parquet을 읽어 (closes, volumes) 반환."""
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
    print(f"Loaded {len(close_df.columns)} symbols, {len(close_df)} rows ({close_df.index[0]} ~ {close_df.index[-1]})")
    return close_df, vol_df


# === 시그널 생성 ===


def generate_signals(
    closes: pd.DataFrame,
    lookback: int = 20,
    top_n: int = 3,
    bottom_n: int = 3,
    rebal_days: int = 7,
) -> pd.DataFrame:
    """횡단면 모멘텀 시그널.

    모든 자산의 N일 수익률을 랭킹:
    - 상위 top_n → 롱 (+1)
    - 하위 bottom_n → 숏 (-1)
    - 절대 모멘텀 필터 없음 → 항상 롱+숏 동시 보유 (시장 중립)
    """
    momentum = closes.pct_change(lookback)
    signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    rebal_dates = closes.index[lookback::rebal_days]

    for date in rebal_dates:
        if date not in momentum.index:
            continue

        mom = momentum.loc[date].dropna()
        if len(mom) < top_n + bottom_n:
            continue

        # 상위 N: 롱
        top = mom.nlargest(top_n).index
        signals.loc[date, top] = 1.0

        # 하위 N: 숏
        bottom = mom.nsmallest(bottom_n).index
        signals.loc[date, bottom] = -1.0

    # 리밸런싱 사이에는 이전 시그널 유지
    signals = signals.replace(0, np.nan)
    for date in rebal_dates:
        if date in signals.index:
            signals.loc[date] = signals.loc[date].fillna(0)
    signals = signals.ffill().fillna(0)

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
    vol_target: float | None = 0.20,
    vol_lookback: int = 20,
) -> BacktestResult:
    daily_returns = closes.pct_change().fillna(0)

    # 균등 배분: 롱+숏 각각 1/n
    n_positions = signals.abs().sum(axis=1).replace(0, 1)
    weights = signals.div(n_positions, axis=0)

    # 변동성 타겟팅
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

    return BacktestResult(
        equity=equity,
        returns=port_returns,
        positions=weights,
        trades=trades,
        metrics=compute_metrics(port_returns, equity),
    )


# === 메트릭 ===


def compute_metrics(returns: pd.Series, equity: pd.Series) -> dict[str, float]:
    days = len(returns)
    years = days / 365

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1
    vol = returns.std() * np.sqrt(365)
    sharpe = (returns.mean() * 365) / max(vol, 1e-8)

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
        "Days": days,
    }


# === 시각화 ===


def plot_equity(result: BacktestResult, title: str = "Cross-Sectional Momentum") -> None:
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
    ax.set_title("Underwater", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    n_pos = result.positions.abs().gt(0).sum(axis=1)
    ax.bar(n_pos.index, n_pos.values, width=1.5, alpha=0.5, color="steelblue")
    ax.set_ylabel("# Positions")
    ax.set_title("Active Positions", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "cross_sectional_momentum_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: research/cross_sectional_momentum_result.png")


def print_metrics(result: BacktestResult, params: dict) -> None:
    print("\n" + "=" * 60)
    print("CROSS-SECTIONAL MOMENTUM BACKTEST RESULT")
    print("=" * 60)
    print(f"Parameters: {params}")
    print(f"Trades: {result.trades}")
    print("-" * 40)
    for k, v in result.metrics.items():
        if isinstance(v, float):
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


def parameter_sweep(closes: pd.DataFrame) -> pd.DataFrame:
    results = []
    lookbacks = [10, 20, 30, 60]
    top_ns = [2, 3, 4]
    rebal_days_list = [3, 7, 14]
    vol_targets = [None, 0.15, 0.20, 0.30]

    total = len(lookbacks) * len(top_ns) * len(rebal_days_list) * len(vol_targets)
    print(f"\nRunning parameter sweep: {total} combinations")

    for lb in lookbacks:
        for tn in top_ns:
            for rd in rebal_days_list:
                sigs = generate_signals(closes, lookback=lb, top_n=tn, bottom_n=tn, rebal_days=rd)
                for vt in vol_targets:
                    res = backtest(closes, sigs, vol_target=vt)
                    results.append({
                        "lookback": lb,
                        "top_n": tn,
                        "rebal_days": rd,
                        "vol_target": vt or 0,
                        "Sharpe": res.metrics["Sharpe"],
                        "CAGR": res.metrics["CAGR"],
                        "MDD": res.metrics["MDD"],
                        "Calmar": res.metrics["Calmar"],
                        "Trades": res.trades,
                    })

    df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
    print("\n--- Top 15 by Sharpe ---")
    print(df.head(15).to_string(index=False, float_format="{:.4f}".format))
    print("\n--- Top 15 by Calmar (Sharpe >= 0.5) ---")
    filtered = df[df["Sharpe"] >= 0.5].sort_values("Calmar", ascending=False)
    print(filtered.head(15).to_string(index=False, float_format="{:.4f}".format))
    return df


# === 메인 ===


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Sectional Momentum backtest")
    parser.add_argument("--lookback", type=int, default=20, help="Momentum lookback period")
    parser.add_argument("--top-n", type=int, default=3, help="Number of long/short positions")
    parser.add_argument("--rebal-days", type=int, default=7, help="Rebalancing frequency (days)")
    parser.add_argument("--vol-target", type=float, default=0.20, help="Vol target (0 to disable)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--interval", default="1d", help="OHLCV interval")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot")
    args = parser.parse_args()

    closes, volumes = load_data(args.interval)
    vt = args.vol_target if args.vol_target > 0 else None

    if args.sweep:
        parameter_sweep(closes)
        return

    params = {
        "lookback": args.lookback,
        "top_n": args.top_n,
        "rebal_days": args.rebal_days,
        "vol_target": vt,
    }

    signals = generate_signals(closes, args.lookback, args.top_n, args.top_n, args.rebal_days)
    result = backtest(closes, signals, vol_target=vt)
    print_metrics(result, params)

    if not args.no_plot:
        plot_equity(result, f"XS Momentum (lb={args.lookback}, top={args.top_n}, rebal={args.rebal_days}d)")


if __name__ == "__main__":
    main()
