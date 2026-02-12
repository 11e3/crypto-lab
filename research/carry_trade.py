"""Carry Trade (펀딩비 수취) 리서치 스크립트 (바이낸스 선물).

펀딩비 수취 전략: 양수 펀딩비 종목 숏, 음수 펀딩비 종목 롱.
- Funding Rate MR과 반대 철학: 극단값 회귀가 아닌 지속적 캐리 수취
- 펀딩비가 양수 = 롱이 숏에게 지불 → 숏 진입으로 펀딩비 수취
- 횡단면 랭킹: 펀딩비 크기 순으로 상위 N 숏, 하위 N 롱
- 리밸런싱 주기별 시그널 갱신

결과: ⚠️ Sharpe 경계선, MDD FAIL
- Best: Sharpe 1.00, CAGR +45%, MDD -56% (vol target 없음)
  Params: top_n=4, smoothing=3, rebal=7d
- Vol target 적용 시: Sharpe 0.97, MDD -20% (거의 PASS)
  Params: top_n=2, smoothing=3, rebal=7d, vol=0.15
- 모멘텀과 다른 수익원이라 분산 효과 잠재력 있음

Usage:
    python research/funding_rate_mr.py --fetch          # 먼저 펀딩비 수집
    python research/fetch_data.py                       # OHLCV 수집
    python research/carry_trade.py                      # 기본 파라미터
    python research/carry_trade.py --sweep              # 파라미터 스윕
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


def load_data(interval: str = "1d") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(closes, volumes, funding_rates) 반환. 펀딩비는 일별 합산."""
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

    # 펀딩비: 일별 합산 (8h × 3)
    funding: dict[str, pd.Series] = {}
    for symbol in close_df.columns:
        fpath = DATA_DIR / f"{symbol}_funding.parquet"
        if fpath.exists():
            fdf = pd.read_parquet(fpath)
            daily = fdf["funding_rate"].resample("1D").sum()
            funding[symbol] = daily

    fund_df = pd.DataFrame(funding).reindex(close_df.index).fillna(0)

    available = [c for c in close_df.columns if c in fund_df.columns and fund_df[c].abs().sum() > 0]
    print(f"Loaded {len(available)} symbols with funding data, {len(close_df)} rows")

    return close_df[available], vol_df[available], fund_df[available]


# === 시그널 생성 ===


def generate_signals(
    funding: pd.DataFrame,
    top_n: int = 3,
    bottom_n: int = 3,
    smoothing: int = 7,
    rebal_days: int = 3,
    min_funding: float = 0.0001,
) -> pd.DataFrame:
    """캐리 트레이드 시그널.

    펀딩비 이동평균 기반 횡단면 랭킹:
    - 펀딩비 높은 top_n 종목 → 숏 (캐리 수취)
    - 펀딩비 낮은 bottom_n 종목 → 롱 (캐리 수취 or 비용 최소화)
    - min_funding: 최소 펀딩비 크기 (노이즈 필터)
    """
    # 이동평균으로 펀딩비 스무딩
    smooth_funding = funding.rolling(smoothing, min_periods=1).mean()

    signals = pd.DataFrame(0.0, index=funding.index, columns=funding.columns)
    rebal_dates = funding.index[smoothing::rebal_days]

    for date in rebal_dates:
        if date not in smooth_funding.index:
            continue

        fr = smooth_funding.loc[date].dropna()
        if len(fr) < top_n + bottom_n:
            continue

        # 펀딩비 높은 종목: 숏 (캐리 수취)
        top = fr.nlargest(top_n)
        for asset in top.index:
            if abs(top[asset]) >= min_funding:
                signals.loc[date, asset] = -1.0  # 숏

        # 펀딩비 낮은 종목: 롱 (음수 펀딩비면 캐리 수취)
        bottom = fr.nsmallest(bottom_n)
        for asset in bottom.index:
            if abs(bottom[asset]) >= min_funding:
                signals.loc[date, asset] = 1.0  # 롱

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
    vol_target: float | None = 0.15,
    vol_lookback: int = 20,
) -> BacktestResult:
    daily_returns = closes.pct_change().fillna(0)

    # 균등 배분
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


def plot_equity(result: BacktestResult, title: str = "Carry Trade") -> None:
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
    plt.savefig(Path(__file__).parent / "carry_trade_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: research/carry_trade_result.png")


def print_metrics(result: BacktestResult, params: dict) -> None:
    print("\n" + "=" * 60)
    print("CARRY TRADE BACKTEST RESULT")
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


def parameter_sweep(closes: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    results = []
    top_ns = [2, 3, 4]
    smoothings = [3, 7, 14, 30]
    rebal_days_list = [1, 3, 7]
    vol_targets = [None, 0.10, 0.15, 0.25]

    total = len(top_ns) * len(smoothings) * len(rebal_days_list) * len(vol_targets)
    print(f"\nRunning parameter sweep: {total} combinations")

    for tn in top_ns:
        for sm in smoothings:
            for rd in rebal_days_list:
                sigs = generate_signals(funding, top_n=tn, bottom_n=tn, smoothing=sm, rebal_days=rd)
                for vt in vol_targets:
                    res = backtest(closes, sigs, vol_target=vt)
                    results.append({
                        "top_n": tn,
                        "smoothing": sm,
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
    parser = argparse.ArgumentParser(description="Carry Trade (funding rate harvesting) backtest")
    parser.add_argument("--top-n", type=int, default=3, help="Number of short/long positions")
    parser.add_argument("--smoothing", type=int, default=7, help="Funding rate smoothing window")
    parser.add_argument("--rebal-days", type=int, default=3, help="Rebalancing frequency")
    parser.add_argument("--vol-target", type=float, default=0.15, help="Vol target (0 to disable)")
    parser.add_argument("--min-funding", type=float, default=0.0001, help="Min funding rate for entry")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--interval", default="1d", help="OHLCV interval")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot")
    args = parser.parse_args()

    closes, volumes, funding = load_data(args.interval)

    if funding.abs().sum().sum() == 0:
        print("No funding rate data found. Run funding_rate_mr.py --fetch first.")
        return

    vt = args.vol_target if args.vol_target > 0 else None

    if args.sweep:
        parameter_sweep(closes, funding)
        return

    params = {
        "top_n": args.top_n,
        "smoothing": args.smoothing,
        "rebal_days": args.rebal_days,
        "vol_target": vt,
        "min_funding": args.min_funding,
    }

    signals = generate_signals(
        funding, top_n=args.top_n, bottom_n=args.top_n,
        smoothing=args.smoothing, rebal_days=args.rebal_days,
        min_funding=args.min_funding,
    )
    result = backtest(closes, signals, vol_target=vt)
    print_metrics(result, params)

    if not args.no_plot:
        plot_equity(result, f"Carry Trade (top={args.top_n}, sm={args.smoothing}, rebal={args.rebal_days}d)")


if __name__ == "__main__":
    main()
