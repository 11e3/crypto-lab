"""Dual Momentum 전략 리서치 스크립트 (바이낸스 선물).

절대 모멘텀 + 상대 모멘텀 결합.
- 절대: 룩백 기간 수익률 > 0 이면 롱 가능
- 상대: 여러 자산 중 상위 N개만 진입
- 양방향: 절대 모멘텀 < 0 이면 숏 가능 (선물이니까)

결과: ✅ PASS — 전체 전략 중 최고 성과
- Best: Sharpe 1.43, CAGR +52%, MDD -28% (6.4yr)
- Params: lb=10/30, top_n=2, rebal=3d, vol_target=0.25
- Capacity: $5.75M (1% participation, conservative)

Usage:
    python research/fetch_data.py                  # 먼저 데이터 수집
    python research/dual_momentum.py               # 백테스트 실행
    python research/dual_momentum.py --top-n 3 --lookback 20 60 --rebal-days 7
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
# turnover = sum(|weight_change|) 는 각 레그(진입/청산)를 개별 카운트하므로
# 여기에 편도 비용을 곱하면 왕복 비용이 자동 반영됨.
# 예: A→B 교체 시 turnover = |ΔA| + |ΔB| = 0.5 + 0.5 = 1.0
#     cost = 1.0 × 0.0008 = 0.08% (청산 0.04% + 진입 0.04%)
TAKER_FEE = 0.0005  # 0.05% per leg
SLIPPAGE = 0.0003  # 0.03% per leg (보수적: 알트코인 스프레드 고려)
COST_PER_LEG = TAKER_FEE + SLIPPAGE  # 0.08% per leg


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
        volumes[symbol] = df["close"] * df["volume"]  # USD 거래대금

    close_df = pd.DataFrame(closes).dropna(how="all").ffill()
    vol_df = pd.DataFrame(volumes).reindex(close_df.index).fillna(0)
    print(f"Loaded {len(close_df.columns)} symbols, {len(close_df)} rows ({close_df.index[0]} ~ {close_df.index[-1]})")
    return close_df, vol_df


# === 시그널 생성 ===


def compute_momentum(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """룩백 기간 수익률."""
    return closes.pct_change(lookback)


def generate_signals(
    closes: pd.DataFrame,
    lookback_short: int = 20,
    lookback_long: int = 60,
    top_n: int = 3,
    rebal_days: int = 7,
) -> pd.DataFrame:
    """듀얼 모멘텀 시그널 생성.

    Returns:
        DataFrame with same shape as closes.
        값: +1 (롱), -1 (숏), 0 (플랫)
    """
    mom_short = compute_momentum(closes, lookback_short)
    mom_long = compute_momentum(closes, lookback_long)

    signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    # 리밸런싱 날짜만 시그널 계산
    rebal_dates = closes.index[::rebal_days]

    for date in rebal_dates:
        if date not in mom_short.index or date not in mom_long.index:
            continue

        ms = mom_short.loc[date]
        ml = mom_long.loc[date]

        # 절대 모멘텀: 두 룩백 모두 양수 → 롱 후보
        long_candidates = (ms > 0) & (ml > 0)
        short_candidates = (ms < 0) & (ml < 0)

        # 상대 모멘텀: 합산 스코어로 랭킹
        combined_score = ms + ml

        # 롱 상위 N개
        long_scores = combined_score.where(long_candidates).dropna()
        if len(long_scores) > 0:
            top_longs = long_scores.nlargest(min(top_n, len(long_scores))).index
            signals.loc[date, top_longs] = 1.0

        # 숏 하위 N개
        short_scores = combined_score.where(short_candidates).dropna()
        if len(short_scores) > 0:
            top_shorts = short_scores.nsmallest(min(top_n, len(short_scores))).index
            signals.loc[date, top_shorts] = -1.0

    # 리밸런싱 사이에는 이전 시그널 유지
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
    """시그널 기반 포트폴리오 백테스트.

    - 균등 배분 (시그널 있는 자산끼리 1/n)
    - 거래 비용 적용
    - vol_target: 변동성 타겟팅 (e.g., 0.15 = 연 15%). None이면 비활성.
    """
    daily_returns = closes.pct_change().fillna(0)

    # 포지션 가중치: 시그널 있는 자산끼리 균등 배분
    n_positions = signals.abs().sum(axis=1).replace(0, 1)  # 0 방지
    weights = signals.div(n_positions, axis=0) * leverage

    # 변동성 타겟팅: 실현 변동성 대비 포지션 크기 조절
    if vol_target is not None:
        port_ret = (weights.shift(1) * daily_returns).sum(axis=1)
        realized_vol = port_ret.rolling(vol_lookback).std() * np.sqrt(365)
        daily_vol_target = vol_target
        vol_scalar = (daily_vol_target / realized_vol.clip(lower=0.01)).clip(upper=3.0)
        weights = weights.multiply(vol_scalar, axis=0)

    # 거래 비용: 가중치 변화량 × 편도 비용
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    cost = turnover * COST_PER_LEG

    # 포트폴리오 수익률
    port_returns = (weights.shift(1) * daily_returns).sum(axis=1) - cost
    port_returns = port_returns.iloc[1:]  # 첫 행 제거

    equity = (1 + port_returns).cumprod()

    # 거래 횟수 (가중치 변화 발생 시)
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
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    vol = returns.std() * np.sqrt(365)
    sharpe = (returns.mean() * 365) / max(vol, 1e-8)

    # MDD
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = drawdown.min()

    # Calmar
    calmar = cagr / max(abs(mdd), 1e-8)

    # Win rate
    win_rate = (returns > 0).sum() / max((returns != 0).sum(), 1)

    # Profit factor
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


# === Capacity 계산 ===


def estimate_capacity(
    positions: pd.DataFrame,
    volume_usd: pd.DataFrame,
    max_participation: float = 0.01,
) -> dict[str, float]:
    """전략이 소화할 수 있는 최대 자본 추정.

    Args:
        positions: 가중치 DataFrame (backtest 결과)
        volume_usd: 일별 USD 거래대금 DataFrame
        max_participation: 일일 거래대금 중 최대 참여 비율 (기본 1%)

    Returns:
        capacity 관련 메트릭 dict
    """
    # 리밸런싱 시점: 가중치 변화가 있는 날
    weight_changes = positions.diff().abs()
    rebal_mask = weight_changes.sum(axis=1) > 0

    if rebal_mask.sum() == 0:
        return {"capacity_usd": 0, "min_daily_vol_usd": 0, "bottleneck": "N/A"}

    # 리밸런싱 시점에서 각 자산별로 필요한 거래대금 비율
    # 필요 거래대금 = |weight_change| × capital
    # 가용 거래대금 = volume_usd × max_participation
    # capital ≤ volume_usd × max_participation / |weight_change|

    capacities = []
    for date in positions.index[rebal_mask]:
        wc = weight_changes.loc[date]
        vol = volume_usd.loc[date] if date in volume_usd.index else pd.Series(0, index=wc.index)

        for asset in wc.index:
            if wc[asset] > 0.001:  # 의미있는 가중치 변화만
                available = vol.get(asset, 0) * max_participation
                cap = available / wc[asset] if wc[asset] > 0 else float("inf")
                capacities.append(
                    {"date": date, "asset": asset, "weight_change": wc[asset], "volume_usd": vol.get(asset, 0), "capacity": cap}
                )

    if not capacities:
        return {"capacity_usd": 0, "min_daily_vol_usd": 0, "bottleneck": "N/A"}

    cap_df = pd.DataFrame(capacities)

    # 보수적 추정: 5th percentile (최악 5% 시점 기준)
    capacity_p5 = cap_df["capacity"].quantile(0.05)
    # 중앙값 추정
    capacity_median = cap_df["capacity"].quantile(0.50)

    # 병목 자산: capacity가 가장 자주 낮은 자산
    bottleneck = cap_df.groupby("asset")["capacity"].median().idxmin()

    # 최소 일일 거래대금 (리밸런싱일 기준)
    min_vol = volume_usd.loc[rebal_mask].sum(axis=1).min()
    median_vol = volume_usd.loc[rebal_mask].sum(axis=1).median()

    return {
        "capacity_conservative_usd": capacity_p5,
        "capacity_median_usd": capacity_median,
        "bottleneck_asset": bottleneck,
        "min_total_daily_vol_usd": min_vol,
        "median_total_daily_vol_usd": median_vol,
        "participation_rate": max_participation,
    }


# === 시각화 ===


def plot_equity(result: BacktestResult, title: str = "Dual Momentum") -> None:
    """에퀴티 커브 + 드로다운."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    # 에퀴티 커브
    ax = axes[0]
    ax.plot(result.equity.index, result.equity.values, linewidth=1.2)
    ax.set_title(f"{title} — Equity Curve", fontsize=13)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    # 드로다운
    ax = axes[1]
    peak = result.equity.cummax()
    dd = (result.equity - peak) / peak
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater", fontsize=11)
    ax.grid(True, alpha=0.3)

    # 포지션 수
    ax = axes[2]
    n_pos = result.positions.abs().gt(0).sum(axis=1)
    ax.bar(n_pos.index, n_pos.values, width=1.5, alpha=0.5, color="steelblue")
    ax.set_ylabel("# Positions")
    ax.set_title("Active Positions", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "dual_momentum_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: research/dual_momentum_result.png")


def print_metrics(result: BacktestResult, params: dict) -> None:
    """결과 출력."""
    print("\n" + "=" * 60)
    print("DUAL MOMENTUM BACKTEST RESULT")
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

    # 통과 기준 체크
    sharpe = result.metrics["Sharpe"]
    mdd = result.metrics["MDD"]
    print(f"\n  Sharpe >= 1.0 ? {'PASS' if sharpe >= 1.0 else 'FAIL'} ({sharpe:.2f})")
    print(f"  MDD >= -30%   ? {'PASS' if mdd >= -0.30 else 'FAIL'} ({mdd:.2%})")


# === 파라미터 스윕 ===


def parameter_sweep(closes: pd.DataFrame) -> pd.DataFrame:
    """주요 파라미터 조합 스윕."""
    results = []
    lookbacks = [(10, 30), (20, 60), (30, 90), (20, 120)]
    top_ns = [2, 3, 5]
    rebal_days_list = [3, 7, 14]
    vol_targets = [None, 0.15, 0.25, 0.40]

    total = len(lookbacks) * len(top_ns) * len(rebal_days_list) * len(vol_targets)
    print(f"\nRunning parameter sweep: {total} combinations")

    for lb_s, lb_l in lookbacks:
        for top_n in top_ns:
            for rebal in rebal_days_list:
                sigs = generate_signals(closes, lb_s, lb_l, top_n, rebal)
                for vt in vol_targets:
                    res = backtest(closes, sigs, vol_target=vt)
                    results.append(
                        {
                            "lb_s": lb_s,
                            "lb_l": lb_l,
                            "top_n": top_n,
                            "rebal": rebal,
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
    print("\n--- Top 15 by Calmar (Sharpe >= 0.8) ---")
    filtered = df[df["Sharpe"] >= 0.8].sort_values("Calmar", ascending=False)
    print(filtered.head(15).to_string(index=False, float_format="{:.4f}".format))
    return df


# === 메인 ===


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual Momentum backtest")
    parser.add_argument("--lookback", nargs=2, type=int, default=[20, 60], help="Short/long lookback periods")
    parser.add_argument("--top-n", type=int, default=3, help="Top N assets to hold")
    parser.add_argument("--rebal-days", type=int, default=7, help="Rebalancing frequency (days)")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier")
    parser.add_argument("--vol-target", type=float, default=None, help="Annualized vol target (e.g., 0.15)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--interval", default="1d", help="Data interval")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot")
    args = parser.parse_args()

    closes, volumes = load_data(args.interval)

    if args.sweep:
        parameter_sweep(closes)
        return

    lb_short, lb_long = args.lookback
    params = {
        "lookback_short": lb_short,
        "lookback_long": lb_long,
        "top_n": args.top_n,
        "rebal_days": args.rebal_days,
        "leverage": args.leverage,
    }

    signals = generate_signals(closes, lb_short, lb_long, args.top_n, args.rebal_days)
    result = backtest(closes, signals, args.leverage, vol_target=args.vol_target)
    params["vol_target"] = args.vol_target
    print_metrics(result, params)

    # Capacity 분석
    cap = estimate_capacity(result.positions, volumes)
    print("\n--- CAPACITY ANALYSIS (1% participation) ---")
    print(f"  Conservative (5th pct): ${cap['capacity_conservative_usd']:,.0f}")
    print(f"  Median estimate:        ${cap['capacity_median_usd']:,.0f}")
    print(f"  Bottleneck asset:       {cap['bottleneck_asset']}")
    print(f"  Min daily vol (rebal):  ${cap['min_total_daily_vol_usd']:,.0f}")
    print(f"  Median daily vol:       ${cap['median_total_daily_vol_usd']:,.0f}")

    if not args.no_plot:
        plot_equity(result, f"Dual Momentum (lb={lb_short}/{lb_long}, top={args.top_n}, rebal={args.rebal_days}d)")


if __name__ == "__main__":
    main()
