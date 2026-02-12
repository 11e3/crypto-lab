"""Trend Following + Volatility Targeting 리서치 스크립트 (바이낸스 선물).

EMA 크로스 기반 추세추종 + 변동성 타겟팅으로 레버리지 자동 조절.
- 빠른 EMA > 느린 EMA → 롱
- 빠른 EMA < 느린 EMA → 숏
- 포지션 크기 = 목표변동성 / 실현변동성 (최대 3x 캡)

결과: ✅ PASS — MDD 최저
- Best: Sharpe 1.10, CAGR +28%, MDD -22% (6.4yr)
- Params: EMA 5/20, vol_target=0.20
- 듀얼 모멘텀 대비 CAGR 낮지만 MDD 관리 우수

Usage:
    python research/fetch_data.py                      # 먼저 데이터 수집
    python research/trend_following.py                  # 기본 파라미터
    python research/trend_following.py --sweep          # 파라미터 스윕
    python research/trend_following.py --ema 10 40 --vol-target 0.20
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
    ema_fast: int = 20,
    ema_slow: int = 60,
) -> pd.DataFrame:
    """EMA 크로스 시그널. 각 자산 독립적으로 롱/숏.

    Returns:
        DataFrame: +1 (롱), -1 (숏). 모든 자산에 항상 포지션.
    """
    signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    for col in closes.columns:
        fast = closes[col].ewm(span=ema_fast, adjust=False).mean()
        slow = closes[col].ewm(span=ema_slow, adjust=False).mean()
        signals[col] = np.where(fast > slow, 1.0, -1.0)

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
    vol_target: float = 0.20,
    vol_lookback: int = 20,
    max_leverage: float = 3.0,
    equal_weight: bool = True,
) -> BacktestResult:
    """Trend following 백테스트.

    - 각 자산에 항상 롱 or 숏 포지션
    - equal_weight: 모든 자산 균등 배분 (1/n)
    - vol_target: 포트폴리오 변동성 타겟 (연율화)
    """
    daily_returns = closes.pct_change().fillna(0)
    n_assets = len(closes.columns)

    # 균등 배분: 각 자산에 1/n
    if equal_weight:
        weights = signals / n_assets
    else:
        weights = signals.div(signals.abs().sum(axis=1), axis=0)

    # 변동성 타겟팅
    port_ret = (weights.shift(1) * daily_returns).sum(axis=1)
    realized_vol = port_ret.rolling(vol_lookback).std() * np.sqrt(365)
    vol_scalar = (vol_target / realized_vol.clip(lower=0.01)).clip(upper=max_leverage)
    weights = weights.multiply(vol_scalar, axis=0)

    # 거래 비용
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    cost = turnover * COST_PER_LEG

    # 포트폴리오 수익률
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
    days = len(returns)
    years = days / 365

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1
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
    }


# === 시각화 ===


def plot_equity(result: BacktestResult, title: str = "Trend Following") -> None:
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

    # 평균 레버리지
    ax = axes[2]
    avg_lev = result.positions.abs().sum(axis=1)
    ax.plot(avg_lev.index, avg_lev.values, linewidth=0.8, color="steelblue")
    ax.set_ylabel("Leverage")
    ax.set_title("Total Leverage", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "trend_following_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: research/trend_following_result.png")


def print_metrics(result: BacktestResult, params: dict) -> None:
    print("\n" + "=" * 60)
    print("TREND FOLLOWING BACKTEST RESULT")
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
    ema_pairs = [(5, 20), (10, 40), (20, 60), (20, 100), (30, 90), (50, 200)]
    vol_targets = [0.10, 0.15, 0.20, 0.30, 0.40]
    max_leverages = [2.0, 3.0]

    total = len(ema_pairs) * len(vol_targets) * len(max_leverages)
    print(f"\nRunning parameter sweep: {total} combinations")

    for ef, es in ema_pairs:
        sigs = generate_signals(closes, ef, es)
        for vt in vol_targets:
            for ml in max_leverages:
                res = backtest(closes, sigs, vol_target=vt, max_leverage=ml)
                results.append(
                    {
                        "ema_fast": ef,
                        "ema_slow": es,
                        "vol_target": vt,
                        "max_lev": ml,
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
    parser = argparse.ArgumentParser(description="Trend Following backtest")
    parser.add_argument("--ema", nargs=2, type=int, default=[20, 60], help="Fast/slow EMA periods")
    parser.add_argument("--vol-target", type=float, default=0.20, help="Annualized vol target")
    parser.add_argument("--max-leverage", type=float, default=3.0, help="Max leverage cap")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--interval", default="1d", help="Data interval")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot")
    args = parser.parse_args()

    closes, volumes = load_data(args.interval)

    if args.sweep:
        parameter_sweep(closes)
        return

    ema_fast, ema_slow = args.ema
    params = {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "vol_target": args.vol_target,
        "max_leverage": args.max_leverage,
    }

    signals = generate_signals(closes, ema_fast, ema_slow)
    result = backtest(closes, signals, vol_target=args.vol_target, max_leverage=args.max_leverage)
    print_metrics(result, params)

    if not args.no_plot:
        plot_equity(result, f"Trend Following (EMA {ema_fast}/{ema_slow}, vol={args.vol_target})")


if __name__ == "__main__":
    main()
