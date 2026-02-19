"""전략 포트폴리오 조합 테스트.

개별 전략의 일별 수익률을 equal-weight / risk-parity로 합산하여
분산 효과 확인.

Usage:
    python research/portfolio_combo.py
    python research/portfolio_combo.py --vol-target 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

TAKER_FEE = 0.0005
SLIPPAGE = 0.0003
COST_PER_LEG = TAKER_FEE + SLIPPAGE


def load_data(interval: str = "1d"):
    """Full OHLCV load."""
    files = sorted(DATA_DIR.glob(f"*_{interval}.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR}")

    closes_d, highs_d, lows_d, volumes_d = {}, {}, {}, {}
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
    return close_df, high_df, low_df, vol_df


def compute_metrics(returns: pd.Series) -> dict[str, float]:
    """핵심 성과 지표."""
    equity = (1 + returns).cumprod()
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

    return {
        "CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MDD": mdd, "Calmar": calmar,
    }


def run_strategy_returns(
    name: str,
    closes: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    volumes: pd.DataFrame,
    vol_target: float | None = None,
) -> pd.Series:
    """각 전략의 best config로 일별 수익률 생성."""

    if name == "dual_momentum":
        from research.dual_momentum import backtest, generate_signals
        sigs = generate_signals(closes, lookback_fast=10, lookback_slow=30,
                                top_n=2, rebal_days=3, direction="long")
        res = backtest(closes, sigs, vol_target=0.25 if vol_target is None else vol_target)
        return res.returns

    elif name == "low_vol_trend":
        from research.low_vol_trend import backtest, generate_signals
        sigs = generate_signals(closes, highs, lows,
                                atr_period=20, high_lookback=80, top_n=3,
                                rebal_days=3, direction="long", vol_weight=0.3)
        res = backtest(closes, sigs, vol_target=vol_target)
        return res.returns

    elif name == "volume_momentum":
        from research.volume_momentum import backtest, generate_signals
        sigs = generate_signals(closes, volumes,
                                mom_lookback=5, vol_ma_short=10, vol_ma_long=40,
                                top_n=2, rebal_days=7, direction="long", mom_weight=0.7)
        res = backtest(closes, sigs, vol_target=vol_target)
        return res.returns

    elif name == "short_high_vol":
        from research.short_high_vol import backtest, generate_signals
        sigs = generate_signals(closes, highs, lows,
                                atr_period=14, vol_metric="natr",
                                top_n=2, rebal_days=14, direction="long")
        res = backtest(closes, sigs, vol_target=0.15 if vol_target is None else vol_target)
        return res.returns

    raise ValueError(f"Unknown strategy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio combination test")
    parser.add_argument("--vol-target", type=float, default=None,
                        help="Override vol target for all strategies")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    closes, highs, lows, volumes = load_data()
    print(f"Loaded {len(closes.columns)} symbols, {len(closes)} rows\n")

    strategies = ["dual_momentum", "low_vol_trend", "volume_momentum", "short_high_vol"]

    # === 개별 전략 수익률 ===
    all_returns: dict[str, pd.Series] = {}
    for name in strategies:
        ret = run_strategy_returns(name, closes, highs, lows, volumes, args.vol_target)
        all_returns[name] = ret
        m = compute_metrics(ret)
        pass_s = "PASS" if m["Sharpe"] >= 1.0 else "FAIL"
        pass_m = "PASS" if m["MDD"] >= -0.30 else "FAIL"
        print(f"  {name:<20s}  Sharpe {m['Sharpe']:6.2f} [{pass_s}]  "
              f"CAGR {m['CAGR']:+7.2%}  MDD {m['MDD']:+7.2%} [{pass_m}]  "
              f"Calmar {m['Calmar']:6.2f}")

    # === 공통 날짜 정렬 ===
    ret_df = pd.DataFrame(all_returns)
    ret_df = ret_df.dropna(how="any")
    print(f"\nCommon period: {len(ret_df)} days ({ret_df.index[0].date()} ~ {ret_df.index[-1].date()})")

    # === 조합 1: Equal Weight ===
    print("\n" + "=" * 70)
    print("PORTFOLIO COMBINATIONS")
    print("=" * 70)

    combos = {
        "EW (all 4)": ret_df.mean(axis=1),
        "EW (top 3: dm+lvt+vm)": ret_df[["dual_momentum", "low_vol_trend", "volume_momentum"]].mean(axis=1),
        "EW (dm+shv)": ret_df[["dual_momentum", "short_high_vol"]].mean(axis=1),
        "EW (lvt+vm)": ret_df[["low_vol_trend", "volume_momentum"]].mean(axis=1),
        "EW (dm+lvt)": ret_df[["dual_momentum", "low_vol_trend"]].mean(axis=1),
    }

    # === 조합 2: Inverse-Vol Weight ===
    vols = ret_df.rolling(60).std().iloc[-1]
    inv_vol_w = (1 / vols) / (1 / vols).sum()
    combos["InvVol (all 4)"] = (ret_df * inv_vol_w).sum(axis=1)

    # === 조합 3: Risk Parity (equal risk contribution approx) ===
    # Simple: scale each to same vol, then equal weight
    target_vol = 0.15 / np.sqrt(365)  # 15% annualized target per strategy
    scaled = pd.DataFrame()
    for col in ret_df.columns:
        realized = ret_df[col].rolling(60).std()
        scale = (target_vol / realized.clip(lower=1e-6)).clip(upper=5.0)
        scaled[col] = ret_df[col] * scale
    rp_ret = scaled.mean(axis=1).dropna()
    combos["RiskParity (all 4)"] = rp_ret

    # 결과 출력
    print(f"\n{'Combo':<28s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>8s} {'Vol':>7s}")
    print("-" * 70)
    for name, ret in combos.items():
        m = compute_metrics(ret)
        pass_s = "o" if m["Sharpe"] >= 1.0 else "x"
        pass_m = "o" if m["MDD"] >= -0.30 else "x"
        print(f"  {name:<26s} {m['Sharpe']:6.2f}[{pass_s}] {m['CAGR']:+7.2%} "
              f"{m['MDD']:+7.2%}[{pass_m}] {m['Calmar']:7.2f} {m['Vol']:6.2%}")

    # === 상관관계 ===
    print(f"\n--- Strategy Return Correlations ---")
    corr = ret_df.corr()
    print(corr.round(3).to_string())

    # === 연도별 수익률 (best combo) ===
    best_name = max(combos, key=lambda k: compute_metrics(combos[k])["Sharpe"])
    best_ret = combos[best_name]
    best_equity = (1 + best_ret).cumprod()
    print(f"\n--- Yearly Returns: {best_name} ---")
    yearly = best_ret.groupby(best_ret.index.year).apply(lambda r: (1 + r).prod() - 1)
    for year, ret in yearly.items():
        print(f"  {year}: {ret:+.2%}")

    # === 시각화 ===
    if not args.no_plot:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

        ax = axes[0]
        for name in strategies:
            eq = (1 + all_returns[name].reindex(ret_df.index).fillna(0)).cumprod()
            ax.plot(eq.index, eq.values, alpha=0.5, linewidth=0.8, label=name)
        # Best combo
        ax.plot(best_equity.index, best_equity.values, linewidth=2.0,
                color="black", label=f"{best_name}")
        ax.set_title("Individual Strategies vs Portfolio", fontsize=13)
        ax.set_ylabel("Equity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        ax = axes[1]
        peak = best_equity.cummax()
        dd = (best_equity - peak) / peak
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
        ax.set_ylabel("Drawdown")
        ax.set_title(f"Underwater: {best_name}", fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        print(f"\n--- Strategy Correlations (rolling 60d) ---")
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                rc = ret_df[s1].rolling(60).corr(ret_df[s2])
                ax.plot(rc.index, rc.values, alpha=0.5, linewidth=0.8,
                        label=f"{s1[:3]}-{s2[:3]}")
        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_title("Rolling 60d Pairwise Correlations", fontsize=11)
        ax.set_ylabel("Correlation")
        ax.legend(fontsize=6, ncol=3)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = Path(__file__).parent / "portfolio_combo_result.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
