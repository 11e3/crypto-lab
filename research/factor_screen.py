"""Factor Screening — 바이낸스 선물 데이터 기반 예측력 분석.

~30개 factor의 Information Coefficient(IC)를 측정하여
어디에 edge가 있는지 데이터 기반으로 발견.

IC = spearman_corr(factor[t], forward_return[t+horizon])
- IC > 0: factor가 높을수록 미래 수익률 높음 (momentum-like)
- IC < 0: factor가 높을수록 미래 수익률 낮음 (mean-reversion-like)
- |IC| > 0.03 이상이면 통계적으로 의미 있는 수준 (crypto)

Usage:
    python research/fetch_data.py                    # 먼저 데이터 수집
    python research/factor_screen.py                 # 전체 스크린
    python research/factor_screen.py --horizon 1 3   # 특정 horizon만
    python research/factor_screen.py --top 15        # 상위 15개 출력
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).parent / "data"


# === 데이터 로드 ===


def load_data(interval: str = "1d") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """research/data/ 에서 parquet 로드 → (closes, highs_lows, volumes).

    Returns:
        closes: 종가 DataFrame (index=dates, columns=symbols)
        ohlcv: 전체 OHLCV dict (symbol → DataFrame)
        volumes: USD 거래대금 DataFrame
    """
    files = sorted(DATA_DIR.glob(f"*_{interval}.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR} for interval={interval}")

    closes: dict[str, pd.Series] = {}
    highs: dict[str, pd.Series] = {}
    lows: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}

    for f in files:
        symbol = f.stem.replace(f"_{interval}", "")
        df = pd.read_parquet(f)
        closes[symbol] = df["close"]
        highs[symbol] = df["high"]
        lows[symbol] = df["low"]
        opens[symbol] = df["open"]
        volumes[symbol] = df["close"] * df["volume"]

    close_df = pd.DataFrame(closes).dropna(how="all").ffill()
    high_df = pd.DataFrame(highs).reindex(close_df.index).ffill()
    low_df = pd.DataFrame(lows).reindex(close_df.index).ffill()
    open_df = pd.DataFrame(opens).reindex(close_df.index).ffill()
    vol_df = pd.DataFrame(volumes).reindex(close_df.index).fillna(0)

    print(f"Loaded {len(close_df.columns)} symbols, {len(close_df)} rows "
          f"({close_df.index[0].date()} ~ {close_df.index[-1].date()})")

    return close_df, high_df, low_df, open_df, vol_df


# === Factor 계산 ===


def compute_factors(
    closes: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    opens: pd.DataFrame,
    volumes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """~30개 factor를 DataFrame(index=dates, columns=symbols)으로 계산."""
    factors: dict[str, pd.DataFrame] = {}
    returns = closes.pct_change()

    # --- Momentum (6) ---
    for lb in [5, 10, 20, 40, 60, 120]:
        factors[f"momentum_{lb}d"] = closes.pct_change(lb)

    # --- Mean Reversion (4) ---
    for ma_period in [10, 20, 40]:
        ma = closes.rolling(ma_period).mean()
        std = closes.rolling(ma_period).std()
        factors[f"zscore_vs_ma{ma_period}"] = (closes - ma) / std.replace(0, np.nan)

    # RSI(14)
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    factors["rsi_14"] = 100 - 100 / (1 + rs)

    # --- Volatility (4) ---
    factors["realized_vol_20d"] = returns.rolling(20).std() * np.sqrt(365)

    # ATR(14) normalized by close
    tr = pd.concat([
        highs - lows,
        (highs - closes.shift(1)).abs(),
        (lows - closes.shift(1)).abs(),
    ]).groupby(level=0).max()
    # Rebuild as proper DataFrame matching closes shape
    true_range = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)
    for sym in closes.columns:
        hl = highs[sym] - lows[sym]
        hc = (highs[sym] - closes[sym].shift(1)).abs()
        lc = (lows[sym] - closes[sym].shift(1)).abs()
        true_range[sym] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    factors["natr_14"] = atr_14 / closes * 100

    # Vol change: current vol vs 60d ago
    vol_20 = returns.rolling(20).std()
    vol_60 = returns.rolling(60).std()
    factors["vol_change"] = vol_20 / vol_60.replace(0, np.nan)

    # Vol of vol
    factors["vol_of_vol"] = vol_20.rolling(20).std()

    # --- Volume (3) ---
    vol_5 = volumes.rolling(5).mean()
    vol_20_ma = volumes.rolling(20).mean()
    factors["volume_ratio_5_20"] = vol_5 / vol_20_ma.replace(0, np.nan)

    # Volume-price divergence: price up but volume down (or vice versa)
    price_ret_20 = closes.pct_change(20)
    vol_ret_20 = volumes.pct_change(20)
    factors["vol_price_divergence"] = price_ret_20 - vol_ret_20

    # Volume momentum
    factors["volume_momentum_10d"] = volumes.pct_change(10)

    # --- Technical (5) ---
    # Bollinger %B
    ma20 = closes.rolling(20).mean()
    std20 = closes.rolling(20).std()
    factors["bollinger_pct_b"] = (closes - (ma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)

    # MACD signal
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    factors["macd_histogram"] = macd_line - macd_signal

    # EMA cross (8/21)
    ema8 = closes.ewm(span=8, adjust=False).mean()
    ema21 = closes.ewm(span=21, adjust=False).mean()
    factors["ema_cross_8_21"] = (ema8 - ema21) / closes

    # Distance from 50d high/low
    high_50 = closes.rolling(50).max()
    low_50 = closes.rolling(50).min()
    factors["dist_from_50d_high"] = closes / high_50 - 1
    factors["dist_from_50d_low"] = closes / low_50 - 1

    # --- Cross-sectional (4) ---
    # Relative strength rank (cross-sectional percentile of 20d return)
    ret_20 = closes.pct_change(20)
    factors["relative_strength_rank"] = ret_20.rank(axis=1, pct=True)

    # Relative volume rank
    factors["relative_volume_rank"] = volumes.rank(axis=1, pct=True)

    # Cross-sectional momentum (deviation from cross-sectional mean)
    cs_mean = ret_20.mean(axis=1)
    factors["cs_momentum"] = ret_20.sub(cs_mean, axis=0)

    # Cross-sectional vol rank
    vol_rank = vol_20.rank(axis=1, pct=True)
    factors["cs_vol_rank"] = vol_rank

    # --- Microstructure (2) ---
    # High-low range ratio (normalized)
    range_ratio = (highs - lows) / closes
    factors["hl_range_ratio"] = range_ratio.rolling(5).mean()

    # Close-open ratio (intraday momentum)
    factors["close_open_ratio"] = (closes - opens) / (highs - lows).replace(0, np.nan)

    print(f"Computed {len(factors)} factors")
    return factors


# === IC 계산 ===


def compute_ic_stats(
    factors: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    horizons: list[int],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """각 factor × horizon 조합의 IC 통계 계산.

    Returns:
        ic_stats: 요약 통계 DataFrame (index=factor_name, columns=[horizon, mean_ic, ...])
        ic_series: factor별 날짜별 IC 시계열 dict
    """
    forward_returns = {}
    for h in horizons:
        forward_returns[h] = closes.pct_change(h).shift(-h)

    results = []
    ic_series: dict[str, pd.DataFrame] = {}

    for factor_name, factor_df in factors.items():
        daily_ics: dict[int, list[float]] = {h: [] for h in horizons}

        # 공통 날짜
        valid_dates = factor_df.dropna(how="all").index

        for date in valid_dates:
            for h in horizons:
                if date not in forward_returns[h].index:
                    continue

                f_vals = factor_df.loc[date].dropna()
                r_vals = forward_returns[h].loc[date].reindex(f_vals.index).dropna()

                common = f_vals.index.intersection(r_vals.index)
                if len(common) < 4:  # 최소 4개 심볼 필요
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", stats.ConstantInputWarning)
                    ic, _ = stats.spearmanr(f_vals[common], r_vals[common])
                if not np.isnan(ic):
                    daily_ics[h].append(ic)

        # factor별 IC 시계열 저장
        for h in horizons:
            ics = daily_ics[h]
            if len(ics) < 30:  # 최소 30일 데이터
                results.append({
                    "factor": factor_name,
                    "horizon": h,
                    "mean_ic": np.nan,
                    "std_ic": np.nan,
                    "ir": np.nan,
                    "t_stat": np.nan,
                    "hit_rate": np.nan,
                    "n_obs": len(ics),
                })
                continue

            ic_arr = np.array(ics)
            mean_ic = np.mean(ic_arr)
            std_ic = np.std(ic_arr, ddof=1)
            ir = mean_ic / std_ic if std_ic > 0 else 0.0
            t_stat = mean_ic / (std_ic / np.sqrt(len(ic_arr))) if std_ic > 0 else 0.0
            hit_rate = np.mean(ic_arr > 0)

            results.append({
                "factor": factor_name,
                "horizon": h,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ir": ir,
                "t_stat": t_stat,
                "hit_rate": hit_rate,
                "n_obs": len(ics),
            })

        # IC 시계열 저장 (7d horizon 기준)
        if 7 in daily_ics and len(daily_ics[7]) > 0:
            ic_series[factor_name] = pd.Series(daily_ics[7])

    ic_stats = pd.DataFrame(results)
    return ic_stats, ic_series


def compute_factor_correlations(
    factors: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Factor 간 상관관계 (중복 factor 필터용)."""
    # 각 factor의 cross-sectional rank 평균을 시계열로 변환
    factor_ts = {}
    for name, df in factors.items():
        ranked = df.rank(axis=1, pct=True)
        factor_ts[name] = ranked.mean(axis=1)

    ts_df = pd.DataFrame(factor_ts).dropna()
    return ts_df.corr(method="spearman")


# === 시각화 ===


def plot_results(
    ic_stats: pd.DataFrame,
    ic_series: dict[str, pd.DataFrame],
    top_n: int = 10,
) -> None:
    """IC heatmap + top factor rolling IC."""
    # Filter to 7d horizon for ranking
    stats_7d = ic_stats[ic_stats["horizon"] == 7].dropna(subset=["ir"])
    if stats_7d.empty:
        # Fallback to longest available horizon
        max_h = ic_stats["horizon"].max()
        stats_7d = ic_stats[ic_stats["horizon"] == max_h].dropna(subset=["ir"])

    top_factors = stats_7d.nlargest(top_n, "ir")["factor"].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1]})

    # Panel 1: IC heatmap (factor × horizon)
    ax = axes[0]
    pivot = ic_stats.pivot(index="factor", columns="horizon", values="mean_ic")
    pivot = pivot.loc[pivot.abs().max(axis=1).nlargest(top_n).index]
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{h}d" for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"Top {top_n} Factors — Mean IC by Horizon", fontsize=12)
    plt.colorbar(im, ax=ax, label="Mean IC")

    # Panel 2: Rolling IC for top factors (7d horizon)
    ax = axes[1]
    for factor_name in top_factors[:5]:
        if factor_name in ic_series:
            s = pd.Series(ic_series[factor_name])
            rolling = s.rolling(60, min_periods=30).mean()
            ax.plot(rolling.values, label=factor_name, alpha=0.8, linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_title("Rolling 60d IC (7d forward return)", fontsize=12)
    ax.set_ylabel("IC")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "factor_screen_result.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_results(ic_stats: pd.DataFrame, horizons: list[int], top_n: int) -> None:
    """콘솔 출력."""
    print("\n" + "=" * 80)
    print("FACTOR SCREENING RESULTS")
    print("=" * 80)

    for h in horizons:
        subset = ic_stats[ic_stats["horizon"] == h].dropna(subset=["ir"])
        if subset.empty:
            continue

        ranked = subset.sort_values("ir", ascending=False)
        print(f"\n--- {h}d Forward Return (Top {top_n} by IR) ---")
        print(f"{'Factor':<30} {'Mean IC':>8} {'Std IC':>8} {'IR':>8} {'t-stat':>8} {'Hit%':>6} {'N':>5}")
        print("-" * 80)

        for _, row in ranked.head(top_n).iterrows():
            ic_sign = "+" if row["mean_ic"] > 0 else ""
            print(
                f"{row['factor']:<30} "
                f"{ic_sign}{row['mean_ic']:>7.4f} "
                f"{row['std_ic']:>8.4f} "
                f"{row['ir']:>8.4f} "
                f"{row['t_stat']:>8.2f} "
                f"{row['hit_rate']:>5.1%} "
                f"{int(row['n_obs']):>5}"
            )

        # Bottom factors (mean reversion signals)
        print(f"\n--- {h}d Forward Return (Bottom {min(5, top_n)} by IR - Mean Reversion Signals) ---")
        print(f"{'Factor':<30} {'Mean IC':>8} {'Std IC':>8} {'IR':>8} {'t-stat':>8} {'Hit%':>6} {'N':>5}")
        print("-" * 80)
        for _, row in ranked.tail(min(5, top_n)).iloc[::-1].iterrows():
            ic_sign = "+" if row["mean_ic"] > 0 else ""
            print(
                f"{row['factor']:<30} "
                f"{ic_sign}{row['mean_ic']:>7.4f} "
                f"{row['std_ic']:>8.4f} "
                f"{row['ir']:>8.4f} "
                f"{row['t_stat']:>8.2f} "
                f"{row['hit_rate']:>5.1%} "
                f"{int(row['n_obs']):>5}"
            )

    print("\n" + "=" * 80)
    print("Guide:")
    print("  - IR > 0.05: meaningful predictive power (strategy candidate)")
    print("  - |t-stat| > 2.0: statistically significant")
    print("  - Hit% > 55%: directionally stable")
    print("  - Negative IC factor -> use as mean reversion signal")
    print("=" * 80)


# === 메인 ===


def main() -> None:
    parser = argparse.ArgumentParser(description="Factor screening for crypto futures")
    parser.add_argument("--interval", default="1d", help="Data interval")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 3, 7, 14],
                        help="Forward return horizons (days)")
    parser.add_argument("--top", type=int, default=10, help="Top N factors to display")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation")
    parser.add_argument("--corr", action="store_true", help="Print factor correlation matrix")
    args = parser.parse_args()

    closes, highs, lows, opens, volumes = load_data(args.interval)

    factors = compute_factors(closes, highs, lows, opens, volumes)

    print(f"\nComputing IC for horizons: {args.horizon} ...")
    ic_stats, ic_series = compute_ic_stats(factors, closes, args.horizon)

    # 결과 저장
    out_csv = Path(__file__).parent / "factor_screen_result.csv"
    ic_stats.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"Saved: {out_csv}")

    # 콘솔 출력
    print_results(ic_stats, args.horizon, args.top)

    # Factor 상관관계
    if args.corr:
        corr = compute_factor_correlations(factors)
        print("\n--- Factor Correlation (top 10 by abs mean corr) ---")
        mean_corr = corr.abs().mean().sort_values(ascending=False)
        top_corr = mean_corr.head(10).index.tolist()
        print(corr.loc[top_corr, top_corr].round(2).to_string())

    # 시각화
    if not args.no_plot:
        plot_results(ic_stats, ic_series, args.top)


if __name__ == "__main__":
    main()
