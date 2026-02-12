"""Breakout / Channel Breakout 리서치 스크립트 (바이낸스 선물).

N일 채널 돌파 기반 추세추종 (터틀 트레이딩 변형):
- 가격 > N일 고점 → 롱 진입
- 가격 < N일 저점 → 숏 진입
- 청산: 짧은 주기 M일 채널 돌파 반대
- 변동성 타겟팅으로 포지션 크기 조절

결과: ⚠️ Sharpe PASS, MDD FAIL
- Best: Sharpe 1.16, CAGR +60%, MDD -60% (vol target 없음)
  Params: entry=60, exit=20
- Vol target 적용하면 Sharpe 0.90 수준으로 하락
- CAGR은 전략 중 최고이지만 MDD 관리 불가
- Trend Following(EMA)과 수익원 겹침

Usage:
    python research/fetch_data.py                   # 먼저 데이터 수집
    python research/breakout.py                     # 기본 파라미터
    python research/breakout.py --sweep             # 파라미터 스윕
    python research/breakout.py --entry-period 20 --exit-period 10
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
    """(closes, highs, lows) 반환. Breakout은 고가/저가가 필요."""
    files = sorted(DATA_DIR.glob(f"*_{interval}.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR} for interval={interval}")

    closes: dict[str, pd.Series] = {}
    highs: dict[str, pd.Series] = {}
    lows: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for f in files:
        symbol = f.stem.replace(f"_{interval}", "")
        df = pd.read_parquet(f)
        closes[symbol] = df["close"]
        highs[symbol] = df["high"]
        lows[symbol] = df["low"]
        volumes[symbol] = df["close"] * df["volume"]

    close_df = pd.DataFrame(closes).dropna(how="all").ffill()
    high_df = pd.DataFrame(highs).reindex(close_df.index).ffill()
    low_df = pd.DataFrame(lows).reindex(close_df.index).ffill()
    vol_df = pd.DataFrame(volumes).reindex(close_df.index).fillna(0)

    print(f"Loaded {len(close_df.columns)} symbols, {len(close_df)} rows ({close_df.index[0]} ~ {close_df.index[-1]})")
    return close_df, high_df, low_df, vol_df


# === 시그널 생성 ===


def generate_signals(
    closes: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
) -> pd.DataFrame:
    """Donchian Channel Breakout 시그널.

    진입:
    - close > 최근 entry_period일 고점 → 롱
    - close < 최근 entry_period일 저점 → 숏

    청산:
    - 롱 중 close < 최근 exit_period일 저점 → 청산
    - 숏 중 close > 최근 exit_period일 고점 → 청산
    """
    signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    for col in closes.columns:
        c = closes[col].values
        h = highs[col].values
        lo = lows[col].values

        # 채널 계산 (shift(1) 효과: 직전 N일까지만)
        entry_high = pd.Series(h).rolling(entry_period).max().shift(1).values
        entry_low = pd.Series(lo).rolling(entry_period).min().shift(1).values
        exit_high = pd.Series(h).rolling(exit_period).max().shift(1).values
        exit_low = pd.Series(lo).rolling(exit_period).min().shift(1).values

        col_signals = np.zeros(len(c))
        position = 0.0

        for i in range(entry_period, len(c)):
            if np.isnan(entry_high[i]) or np.isnan(entry_low[i]):
                col_signals[i] = position
                continue

            if position == 0:
                # 진입
                if c[i] > entry_high[i]:
                    position = 1.0  # 롱
                elif c[i] < entry_low[i]:
                    position = -1.0  # 숏
            elif position > 0:
                # 롱 청산 (exit 채널 하단 돌파)
                if not np.isnan(exit_low[i]) and c[i] < exit_low[i]:
                    position = 0.0
                # 반전: 숏 진입
                if c[i] < entry_low[i]:
                    position = -1.0
            elif position < 0:
                # 숏 청산 (exit 채널 상단 돌파)
                if not np.isnan(exit_high[i]) and c[i] > exit_high[i]:
                    position = 0.0
                # 반전: 롱 진입
                if c[i] > entry_high[i]:
                    position = 1.0

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
    vol_target: float | None = 0.20,
    vol_lookback: int = 20,
) -> BacktestResult:
    daily_returns = closes.pct_change().fillna(0)
    n_assets = len(closes.columns)

    # 균등 배분: 각 자산 1/n
    weights = signals / n_assets

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


def plot_equity(result: BacktestResult, title: str = "Breakout") -> None:
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
    plt.savefig(Path(__file__).parent / "breakout_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: research/breakout_result.png")


def print_metrics(result: BacktestResult, params: dict) -> None:
    print("\n" + "=" * 60)
    print("BREAKOUT BACKTEST RESULT")
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


def parameter_sweep(closes: pd.DataFrame, highs: pd.DataFrame, lows: pd.DataFrame) -> pd.DataFrame:
    results = []
    entry_periods = [10, 20, 40, 60]
    exit_periods = [5, 10, 20]
    vol_targets = [None, 0.15, 0.20, 0.30]

    # exit_period < entry_period 만 유효
    combos = [(ep, xp) for ep in entry_periods for xp in exit_periods if xp < ep]
    total = len(combos) * len(vol_targets)
    print(f"\nRunning parameter sweep: {total} combinations")

    for ep, xp in combos:
        sigs = generate_signals(closes, highs, lows, entry_period=ep, exit_period=xp)
        for vt in vol_targets:
            res = backtest(closes, sigs, vol_target=vt)
            results.append({
                "entry_period": ep,
                "exit_period": xp,
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
    parser = argparse.ArgumentParser(description="Breakout (Donchian Channel) backtest")
    parser.add_argument("--entry-period", type=int, default=20, help="Entry channel period")
    parser.add_argument("--exit-period", type=int, default=10, help="Exit channel period")
    parser.add_argument("--vol-target", type=float, default=0.20, help="Vol target (0 to disable)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--interval", default="1d", help="OHLCV interval")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot")
    args = parser.parse_args()

    closes, highs, lows, volumes = load_data(args.interval)
    vt = args.vol_target if args.vol_target > 0 else None

    if args.sweep:
        parameter_sweep(closes, highs, lows)
        return

    params = {
        "entry_period": args.entry_period,
        "exit_period": args.exit_period,
        "vol_target": vt,
    }

    signals = generate_signals(closes, highs, lows, args.entry_period, args.exit_period)
    result = backtest(closes, signals, vol_target=vt)
    print_metrics(result, params)

    if not args.no_plot:
        plot_equity(result, f"Breakout (entry={args.entry_period}, exit={args.exit_period})")


if __name__ == "__main__":
    main()
