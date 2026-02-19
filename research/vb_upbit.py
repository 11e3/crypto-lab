"""Volatility Breakout (VB) 개별 자산 백테스트 (업비트 현물).

매수: BTC > MA AND high > target (target = open + (prev_high - prev_low) * k)
매도: close < MA_exit -> 다음날 open에 청산

Usage:
    python research/vb_upbit.py
    python research/vb_upbit.py --k 0.6 --ma-filter 50
    python research/vb_upbit.py --sweep
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 업비트 현물 데이터
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# 업비트 현물 수수료: 0.05% per trade (슬리피지 없음 - 현물)
UPBIT_FEE = 0.0005
COST_PER_LEG_DEFAULT = UPBIT_FEE

BTC_SYMBOL = "KRW-BTC"


def load_data(interval: str = "day") -> dict[str, pd.DataFrame]:
    """각 심볼별 OHLCV DataFrame dict 반환."""
    files = sorted(DATA_DIR.glob(f"*_{interval}.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR}")

    data = {}
    for f in files:
        symbol = f.stem.replace(f"_{interval}", "")
        data[symbol] = pd.read_parquet(f)

    print(f"Loaded {len(data)} symbols: {', '.join(sorted(data.keys()))}")
    return data


@dataclass
class TradeResult:
    entry_date: object
    exit_date: object
    entry_price: float
    exit_price: float
    pnl_pct: float
    holding_days: int


@dataclass
class BacktestResult:
    symbol: str
    equity: pd.Series
    trades: list[TradeResult]
    metrics: dict[str, float]


def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    btc_df: pd.DataFrame,
    k: float = 0.5,
    ma_filter: int = 20,
    ma_exit: int = 5,
    cost_per_leg: float = COST_PER_LEG_DEFAULT,
) -> BacktestResult:
    """단일 심볼 VB 백테스트."""
    df = df.copy()
    btc = btc_df.copy()

    # BTC MA filter
    btc["ma_filter"] = btc["close"].rolling(ma_filter).mean()
    btc_above_ma = btc["ma_filter"].reindex(df.index).ffill()
    btc_close = btc["close"].reindex(df.index).ffill()

    # Target price: open + (prev_high - prev_low) * k
    df["prev_range"] = df["high"].shift(1) - df["low"].shift(1)
    df["target"] = df["open"] + df["prev_range"] * k

    # Exit MA
    df["ma_exit"] = df["close"].rolling(ma_exit).mean()

    # Next day open (for exit price)
    df["next_open"] = df["open"].shift(-1)

    trades: list[TradeResult] = []
    equity_values = []
    equity_dates = []
    capital = 1.0
    in_position = False
    entry_price = 0.0
    entry_date = None

    for i in range(max(ma_filter, ma_exit) + 1, len(df) - 1):
        row = df.iloc[i]
        date = df.index[i]

        if not in_position:
            # 매수 조건: 전일 BTC > MA (look-ahead 방지) + 오늘 high > target
            btc_ok = btc_close.iloc[i - 1] > btc_above_ma.iloc[i - 1]
            target_ok = row["high"] > row["target"]

            if btc_ok and target_ok and not np.isnan(row["target"]):
                in_position = True
                entry_price = row["target"]
                entry_date = date
        else:
            # 매도 조건: close < MA_exit -> 다음날 open에 청산
            if row["close"] < row["ma_exit"] and not np.isnan(row["next_open"]):
                exit_price = row["next_open"]
                pnl_pct = (exit_price / entry_price - 1) - 2 * cost_per_leg
                capital *= (1 + pnl_pct)
                holding = (date - entry_date).days if entry_date else 0
                trades.append(TradeResult(entry_date, date, entry_price, exit_price,
                                          pnl_pct, holding))
                in_position = False

        equity_values.append(capital)
        equity_dates.append(date)

    # 미청산 포지션 처리
    if in_position:
        last = df.iloc[-1]
        exit_price = last["close"]
        pnl_pct = (exit_price / entry_price - 1) - 2 * cost_per_leg
        capital *= (1 + pnl_pct)
        equity_values[-1] = capital
        trades.append(TradeResult(entry_date, df.index[-1], entry_price, exit_price,
                                  pnl_pct, 0))

    equity = pd.Series(equity_values, index=equity_dates)
    metrics = compute_metrics(equity, trades)

    return BacktestResult(symbol=symbol, equity=equity, trades=trades, metrics=metrics)


def compute_metrics(equity: pd.Series, trades: list[TradeResult]) -> dict[str, float]:
    """성과 지표."""
    if len(equity) < 2:
        return {"CAGR": 0, "Sharpe": 0, "MDD": 0, "Calmar": 0, "WinRate": 0,
                "AvgPnL": 0, "NumTrades": 0, "AvgHold": 0, "PF": 0}

    returns = equity.pct_change().dropna()
    days = len(returns)
    years = days / 365

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    growth = max(1 + total_return, 1e-8)
    cagr = growth ** (1 / max(years, 0.01)) - 1

    vol = returns.std() * np.sqrt(365)
    sharpe = (returns.mean() * 365) / max(vol, 1e-8)

    peak = equity.cummax()
    mdd = ((equity - peak) / peak).min()
    calmar = cagr / max(abs(mdd), 1e-8)

    n_trades = len(trades)
    if n_trades > 0:
        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n_trades
        avg_pnl = np.mean(pnls)
        avg_hold = np.mean([t.holding_days for t in trades])
        gross_win = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        pf = gross_win / max(gross_loss, 1e-8)
    else:
        win_rate = avg_pnl = avg_hold = pf = 0.0

    return {
        "CAGR": cagr, "Sharpe": sharpe, "MDD": mdd, "Calmar": calmar,
        "WinRate": win_rate, "AvgPnL": avg_pnl, "NumTrades": n_trades,
        "AvgHold": avg_hold, "PF": pf,
    }


SCRIPT_NAME = Path(__file__).stem


def print_results(results: list[BacktestResult], params: dict) -> None:
    """전 심볼 결과 테이블."""
    print("\n" + "=" * 95)
    print(f"VB UPBIT BACKTEST (k={params['k']}, ma_filter={params['ma_filter']}, "
          f"ma_exit={params['ma_exit']}, cost={params['cost_per_leg']:.4f}/leg)")
    print("=" * 95)
    print(f"{'Symbol':<12s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} "
          f"{'WR%':>5s} {'#Tr':>4s} {'AvgPnL':>8s} {'PF':>5s} {'AvgHold':>7s}")
    print("-" * 95)

    all_equities = {}
    for r in sorted(results, key=lambda x: x.metrics["Sharpe"], reverse=True):
        m = r.metrics
        print(f"  {r.symbol:<10s} {m['Sharpe']:7.2f} {m['CAGR']:+7.2%} {m['MDD']:+7.2%} "
              f"{m['Calmar']:7.2f} {m['WinRate']:4.0%} {m['NumTrades']:4.0f} "
              f"{m['AvgPnL']:+7.2%} {m['PF']:5.2f} {m['AvgHold']:6.1f}d")
        all_equities[r.symbol] = r.equity

    # EW portfolio
    eq_df = pd.DataFrame(all_equities).dropna(how="any")
    if len(eq_df) > 1:
        port_ret = eq_df.pct_change().mean(axis=1).dropna()
        port_eq = (1 + port_ret).cumprod()
        days = len(port_ret)
        years = days / 365
        total_r = port_eq.iloc[-1] / port_eq.iloc[0] - 1
        growth = max(1 + total_r, 1e-8)
        cagr = growth ** (1 / max(years, 0.01)) - 1
        vol = port_ret.std() * np.sqrt(365)
        sharpe = (port_ret.mean() * 365) / max(vol, 1e-8)
        peak = port_eq.cummax()
        mdd = ((port_eq - peak) / peak).min()

        print("-" * 95)
        print(f"  {'EW_PORT':<10s} {sharpe:7.2f} {cagr:+7.2%} {mdd:+7.2%}")

        # 연도별 수익률
        yearly = port_ret.groupby(port_ret.index.year).apply(lambda r: (1 + r).prod() - 1)
        yearly_sharpe = port_ret.groupby(port_ret.index.year).apply(
            lambda r: (r.mean() * 365) / max(r.std() * np.sqrt(365), 1e-8)
        )
        print("-" * 95)
        print(f"  {'Year':<10s} {'Return':>8s} {'Sharpe':>7s}")
        for yr in sorted(yearly.index):
            print(f"  {yr:<10} {yearly[yr]:+7.2%} {yearly_sharpe[yr]:7.2f}")

    print("=" * 95)


def parameter_sweep(
    data: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    cost_per_leg: float = COST_PER_LEG_DEFAULT,
) -> pd.DataFrame:
    """파라미터 스윕."""
    ks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ma_filters = [10, 20, 30, 50, 60, 70, 80]
    ma_exits = [3, 5, 10]

    combos = list(product(ks, ma_filters, ma_exits))
    print(f"\nRunning sweep: {len(combos)} param combos x {len(data)} symbols "
          f"(cost={cost_per_leg:.4f}/leg)")

    rows = []
    for k_val, mf, me in combos:
        equities = {}
        sharpes = []
        cagrs = []
        mdds = []
        for sym, df in data.items():
            res = backtest_symbol(sym, df, btc_df, k_val, mf, me, cost_per_leg)
            sharpes.append(res.metrics["Sharpe"])
            cagrs.append(res.metrics["CAGR"])
            mdds.append(res.metrics["MDD"])
            equities[sym] = res.equity

        # EW portfolio
        eq_df = pd.DataFrame(equities).dropna(how="any")
        port_ret = eq_df.pct_change().mean(axis=1).dropna()
        port_eq = (1 + port_ret).cumprod()
        p_vol = port_ret.std() * np.sqrt(365)
        p_sharpe = (port_ret.mean() * 365) / max(p_vol, 1e-8)
        p_peak = port_eq.cummax()
        p_mdd = ((port_eq - p_peak) / p_peak).min()

        rows.append({
            "k": k_val, "ma_filter": mf, "ma_exit": me,
            "avg_sharpe": np.mean(sharpes), "med_sharpe": np.median(sharpes),
            "min_sharpe": np.min(sharpes),
            "avg_cagr": np.mean(cagrs), "avg_mdd": np.mean(mdds),
            "ew_sharpe": p_sharpe, "ew_mdd": p_mdd,
        })

    df = pd.DataFrame(rows).sort_values("ew_sharpe", ascending=False)
    print(f"\n--- Top 15 by EW Portfolio Sharpe ---")
    print(df.head(15).to_string(index=False, float_format="{:.4f}".format))

    mdd_pass = df[df["ew_mdd"] >= -0.30].sort_values("ew_sharpe", ascending=False)
    if not mdd_pass.empty:
        print(f"\n--- MDD <= 30% (EW Portfolio) ---")
        print(mdd_pass.head(10).to_string(index=False, float_format="{:.4f}".format))
    else:
        print("\n  No combos pass EW MDD <= 30%")

    out_csv = Path(__file__).parent / f"{SCRIPT_NAME}_sweep.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="VB Upbit spot backtest")
    parser.add_argument("--k", type=float, default=0.5, help="Breakout coefficient")
    parser.add_argument("--ma-filter", type=int, default=20, help="BTC MA filter period")
    parser.add_argument("--ma-exit", type=int, default=5, help="Exit MA period")
    parser.add_argument("--cost", type=float, default=COST_PER_LEG_DEFAULT,
                        help="Cost per leg (default 0.0005 = 0.05%%)")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    data = load_data("day")

    if BTC_SYMBOL not in data:
        print(f"{BTC_SYMBOL} required for MA filter")
        return

    btc_df = data[BTC_SYMBOL]

    if args.sweep:
        parameter_sweep(data, btc_df, args.cost)
        return

    params = {"k": args.k, "ma_filter": args.ma_filter, "ma_exit": args.ma_exit,
              "cost_per_leg": args.cost}
    results = []
    for sym, df in data.items():
        res = backtest_symbol(sym, df, btc_df, args.k, args.ma_filter, args.ma_exit, args.cost)
        results.append(res)

    print_results(results, params)

    if not args.no_plot:
        n_syms = len(results)
        ncols = min(3, n_syms)
        nrows = (n_syms + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        if nrows * ncols == 1:
            axes = np.array([axes])
        for ax, r in zip(axes.flat, sorted(results, key=lambda x: x.metrics["Sharpe"], reverse=True)):
            ax.plot(r.equity.index, r.equity.values, linewidth=1.0)
            s = r.metrics["Sharpe"]
            ax.set_title(f"{r.symbol} (Sharpe={s:.2f})", fontsize=10)
            ax.grid(True, alpha=0.3)
        for ax in axes.flat[n_syms:]:
            ax.set_visible(False)
        plt.suptitle(f"VB Upbit k={args.k}, MA_filter={args.ma_filter}, MA_exit={args.ma_exit}",
                     fontsize=13)
        plt.tight_layout()
        out_path = Path(__file__).parent / f"{SCRIPT_NAME}_result.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
