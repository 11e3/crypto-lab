"""Volatility Breakout (VB) individual asset backtest for Upbit spot.

Buy:
    BTC > MA and high > target
    target = open + (prev_high - prev_low) * k

Sell:
    close < MA_exit, then exit at next open

Usage:
    python -m src.research.experiments.vb_upbit
    python -m src.research.experiments.vb_upbit --k 0.6 --ma-filter 50
    python -m src.research.experiments.vb_upbit --sweep
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.core import (
    build_equal_weight_portfolio,
    compute_equity_trade_metrics,
    compute_yearly_return_and_sharpe,
    load_parquet_ohlcv_by_symbol,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "raw"

# Upbit spot assumptions (per trade / per leg).
UPBIT_FEE = 0.0005
UPBIT_SLIPPAGE = 0.0005
COST_PER_LEG_DEFAULT = UPBIT_FEE + UPBIT_SLIPPAGE

BTC_SYMBOL = "KRW-BTC"
SCRIPT_NAME = Path(__file__).stem
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "results" / SCRIPT_NAME


def ensure_output_dir(path: Path) -> Path:
    """Create output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_data(interval: str = "day") -> dict[str, pd.DataFrame]:
    """Return symbol keyed OHLCV dataframes."""
    data = load_parquet_ohlcv_by_symbol(DATA_DIR, interval)
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
    """Run VB backtest for a single symbol."""
    df = df.copy()
    btc = btc_df.copy()

    btc["ma_filter"] = btc["close"].rolling(ma_filter).mean()
    btc_above_ma = btc["ma_filter"].reindex(df.index).ffill()
    btc_close = btc["close"].reindex(df.index).ffill()

    df["prev_range"] = df["high"].shift(1) - df["low"].shift(1)
    df["target"] = df["open"] + df["prev_range"] * k
    df["ma_exit"] = df["close"].rolling(ma_exit).mean()
    df["next_open"] = df["open"].shift(-1)

    trades: list[TradeResult] = []
    equity_values: list[float] = []
    equity_dates: list[object] = []
    capital = 1.0
    in_position = False
    entry_price = 0.0
    entry_date: object | None = None

    for i in range(max(ma_filter, ma_exit) + 1, len(df) - 1):
        row = df.iloc[i]
        date = df.index[i]

        if not in_position:
            btc_ok = btc_close.iloc[i - 1] > btc_above_ma.iloc[i - 1]
            target_ok = row["high"] > row["target"]
            if btc_ok and target_ok and not np.isnan(row["target"]):
                in_position = True
                entry_price = float(row["target"])
                entry_date = date
        elif row["close"] < row["ma_exit"] and not np.isnan(row["next_open"]):
            exit_price = float(row["next_open"])
            pnl_pct = (exit_price / entry_price - 1.0) - 2.0 * cost_per_leg
            capital *= 1.0 + pnl_pct
            holding = (date - entry_date).days if entry_date is not None else 0
            trades.append(
                TradeResult(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=float(pnl_pct),
                    holding_days=holding,
                )
            )
            in_position = False

        equity_values.append(capital)
        equity_dates.append(date)

    if in_position:
        last = df.iloc[-1]
        exit_price = float(last["close"])
        pnl_pct = (exit_price / entry_price - 1.0) - 2.0 * cost_per_leg
        capital *= 1.0 + pnl_pct
        equity_values[-1] = capital
        trades.append(
            TradeResult(
                entry_date=entry_date,
                exit_date=df.index[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=float(pnl_pct),
                holding_days=0,
            )
        )

    equity = pd.Series(equity_values, index=equity_dates)
    metrics = compute_metrics(equity, trades)
    return BacktestResult(symbol=symbol, equity=equity, trades=trades, metrics=metrics)


def compute_metrics(equity: pd.Series, trades: list[TradeResult]) -> dict[str, float]:
    """Compute performance metrics."""
    trade_pnls = [trade.pnl_pct for trade in trades]
    trade_holding_days = [trade.holding_days for trade in trades]
    return compute_equity_trade_metrics(equity, trade_pnls, trade_holding_days)


def print_results(results: list[BacktestResult], params: dict[str, float]) -> None:
    """Print per-symbol and portfolio summary table."""
    print("\n" + "=" * 95)
    print(
        f"VB UPBIT BACKTEST (k={params['k']}, ma_filter={params['ma_filter']}, "
        f"ma_exit={params['ma_exit']}, cost={params['cost_per_leg']:.4f}/leg)"
    )
    print("=" * 95)
    print(
        f"{'Symbol':<12s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} "
        f"{'WR%':>5s} {'#Tr':>4s} {'AvgPnL':>8s} {'PF':>5s} {'AvgHold':>7s}"
    )
    print("-" * 95)

    all_equities: dict[str, pd.Series] = {}
    for result in sorted(results, key=lambda value: value.metrics["Sharpe"], reverse=True):
        metric = result.metrics
        print(
            f"  {result.symbol:<10s} {metric['Sharpe']:7.2f} {metric['CAGR']:+7.2%} "
            f"{metric['MDD']:+7.2%} {metric['Calmar']:7.2f} {metric['WinRate']:4.0%} "
            f"{metric['NumTrades']:4.0f} {metric['AvgPnL']:+7.2%} {metric['PF']:5.2f} "
            f"{metric['AvgHold']:6.1f}d"
        )
        all_equities[result.symbol] = result.equity

    portfolio = build_equal_weight_portfolio(all_equities)
    if portfolio is not None:
        print("-" * 95)
        print(
            f"  {'EW_PORT':<10s} {portfolio.sharpe:7.2f} {portfolio.cagr:+7.2%} {portfolio.mdd:+7.2%}"
        )

        yearly_returns, yearly_sharpe = compute_yearly_return_and_sharpe(portfolio.returns)
        print("-" * 95)
        print(f"  {'Year':<10s} {'Return':>8s} {'Sharpe':>7s}")
        for year in sorted(yearly_returns.index):
            print(f"  {year:<10} {yearly_returns[year]:+7.2%} {yearly_sharpe[year]:7.2f}")

    print("=" * 95)


def parameter_sweep(
    data: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    out_dir: Path,
    cost_per_leg: float = COST_PER_LEG_DEFAULT,
) -> pd.DataFrame:
    """Run parameter sweep and save ranked result table."""
    ks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ma_filters = [10, 20, 30, 50, 60, 70, 80]
    ma_exits = [3, 5, 10]

    combos = list(product(ks, ma_filters, ma_exits))
    print(
        f"\nRunning sweep: {len(combos)} param combos x {len(data)} symbols (cost={cost_per_leg:.4f}/leg)"
    )

    rows: list[dict[str, float]] = []
    for k_value, ma_filter, ma_exit in combos:
        equities: dict[str, pd.Series] = {}
        sharpes: list[float] = []
        cagrs: list[float] = []
        mdds: list[float] = []

        for symbol, frame in data.items():
            result = backtest_symbol(
                symbol, frame, btc_df, k_value, ma_filter, ma_exit, cost_per_leg
            )
            sharpes.append(result.metrics["Sharpe"])
            cagrs.append(result.metrics["CAGR"])
            mdds.append(result.metrics["MDD"])
            equities[symbol] = result.equity

        portfolio = build_equal_weight_portfolio(equities)
        ew_sharpe = portfolio.sharpe if portfolio is not None else float("nan")
        ew_mdd = portfolio.mdd if portfolio is not None else float("nan")

        rows.append(
            {
                "k": float(k_value),
                "ma_filter": float(ma_filter),
                "ma_exit": float(ma_exit),
                "avg_sharpe": float(np.mean(sharpes)),
                "med_sharpe": float(np.median(sharpes)),
                "min_sharpe": float(np.min(sharpes)),
                "avg_cagr": float(np.mean(cagrs)),
                "avg_mdd": float(np.mean(mdds)),
                "ew_sharpe": float(ew_sharpe),
                "ew_mdd": float(ew_mdd),
            }
        )

    df = pd.DataFrame(rows).sort_values("ew_sharpe", ascending=False)
    print("\n--- Top 15 by EW Portfolio Sharpe ---")
    print(df.head(15).to_string(index=False, float_format="{:.4f}".format))

    mdd_pass = df[df["ew_mdd"] >= -0.30].sort_values("ew_sharpe", ascending=False)
    if not mdd_pass.empty:
        print("\n--- MDD <= 30% (EW Portfolio) ---")
        print(mdd_pass.head(10).to_string(index=False, float_format="{:.4f}".format))
    else:
        print("\n  No combos pass EW MDD <= 30%")

    ensure_output_dir(out_dir)
    out_csv = out_dir / f"{SCRIPT_NAME}_sweep.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="VB Upbit spot backtest")
    parser.add_argument("--k", type=float, default=0.5, help="Breakout coefficient")
    parser.add_argument("--ma-filter", type=int, default=20, help="BTC MA filter period")
    parser.add_argument("--ma-exit", type=int, default=5, help="Exit MA period")
    parser.add_argument(
        "--cost",
        type=float,
        default=COST_PER_LEG_DEFAULT,
        help="Cost per leg (default 0.0010 = 0.10%%)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for csv/png artifacts",
    )
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    data = load_data("day")
    if BTC_SYMBOL not in data:
        print(f"{BTC_SYMBOL} required for MA filter")
        return

    btc_df = data[BTC_SYMBOL]
    out_dir = ensure_output_dir(args.out_dir)

    if args.sweep:
        parameter_sweep(data, btc_df, out_dir=out_dir, cost_per_leg=args.cost)
        return

    params = {
        "k": args.k,
        "ma_filter": float(args.ma_filter),
        "ma_exit": float(args.ma_exit),
        "cost_per_leg": args.cost,
    }
    results = [
        backtest_symbol(symbol, frame, btc_df, args.k, args.ma_filter, args.ma_exit, args.cost)
        for symbol, frame in data.items()
    ]
    print_results(results, params)

    if not args.no_plot:
        n_symbols = len(results)
        ncols = min(3, n_symbols)
        nrows = (n_symbols + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        if nrows * ncols == 1:
            axes = np.array([axes])

        sorted_results = sorted(results, key=lambda value: value.metrics["Sharpe"], reverse=True)
        for axis, result in zip(axes.flat, sorted_results, strict=False):
            axis.plot(result.equity.index, result.equity.values, linewidth=1.0)
            sharpe = result.metrics["Sharpe"]
            axis.set_title(f"{result.symbol} (Sharpe={sharpe:.2f})", fontsize=10)
            axis.grid(True, alpha=0.3)

        for axis in axes.flat[n_symbols:]:
            axis.set_visible(False)

        plt.suptitle(
            f"VB Upbit k={args.k}, MA_filter={args.ma_filter}, MA_exit={args.ma_exit}",
            fontsize=13,
        )
        plt.tight_layout()
        out_path = out_dir / f"{SCRIPT_NAME}_result.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
