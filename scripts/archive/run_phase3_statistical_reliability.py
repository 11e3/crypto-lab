"""
Phase 3: Statistical Reliability Runner
- Monte Carlo simulation (existing module)
- Bootstrap analysis (new)
Generates summary outputs for reliability metrics.

Note: Migrated from VanillaVBO to VanillaVBO with feature flags (2026-01-08)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root resolves to the repository root (one level above 'scripts')
# Previously used parent.parent.parent which pointed outside the repo on Windows.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester.bootstrap_analysis import BootstrapAnalyzer  # noqa: E402
from src.backtester.engine import BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from src.backtester.monte_carlo import MonteCarloResult, MonteCarloSimulator  # noqa: E402
from src.strategies.volatility_breakout.vbo import VanillaVBO  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_data(periods=1500) -> pd.DataFrame:
    dates = pd.date_range(start="2019-01-01", periods=periods, freq="D")
    np.random.seed(123)
    returns = np.random.normal(0.0004, 0.02, periods)
    prices = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.normal(0, 0.005, periods)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            "close": prices,
            "volume": np.random.uniform(100, 1000, periods),
        }
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df.set_index("date")


def load_parquet_or_generate(parquet_path: Path | None) -> pd.DataFrame:
    if parquet_path and parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        date_col = (
            "date" if "date" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        )
        if date_col is None:
            idx_name = df.index.name or "index"
            df = df.reset_index()
            date_col = (
                "date"
                if "date" in df.columns
                else ("timestamp" if "timestamp" in df.columns else idx_name)
            )
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "date"})
        # Ensure required columns exist
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        return df.set_index("date")
    return generate_data()


def parse_ticker_interval_from_path(parquet_path: Path) -> tuple[str, str]:
    stem = parquet_path.stem  # e.g., 'KRW-BTC_day'
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "KRW-BTC", "day"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=None, help="OHLCV Parquet 경로")
    args = parser.parse_args()
    parquet_path = Path(args.parquet) if args.parquet else None

    df = load_parquet_or_generate(parquet_path)

    # Parse ticker/interval from parquet path if provided
    ticker = "KRW-BTC"
    interval = "day"
    if parquet_path and parquet_path.exists():
        ticker, interval = parse_ticker_interval_from_path(parquet_path)

    # Use the official backtest engine for realistic metrics when parquet is provided
    if parquet_path and parquet_path.exists():
        ticker, interval = parse_ticker_interval_from_path(parquet_path)
        strat = VanillaVBO(
            use_improved_noise=True,
            use_adaptive_k=True,
            use_dynamic_slippage=False,
            use_cost_calculator=False,
        )
        base_result = run_backtest(
            strategy=strat,
            tickers=[ticker],
            interval=interval,
            data_dir=parquet_path.parent,
            config=BacktestConfig(initial_capital=100_000.0),
        )
    else:
        # Fallback to synthetic data if no parquet provided
        strat = VanillaVBO(
            use_improved_noise=True,
            use_adaptive_k=True,
            use_dynamic_slippage=False,
            use_cost_calculator=False,
        )
        d = strat.calculate_indicators(df.copy())
        d = strat.generate_signals(d)
        equity = [100_000.0]
        position = 0
        entry = 0
        for _, row in d.iterrows():
            signal = row.get("signal", 0)
            close = row.get("close", 0)
            if signal != 0 and position == 0:
                entry = close
                position = signal
            elif signal * position < 0:
                if position != 0:
                    pnl = (close - entry) * position / entry
                    equity.append(equity[-1] * (1 + pnl))
                    position = signal
                    entry = close if signal != 0 else 0
                else:
                    position = 0
            if position == 0 and len(equity) > 1:
                equity.append(equity[-1])
        if position != 0 and len(d) > 0:
            last_close = d.iloc[-1].get("close", entry)
            pnl = (last_close - entry) * position / entry
            equity.append(equity[-1] * (1 + pnl))
        base_result = BacktestResult()
        base_result.equity_curve = np.array(equity)
        base_result.total_return = (equity[-1] - equity[0]) / equity[0]
        base_result.config = BacktestConfig(initial_capital=equity[0])

    # Monte Carlo (existing module)
    mc = MonteCarloSimulator(base_result)
    mc_result: MonteCarloResult = mc.simulate(n_simulations=500, method="bootstrap", random_seed=42)

    # Bootstrap (new)
    boot = BootstrapAnalyzer(
        df,
        strategy_factory=lambda: VanillaVBO(
            use_improved_noise=True,
            use_adaptive_k=True,
            use_dynamic_slippage=False,
            use_cost_calculator=False,
        ),
        ticker=ticker if parquet_path and parquet_path.exists() else "KRW-BTC",
        interval=interval if parquet_path and parquet_path.exists() else "day",
    )
    # Optimized parameters: block_size=30 (better autocorrelation capture), n_samples=300 (convergence)
    boot_result = boot.analyze(n_samples=300, block_size=30)

    # Output summary
    print("Phase 3 Reliability Summary")
    print("- Monte Carlo:")
    print(
        f"  Mean CAGR: {mc_result.mean_cagr:.2f}% (95% CI: {mc_result.cagr_ci_lower:.2f}% ~ {mc_result.cagr_ci_upper:.2f}%)"
    )
    print(
        f"  Mean MDD: {mc_result.mean_mdd:.2f}% (95% CI: {mc_result.mdd_ci_lower:.2f}% ~ {mc_result.mdd_ci_upper:.2f}%)"
    )
    print(
        f"  Mean Sharpe: {mc_result.mean_sharpe:.2f} (95% CI: {mc_result.sharpe_ci_lower:.2f} ~ {mc_result.sharpe_ci_upper:.2f})"
    )
    print("- Bootstrap:")
    print(
        f"  Mean Return: {boot_result.mean_return:.2%} (95% CI: {boot_result.ci_return_95[0]:.2%} ~ {boot_result.ci_return_95[1]:.2%})"
    )
    print(
        f"  Mean Sharpe: {boot_result.mean_sharpe:.2f} (95% CI: {boot_result.ci_sharpe_95[0]:.2f} ~ {boot_result.ci_sharpe_95[1]:.2f})"
    )
    print(
        f"  Mean MDD: {boot_result.mean_mdd:.2%} (95% CI: {boot_result.ci_mdd_95[0]:.2%} ~ {boot_result.ci_mdd_95[1]:.2%})"
    )

    # Save simple text report
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    txt = (
        "Phase 3 Reliability Summary\n"
        f"Monte Carlo: Mean CAGR {mc_result.mean_cagr:.2f}% (CI {mc_result.cagr_ci_lower:.2f}%~{mc_result.cagr_ci_upper:.2f}%), "
        f"Mean MDD {mc_result.mean_mdd:.2f}%\n"
        f"Bootstrap: Mean Return {boot_result.mean_return:.2%} (CI {boot_result.ci_return_95[0]:.2%}~{boot_result.ci_return_95[1]:.2%}), "
        f"Mean Sharpe {boot_result.mean_sharpe:.2f}\n"
    )
    (reports_dir / "phase3_reliability_summary.txt").write_text(txt, encoding="utf-8")
    print(f"Saved: {reports_dir / 'phase3_reliability_summary.txt'}")


if __name__ == "__main__":
    main()
