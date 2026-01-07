"""
Comprehensive Strategy Comparison & Benchmarking Example

This example demonstrates:
1. Comparing 4 different strategy families (VBO, Momentum, Mean Reversion, Pair Trading)
2. Evaluating multiple configurations for each strategy
3. Risk-adjusted performance analysis
4. Trade statistics and drawdown analysis
"""

from src.backtester import BacktestConfig, run_backtest
from src.strategies.mean_reversion import MeanReversion
from src.strategies.momentum import Momentum
from src.strategies.pair_trading import PairTrading
from src.strategies.volatility_breakout import VanillaVBO
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def run_strategy_comparison():
    """
    Compare 4 major strategy types with different configurations.

    Strategies tested:
    1. Volatility Breakout (VBO) - Trend-following
    2. Momentum - Trend-following with velocity
    3. Mean Reversion - Oscillator-based
    4. Pair Trading - Statistical arbitrage
    """

    # Configuration for all backtests
    config = BacktestConfig(
        tickers=["KRW-BTC", "KRW-ETH", "KRW-XRP"],
        interval="day",
        initial_capital=1000000.0,
        fee_rate=0.0005,
        max_slots=3,
        exclude_current_day=True,
    )

    results = {}

    # =========================================================================
    # Strategy 1: Volatility Breakout (Trend-Following)
    # =========================================================================
    logger.info("Running Volatility Breakout strategy...")
    try:
        vbo_strategy = VanillaVBO()
        result = run_backtest(config, vbo_strategy)
        results["VBO (Vanilla)"] = result
        logger.info(
            f"VBO: Return={result.metrics.total_return_pct:.2f}%, Sharpe={result.metrics.sharpe_ratio:.2f}"
        )
    except Exception as e:
        logger.error(f"VBO failed: {e}")

    # =========================================================================
    # Strategy 2: Momentum (Trend-Following with Acceleration)
    # =========================================================================
    logger.info("Running Momentum strategy...")
    try:
        momentum_strategy = Momentum()
        result = run_backtest(config, momentum_strategy)
        results["Momentum (Standard)"] = result
        logger.info(
            f"Momentum: Return={result.metrics.total_return_pct:.2f}%, Sharpe={result.metrics.sharpe_ratio:.2f}"
        )
    except Exception as e:
        logger.error(f"Momentum failed: {e}")

    # =========================================================================
    # Strategy 3: Mean Reversion (Oscillator-Based)
    # =========================================================================
    logger.info("Running Mean Reversion strategy...")
    try:
        mr_strategy = MeanReversion()
        result = run_backtest(config, mr_strategy)
        results["Mean Reversion (Standard)"] = result
        logger.info(
            f"Mean Reversion: Return={result.metrics.total_return_pct:.2f}%, Sharpe={result.metrics.sharpe_ratio:.2f}"
        )
    except Exception as e:
        logger.error(f"Mean Reversion failed: {e}")

    # =========================================================================
    # Strategy 4: Pair Trading (Statistical Arbitrage)
    # =========================================================================
    logger.info("Running Pair Trading strategy...")
    try:
        # Pair trading uses 2 correlated assets
        pair_config = BacktestConfig(
            tickers=["KRW-BTC", "KRW-ETH"],  # Use 2 assets for pairs
            interval="day",
            initial_capital=1000000.0,
            fee_rate=0.0005,
            max_slots=1,
            exclude_current_day=True,
        )
        pt_strategy = PairTrading()
        result = run_backtest(pair_config, pt_strategy)
        results["Pair Trading (BTC-ETH)"] = result
        logger.info(
            f"Pair Trading: Return={result.metrics.total_return_pct:.2f}%, Sharpe={result.metrics.sharpe_ratio:.2f}"
        )
    except Exception as e:
        logger.error(f"Pair Trading failed: {e}")

    # =========================================================================
    # Print Comprehensive Comparison Report
    # =========================================================================
    print_strategy_comparison_report(results)

    return results


def print_strategy_comparison_report(results: dict) -> None:
    """
    Print detailed strategy comparison report.

    Includes:
    - Performance metrics (return, Sharpe, Sortino, Calmar)
    - Risk metrics (max drawdown, volatility)
    - Trade statistics (win rate, profit factor)
    - Risk-adjusted returns
    """

    if not results:
        print("No results to compare.")
        return

    print("\n" + "=" * 120)
    print("STRATEGY COMPARISON REPORT")
    print("=" * 120)

    # =========================================================================
    # Section 1: Performance Metrics
    # =========================================================================
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 120)
    print(
        f"{'Strategy':<25} {'Return':<12} {'CAGR':<12} {'Sharpe':<12} "
        f"{'Sortino':<12} {'Calmar':<12}"
    )
    print("-" * 120)

    for name, result in results.items():
        m = result.metrics
        print(
            f"{name:<25} {m.total_return_pct:>10.2f}% {m.cagr_pct:>10.2f}% "
            f"{m.sharpe_ratio:>10.2f} {m.sortino_ratio:>10.2f} {m.calmar_ratio:>10.2f}"
        )

    # =========================================================================
    # Section 2: Risk Metrics
    # =========================================================================
    print("\n⚠️  RISK METRICS")
    print("-" * 120)
    print(
        f"{'Strategy':<25} {'Max DD':<12} {'Volatility':<12} {'Annual Vol':<12} "
        f"{'VaR 95%':<12} {'Underwater %':<12}"
    )
    print("-" * 120)

    for name, result in results.items():
        m = result.metrics
        print(
            f"{name:<25} {m.mdd_pct:>10.2f}% {m.volatility_pct:>10.2f}% "
            f"{m.annual_volatility_pct:>10.2f}% {m.value_at_risk_pct:>10.2f}% "
            f"{m.underwater_pct:>10.2f}%"
        )

    # =========================================================================
    # Section 3: Trade Statistics
    # =========================================================================
    print("\n📈 TRADE STATISTICS")
    print("-" * 120)
    print(
        f"{'Strategy':<25} {'Trades':<8} {'Winners':<8} {'Losers':<8} "
        f"{'Win Rate':<12} {'Profit Factor':<12} {'Avg Trade':<12}"
    )
    print("-" * 120)

    for name, result in results.items():
        m = result.metrics
        print(
            f"{name:<25} {m.total_trades:>6} {m.winning_trades:>6} "
            f"{m.losing_trades:>6} {m.win_rate_pct:>10.2f}% {m.profit_factor:>10.2f} "
            f"{m.avg_trade_pct:>10.2f}%"
        )

    # =========================================================================
    # Section 4: Best Performers by Metric
    # =========================================================================
    print("\n🏆 BEST PERFORMERS BY METRIC")
    print("-" * 120)

    metrics_to_check = [
        ("Total Return", lambda m: m.total_return_pct),
        ("CAGR", lambda m: m.cagr_pct),
        ("Sharpe Ratio", lambda m: m.sharpe_ratio),
        ("Win Rate", lambda m: m.win_rate_pct),
        ("Profit Factor", lambda m: m.profit_factor),
        ("Lowest Max DD", lambda m: -m.mdd_pct),  # Negative because lower is better
    ]

    for metric_name, metric_fn in metrics_to_check:
        best_name = max(results.items(), key=lambda x: metric_fn(x[1].metrics))[0]
        best_value = metric_fn(results[best_name].metrics)
        print(f"  {metric_name:<20}: {best_name:<30} ({best_value:>10.2f})")

    # =========================================================================
    # Section 5: Ranking Table
    # =========================================================================
    print("\n📋 OVERALL RANKING (by Sharpe Ratio)")
    print("-" * 120)

    ranked = sorted(results.items(), key=lambda x: x[1].metrics.sharpe_ratio, reverse=True)
    for rank, (name, result) in enumerate(ranked, 1):
        m = result.metrics
        print(
            f"  {rank}. {name:<30} Sharpe={m.sharpe_ratio:>7.2f}, "
            f"Return={m.total_return_pct:>8.2f}%, MaxDD={m.mdd_pct:>7.2f}%, "
            f"WinRate={m.win_rate_pct:>6.2f}%"
        )

    print("\n" + "=" * 120)
    print("End of Report")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    results = run_strategy_comparison()
