"""CLI subcommand: analyze (strategy/results analysis suite)."""

from __future__ import annotations

import argparse
from typing import Any


def register(subparsers: Any) -> None:
    """Register analyze subcommand."""
    p = subparsers.add_parser("analyze", help="Run strategy analysis suite")
    p.add_argument("--tickers", nargs="+", required=True, metavar="TICKER")
    p.add_argument("--strategy", required=True, metavar="NAME")
    p.add_argument("--start", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--capital", type=float, default=1_000_000, metavar="KRW")
    p.add_argument("--slots", type=int, default=5, metavar="N")
    p.add_argument("--fee", type=float, default=0.0005, metavar="RATE")
    p.add_argument("--interval", default="day", metavar="INTERVAL")
    p.add_argument("--shuffles", type=int, default=100, metavar="N",
                   help="Number of permutation shuffles (default: 100)")
    p.add_argument("--bootstrap-samples", type=int, default=100, metavar="N",
                   dest="bootstrap_samples",
                   help="Number of bootstrap samples (default: 100)")
    p.add_argument("--skip-perm", action="store_true", dest="skip_perm",
                   help="Skip permutation test")
    p.add_argument("--skip-robust", action="store_true", dest="skip_robust",
                   help="Skip robustness analysis (bootstrap + parameter sweep)")
    p.set_defaults(func=_run_analyze)


def _run_analyze(args: argparse.Namespace) -> None:
    from src.backtester.analysis.strategy_analysis import run_strategy_analysis
    from src.cli._helpers import build_config, parse_date
    from src.strategies.registry import registry

    cls = registry.get_class(args.strategy)
    if cls is None:
        raise SystemExit(f"Unknown strategy: {args.strategy!r}")

    name = args.strategy
    config = build_config(args)
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    def factory_0() -> Any:
        return registry.create(name)

    def factory_p(params: dict[str, Any]) -> Any:
        return registry.create(name, **params)

    print(f"\nRunning analysis for {name} on {', '.join(args.tickers)} ...")
    if not args.skip_perm:
        print(f"  Permutation test: {args.shuffles} shuffles")
    if not args.skip_robust:
        print(f"  Bootstrap: {args.bootstrap_samples} samples")
    print()

    result = run_strategy_analysis(
        strategy_name=name,
        strategy_factory_0=factory_0,
        strategy_factory_p=factory_p,
        tickers=args.tickers,
        interval=args.interval,
        config=config,
        start_date=start_date,
        end_date=end_date,
        n_shuffles=args.shuffles,
        n_bootstrap=args.bootstrap_samples,
        run_permutation=not args.skip_perm,
        run_robustness=not args.skip_robust,
    )

    _print_analysis_result(result, args)


def _print_analysis_result(result: Any, args: argparse.Namespace) -> None:
    from src.backtester.analysis.strategy_analysis import StrategyAnalysisResult

    r: StrategyAnalysisResult = result
    bt = r.backtest
    tickers_str = ", ".join(r.tickers)

    start_str = args.start or "N/A"
    end_str = args.end or "N/A"

    print(f"=== Strategy Analysis: {r.strategy_name} on {tickers_str} ===")
    print()

    # --- Section 1: Backtest results ---
    print(f"[1/4] 백테스트 기본 성과 ({start_str} ~ {end_str})")

    perf = r.performance
    sortino = perf.sortino_ratio if perf else 0.0
    volatility = perf.volatility_pct if perf else 0.0
    profit_factor = perf.profit_factor if perf else bt.profit_factor

    print(f"  CAGR         : {bt.cagr:>7.2f}%    Sharpe   : {bt.sharpe_ratio:.2f}")
    print(f"  Sortino      : {sortino:>7.2f}     Calmar   : {bt.calmar_ratio:.2f}")
    print(f"  MDD          : {bt.mdd:>7.2f}%    Volatility: {volatility:.2f}%")
    print(f"  Win Rate     : {bt.win_rate:.1f}%     Profit Factor: {profit_factor:.2f}"
          f"    Trades: {bt.total_trades}")
    print()

    # --- Section 2: Benchmark comparison ---
    print("[2/4] 벤치마크 비교 (vs Buy & Hold)")
    if r.benchmark is not None:
        bm = r.benchmark
        cagr_diff = bt.cagr - bm.cagr
        mdd_diff = bm.mdd - bt.mdd
        sharpe_diff = bt.sharpe_ratio - bm.sharpe

        sign_cagr = "+" if cagr_diff >= 0 else ""
        sign_mdd = "+" if mdd_diff >= 0 else ""
        sign_sharpe = "+" if sharpe_diff >= 0 else ""

        print(f"  {'항목':<8}  {'전략':>8}    {'B&H':>8}    {'차이':>12}")
        print(f"  {'CAGR':<8}  {bt.cagr:>7.2f}%    {bm.cagr:>7.2f}%    "
              f"({sign_cagr}{cagr_diff:.2f}%p)")
        print(f"  {'MDD':<8}  {bt.mdd:>7.2f}%    {bm.mdd:>7.2f}%    "
              f"({sign_mdd}{mdd_diff:.2f}%p 낙폭)")
        print(f"  {'Sharpe':<8}  {bt.sharpe_ratio:>8.2f}    {bm.sharpe:>8.2f}    "
              f"({sign_sharpe}{sharpe_diff:.2f})")
    else:
        print("  벤치마크 데이터 없음")
    print()

    # --- Section 3: Statistical significance ---
    print(f"[3/4] 통계적 유의성 (Permutation Test, n={args.shuffles})")
    if r.permutation is not None:
        p = r.permutation
        verdict = "5% 유의수준에서 알파 확인됨" if p.is_statistically_significant else "유의하지 않음"
        print(f"  Z-score : {p.z_score:.2f}    P-value : {p.p_value:.4f}")
        print(f"  → {verdict}")
    else:
        print("  Permutation test skipped (--skip-perm)")
    print()

    # --- Section 4: Robustness ---
    print("[4/4] 강건성")
    if r.bootstrap is not None:
        bs = r.bootstrap
        ci_ret = bs.ci_return_95
        ci_sh = bs.ci_sharpe_95
        print(f"  CAGR 95% CI  : [{ci_ret[0]:.1f}%, {ci_ret[1]:.1f}%]")
        print(f"  Sharpe 95% CI: [{ci_sh[0]:.2f},  {ci_sh[1]:.2f}]")
    else:
        print("  Bootstrap: skipped (--skip-robust)")

    if r.robustness is not None:
        rob = r.robustness
        print(f"  파라미터 이웃 성공률: {rob.neighbor_success_rate:.1%}")
        if rob.sensitivity_scores:
            sorted_sens = sorted(rob.sensitivity_scores.items(), key=lambda x: -x[1])
            sens_str = " > ".join(f"{k}({v:.2f})" for k, v in sorted_sens)
            print(f"  민감 파라미터: {sens_str}")
    else:
        print("  Robustness: skipped (--skip-robust)")
    print()

    # --- Go-live checklist ---
    print("=== 자동매매 전환 평가 ===")
    passed_count = sum(1 for c in r.go_live_checks if c.passed)
    total_count = len(r.go_live_checks)

    for check in r.go_live_checks:
        tag = f"[{check.level}]"
        print(f"  {tag:<6} {check.description} ({check.detail})")

    verdict_emoji = "✓" if passed_count == total_count else "✗"
    verdict_text = "전환 가능" if passed_count >= 4 else "조건 미달"
    print(f"  → 종합: {verdict_text} {verdict_emoji}  ({passed_count}/{total_count})")


__all__ = ["register"]
