"""
Bootstrap 안정화 테스트 및 최적화

목표:
1. Block length 최적화 (현재 20 → 최적값 찾기)
2. Sample count 안정화 (100 → 500+)
3. OHLC resampling 일관성 검증

Note: Migrated from VanillaVBO to VanillaVBO with feature flags (2026-01-08)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester.bootstrap_analysis import BootstrapAnalyzer
from src.backtester.engine import BacktestConfig
from src.strategies.volatility_breakout.vbo import VanillaVBO


def load_data() -> pd.DataFrame:
    """실데이터 로드"""
    data_path = Path("c:/workspace/dev/crypto-quant-system/data/raw/KRW-BTC_day.parquet")
    df = pd.read_parquet(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.rename(columns={"timestamp": "date"}).set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def test_block_size_sensitivity():
    """Block size별 Bootstrap 결과 안정성 테스트"""
    print("=" * 80)
    print("TEST 1: Block Size Sensitivity Analysis")
    print("=" * 80)

    data = load_data()
    print(f"데이터 기간: {data.index[0]} ~ {data.index[-1]} ({len(data)} days)")

    # 테스트할 block size들 (daily autocorrelation 고려)
    block_sizes = [5, 10, 15, 20, 30, 40, 60]
    results = {}

    for block_size in block_sizes:
        print(f"\n[Block size = {block_size}]")

        boot = BootstrapAnalyzer(
            data=data,
            strategy_factory=lambda: VanillaVBO(
                sma_period=4,
                trend_sma_period=8,
                use_improved_noise=True,
                use_adaptive_k=True,
            ),
            backtest_config=BacktestConfig(
                initial_capital=100.0,
                fee_rate=0.0005,
                slippage_rate=0.0005,
                max_slots=4,
                stop_loss_pct=0.05,
                take_profit_pct=0.15,
            ),
            ticker="KRW-BTC",
            interval="day",
        )

        # 빠른 테스트용 n_samples=50
        result = boot.analyze(n_samples=50, block_size=block_size)

        results[block_size] = result

        ci_width_return = result.ci_return_95[1] - result.ci_return_95[0]
        ci_width_sharpe = result.ci_sharpe_95[1] - result.ci_sharpe_95[0]

        print(f"  Mean Return: {result.mean_return:.2%}")
        print(f"  95% CI Width: {ci_width_return:.2%}")
        print(f"  Mean Sharpe: {result.mean_sharpe:.2f}")
        print(f"  Sharpe CI Width: {ci_width_sharpe:.2f}")
        print(f"  Valid samples: {len(result.returns)}/50")

    # 최적 block size 추천 (CI width가 가장 안정적인 값)
    ci_widths = {
        bs: (results[bs].ci_return_95[1] - results[bs].ci_return_95[0])
        for bs in block_sizes
        if len(results[bs].returns) > 40
    }

    if ci_widths:
        optimal_bs = min(
            ci_widths, key=lambda x: abs(ci_widths[x] - np.median(list(ci_widths.values())))
        )
        print(f"\n✅ 추천 Block Size: {optimal_bs} (중간값에 가장 가까운 안정적 CI)")

    return results


def test_sample_count_convergence():
    """Sample count별 수렴성 테스트"""
    print("\n" + "=" * 80)
    print("TEST 2: Sample Count Convergence")
    print("=" * 80)

    data = load_data()

    # 최적 block size 사용 (테스트 1 결과 반영: 일반적으로 20-30이 적절)
    optimal_block_size = 30

    # 점진적으로 샘플 수 증가
    sample_counts = [50, 100, 200, 300, 500]
    means_return = []
    means_sharpe = []
    ci_widths_return = []
    ci_widths_sharpe = []

    for n_samples in sample_counts:
        print(f"\n[n_samples = {n_samples}]")

        boot = BootstrapAnalyzer(
            data=data,
            strategy_factory=lambda: VanillaVBO(
                sma_period=4,
                trend_sma_period=8,
                use_improved_noise=True,
                use_adaptive_k=True,
            ),
            backtest_config=BacktestConfig(
                initial_capital=100.0,
                fee_rate=0.0005,
                slippage_rate=0.0005,
                max_slots=4,
                stop_loss_pct=0.05,
                take_profit_pct=0.15,
            ),
            ticker="KRW-BTC",
            interval="day",
        )

        result = boot.analyze(n_samples=n_samples, block_size=optimal_block_size)

        means_return.append(result.mean_return)
        means_sharpe.append(result.mean_sharpe)
        ci_widths_return.append(result.ci_return_95[1] - result.ci_return_95[0])
        ci_widths_sharpe.append(result.ci_sharpe_95[1] - result.ci_sharpe_95[0])

        print(f"  Mean Return: {result.mean_return:.2%} ± {ci_widths_return[-1] / 2:.2%}")
        print(f"  Mean Sharpe: {result.mean_sharpe:.2f} ± {ci_widths_sharpe[-1] / 2:.2f}")
        print(f"  Valid samples: {len(result.returns)}/{n_samples}")

    # 수렴성 확인 (마지막 2개 샘플의 평균 변화율)
    if len(means_return) >= 2:
        return_change = abs(means_return[-1] - means_return[-2]) / abs(means_return[-2]) * 100
        sharpe_change = abs(means_sharpe[-1] - means_sharpe[-2]) / abs(means_sharpe[-2]) * 100

        print("\n변화율 (마지막 2개):")
        print(f"  Return: {return_change:.2f}%")
        print(f"  Sharpe: {sharpe_change:.2f}%")

        if return_change < 5 and sharpe_change < 5:
            print(f"✅ 수렴 달성: {sample_counts[-2]} 샘플 이상 권장")
        else:
            print(f"⚠️  수렴 불충분: {sample_counts[-1]} 이상 권장")

    return {
        "sample_counts": sample_counts,
        "means_return": means_return,
        "means_sharpe": means_sharpe,
        "ci_widths_return": ci_widths_return,
        "ci_widths_sharpe": ci_widths_sharpe,
    }


def test_ohlc_consistency():
    """OHLC resampling 일관성 검증"""
    print("\n" + "=" * 80)
    print("TEST 3: OHLC Resampling Consistency")
    print("=" * 80)

    data = load_data()

    boot = BootstrapAnalyzer(
        data=data,
        strategy_factory=lambda: VanillaVBO(
            sma_period=4,
            trend_sma_period=8,
            use_improved_noise=True,
            use_adaptive_k=True,
        ),
        backtest_config=BacktestConfig(
            initial_capital=100.0, fee_rate=0.0005, slippage_rate=0.0005
        ),
        ticker="KRW-BTC",
        interval="day",
    )

    # 여러 번 resampling하여 OHLC 관계 검증
    violations = []
    for i in range(10):
        resampled = boot._resample_data(data, block_size=30)

        # OHLC 제약조건 검증
        high_valid = (resampled["high"] >= resampled["close"]).all() and (
            resampled["high"] >= resampled["open"]
        ).all()
        low_valid = (resampled["low"] <= resampled["close"]).all() and (
            resampled["low"] <= resampled["open"]
        ).all()

        if not (high_valid and low_valid):
            violations.append(i)

        print(f"  Sample {i + 1}: High valid={high_valid}, Low valid={low_valid}")

    if not violations:
        print("\n✅ OHLC 일관성: 모든 샘플 통과 (10/10)")
    else:
        print(f"\n⚠️  OHLC 일관성: {len(violations)}/10 샘플 위반")
        print(f"  위반 샘플: {violations}")

    # Buffer 크기 검증
    sample = boot._resample_data(data, block_size=30)
    high_buffer = (sample["high"] / sample[["open", "close"]].max(axis=1) - 1).median()
    low_buffer = (1 - sample["low"] / sample[["open", "close"]].min(axis=1)).median()

    print("\nBuffer 크기:")
    print(f"  High buffer: {high_buffer * 100:.2f}% (현재 0.5%)")
    print(f"  Low buffer: {low_buffer * 100:.2f}% (현재 0.5%)")

    return len(violations) == 0


if __name__ == "__main__":
    print("Bootstrap Stabilization Test")
    print(f"시작: {datetime.now()}\n")

    # Test 1: Block size sensitivity
    block_results = test_block_size_sensitivity()

    # Test 2: Sample count convergence
    convergence_results = test_sample_count_convergence()

    # Test 3: OHLC consistency
    ohlc_valid = test_ohlc_consistency()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("1. Block Size: 테스트 1 결과 참조")
    print("2. Sample Count: 테스트 2 결과 참조")
    print(f"3. OHLC Consistency: {'✅ PASS' if ohlc_valid else '⚠️ FAIL'}")
    print(f"\n완료: {datetime.now()}")
