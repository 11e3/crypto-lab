"""
Phase 1 실행 스크립트: 과적합 검증 통합 실행.

이 스크립트는 다음을 순서대로 실행합니다:
1. Walk-Forward Analysis (OOS 성과 측정)
2. Parameter Robustness (파라미터 안정성)
3. Permutation Test (통계적 유의성)

결과물:
- reports/wfa_report.html
- reports/robustness_report.html
- reports/permutation_test_report.html
- reports/phase1_summary.md
"""

import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtester.engine import BacktestConfig  # noqa: E402
from src.backtester.permutation_test import PermutationTester  # noqa: E402
from src.backtester.robustness_analysis import RobustnessAnalyzer  # noqa: E402
from src.backtester.walk_forward_auto import WalkForwardAnalyzer  # noqa: E402
from src.strategies.volatility_breakout.vbo import VanillaVBO  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

# 로거 설정
logger = get_logger(__name__)


def load_test_data() -> pd.DataFrame:
    """
    테스트 데이터 로드.

    실제 운영 시 다음으로 교체:
    - data/processed/KRW-BTC.parquet 로드
    - 또는 Upbit API에서 직접 조회
    """
    logger.info("Loading test data...")

    # 테스트용 샘플 데이터 생성 (실제 운영에서는 실제 데이터 로드)
    try:
        data_path = project_root / "data" / "processed" / "KRW-BTC.parquet"
        if data_path.exists():
            data = pd.read_parquet(data_path)
            logger.info(f"Loaded data from {data_path}: {len(data)} rows")
            return data
    except Exception as e:
        logger.warning(f"Failed to load parquet data: {e}")

    # Fallback: 샘플 데이터 생성 (더 현실적인 시뮬레이션)
    logger.warning("Creating synthetic realistic data for testing...")
    np.random.seed(42)
    dates = pd.date_range(start="2018-01-01", end="2026-01-07", freq="D")

    # 등락을 포함한 현실적인 시뮬레이션
    returns = np.random.normal(0.0005, 0.02, len(dates))  # 일일 수익률: 평균 0.05%, std 2%
    prices = 1000 * np.exp(np.cumsum(returns))  # 누적 수익률 반영

    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "close": prices,
            "volume": np.random.randint(500000, 2000000, len(dates)),
        },
        index=dates,
    )

    # high >= close >= low 조건 확인
    data["high"] = data[["open", "close", "high"]].max(axis=1)
    data["low"] = data[["open", "close", "low"]].min(axis=1)

    return data


def run_walk_forward_analysis(data: pd.DataFrame, reports_dir: Path) -> dict:
    """Walk-Forward Analysis 실행."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1.1: Walk-Forward Analysis")
    logger.info("=" * 60)

    analyzer = WalkForwardAnalyzer(
        data=data,
        train_period=252 * 2,  # 2년
        test_period=252,  # 1년
        step=63,  # 3개월 롤링
    )

    def strategy_factory(params):
        return VanillaVBO(
            sma_period=int(params.get("sma_period", 4)),
            trend_sma_period=int(params.get("trend_sma_period", 8)),
            short_noise_period=int(params.get("short_noise_period", 4)),
            long_noise_period=int(params.get("long_noise_period", 8)),
        )

    config = BacktestConfig(initial_capital=1_000_000, fee_rate=0.0005, slippage_rate=0.0005)

    report = analyzer.run(
        strategy_factory=strategy_factory,
        param_ranges={
            "sma_period": [3, 4, 5],
            "trend_sma_period": [7, 8, 9],
            "short_noise_period": [3, 4, 5],
            "long_noise_period": [7, 8, 9],
        },
        backtest_config=config,
        verbose=True,
    )

    # HTML 리포트 저장
    report_path = reports_dir / "01_wfa_report.html"
    analyzer.export_report_html(report, report_path)
    logger.info(f"✅ WFA report saved to {report_path}")

    return {
        "wfa": report,
        "is_avg_return": report.in_sample_avg_return,
        "oos_avg_return": report.out_of_sample_avg_return,
        "overfitting_ratio": report.overfitting_ratio,
        "is_sharpe": report.in_sample_sharpe,
        "oos_sharpe": report.out_of_sample_sharpe,
    }


def run_robustness_analysis(data: pd.DataFrame, reports_dir: Path) -> dict:
    """Parameter Robustness 분석 실행."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1.2: Robustness Analysis")
    logger.info("=" * 60)

    def strategy_factory(params):
        return VanillaVBO(
            sma_period=int(params.get("sma_period", 4)),
            trend_sma_period=int(params.get("trend_sma_period", 8)),
        )

    config = BacktestConfig(initial_capital=1_000_000, fee_rate=0.0005, slippage_rate=0.0005)

    analyzer = RobustnessAnalyzer(data, strategy_factory, config)

    report = analyzer.analyze(
        optimal_params={"sma_period": 4, "trend_sma_period": 8},
        parameter_ranges={"sma_period": [2, 3, 4, 5, 6], "trend_sma_period": [6, 7, 8, 9, 10]},
        verbose=True,
    )

    # HTML 리포트 저장
    report_path = reports_dir / "02_robustness_report.html"
    analyzer.export_report_html(report, report_path)

    # CSV 저장
    csv_path = reports_dir / "02_robustness_results.csv"
    analyzer.export_to_csv(report, csv_path)

    logger.info(f"✅ Robustness report saved to {report_path}")
    logger.info(f"✅ Results CSV saved to {csv_path}")

    return {
        "robustness": report,
        "mean_return": report.mean_return,
        "std_return": report.std_return,
        "neighbor_success_rate": report.neighbor_success_rate,
        "sensitivity_scores": report.sensitivity_scores,
    }


def run_permutation_test(data: pd.DataFrame, reports_dir: Path) -> dict:
    """Permutation Test 실행."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1.3: Permutation Test (Statistical Validation)")
    logger.info("=" * 60)

    def strategy_factory():
        return VanillaVBO()

    config = BacktestConfig(initial_capital=1_000_000, fee_rate=0.0005, slippage_rate=0.0005)

    tester = PermutationTester(data, strategy_factory, config)

    result = tester.run(
        num_shuffles=100,  # 실제: 1000, 테스트: 100
        shuffle_columns=["close"],
        verbose=True,
    )

    # HTML 리포트 저장
    report_path = reports_dir / "03_permutation_test_report.html"
    tester.export_report_html(result, report_path)

    logger.info(f"✅ Permutation test report saved to {report_path}")

    return {
        "permutation": result,
        "z_score": result.z_score,
        "p_value": result.p_value,
        "is_significant": result.is_statistically_significant,
        "interpretation": result.interpretation,
    }


def generate_phase1_summary(
    reports_dir: Path, wfa_results: dict, robustness_results: dict, permutation_results: dict
) -> None:
    """Phase 1 종합 요약 리포트 생성."""
    logger.info("\n" + "=" * 60)
    logger.info("Generating Phase 1 Summary Report")
    logger.info("=" * 60)

    summary = f"""# Phase 1: 과적합 방지 - 종합 검증 리포트

**생성일**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Walk-Forward Analysis 결과

### 핵심 지표
- **In-Sample Avg Return**: {wfa_results["is_avg_return"]:.2%}
- **Out-of-Sample Avg Return**: {wfa_results["oos_avg_return"]:.2%}
- **Overfitting Ratio (OOS/IS)**: {wfa_results["overfitting_ratio"]:.2%}
- **In-Sample Sharpe**: {wfa_results["is_sharpe"]:.2f}
- **Out-of-Sample Sharpe**: {wfa_results["oos_sharpe"]:.2f}

### 해석
- OOS/IS 비율 > 0.3이면 **정상** (과적합 아님)
- OOS/IS 비율 0.1-0.3이면 **경고** (중간 수준 과적합)
- OOS/IS 비율 < 0.1이면 **위험** (심각한 과적합)

**판정**: {
        "✅ 정상 - 과적합 없음"
        if wfa_results["overfitting_ratio"] > 0.3
        else "⚠️ 경고 - 과적합 우려"
        if wfa_results["overfitting_ratio"] > 0.1
        else "❌ 위험 - 심각한 과적합"
    }

---

## 2. Robustness Analysis 결과

### 파라미터 안정성
- **Mean Return**: {robustness_results["mean_return"]:.2%}
- **Std Dev Return**: {robustness_results["std_return"]:.2%}
- **Neighbor Success Rate**: {robustness_results["neighbor_success_rate"]:.1%}

### 파라미터별 민감도
"""

    for param_name, sensitivity in robustness_results["sensitivity_scores"].items():
        interpretation = (
            "High (위험)"
            if sensitivity > 0.7
            else "Medium"
            if sensitivity > 0.4
            else "Low (안정적)"
        )
        summary += f"- **{param_name}**: {sensitivity:.3f} ({interpretation})\n"

    summary += f"""

### 해석
- Neighbor Success Rate > 70%이면 **강건함** (최적값 주변에서 안정적)
- Neighbor Success Rate 50-70%이면 **보통**
- Neighbor Success Rate < 50%이면 **취약함** (과적합 신호)

**판정**: {
        "✅ 강건함 - 파라미터 변화에 안정적"
        if robustness_results["neighbor_success_rate"] > 0.7
        else "⚠️ 보통 - 어느 정도 민감함"
        if robustness_results["neighbor_success_rate"] > 0.5
        else "❌ 취약함 - 과적합 우려"
    }

---

## 3. Permutation Test 결과

### 통계적 유의성
- **Z-score**: {permutation_results["z_score"]:.2f}
- **P-value**: {permutation_results["p_value"]:.4f}
- **Significance**: {permutation_results["z_score"] > 2.0 and "✅ Yes (p < 0.05)" or "❌ No"}

### 해석
- Z-score > 2.0이면 **통계적으로 유의함** (전략이 실제 신호 캡처)
- Z-score 1.0-2.0이면 **약하게 유의함** (신호 존재하나 과적합 우려)
- Z-score < 1.0이면 **유의하지 않음** (우연일 가능성 높음)

**판정**: {
        "✅ 유의함 - 전략의 신호 품질 우수"
        if permutation_results["z_score"] > 2.0
        else "⚠️ 약하게 유의함 - 신호 있으나 과적합 우려"
        if permutation_results["z_score"] > 1.0
        else "❌ 유의하지 않음 - 우연일 가능성 높음 (과적합 의심)"
    }

**메시지**: {permutation_results["interpretation"]}

---

## 4. 종합 판정

### 과적합 위험도 종합 평가

| 항목 | 상태 | 평가 |
|------|------|------|
| Walk-Forward (OOS/IS) | {wfa_results["overfitting_ratio"]:.1%} | {
        "✅"
        if wfa_results["overfitting_ratio"] > 0.3
        else "⚠️"
        if wfa_results["overfitting_ratio"] > 0.1
        else "❌"
    } |
| Robustness (Success Rate) | {robustness_results["neighbor_success_rate"]:.1%} | {
        "✅"
        if robustness_results["neighbor_success_rate"] > 0.7
        else "⚠️"
        if robustness_results["neighbor_success_rate"] > 0.5
        else "❌"
    } |
| Permutation (Z-score) | {permutation_results["z_score"]:.2f} | {
        "✅"
        if permutation_results["z_score"] > 2.0
        else "⚠️"
        if permutation_results["z_score"] > 1.0
        else "❌"
    } |

### 최종 결론

{
        "🎯 PASS: 전략에 유의한 신호가 있으며, 과적합 위험이 낮습니다. 실전 운영 가능 수준입니다."
        if (
            wfa_results["overfitting_ratio"] > 0.3
            and robustness_results["neighbor_success_rate"] > 0.7
            and permutation_results["z_score"] > 2.0
        )
        else "⚠️ WARNING: 과적합 또는 신호 품질 문제 감지. 전략 재검토 필요합니다."
    }

---

## 5. 다음 단계

1. **결과 검토**: 각 리포트(HTML) 상세 분석
2. **파라미터 조정**: 민감도 높은 파라미터 재최적화
3. **추가 검증**: 다른 암호화폐/기간에서 재테스트
4. **Phase 2 진행**: 노이즈 비율 및 슬리피지 안정화

---

## 상세 리포트

- [Walk-Forward Analysis Report](01_wfa_report.html)
- [Robustness Analysis Report](02_robustness_report.html)
- [Permutation Test Report](03_permutation_test_report.html)
"""

    summary_path = reports_dir / "00_phase1_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    logger.info(f"✅ Phase 1 summary saved to {summary_path}")
    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)


def main():
    """메인 실행 함수."""
    logger.info("🚀 Starting Phase 1: Overfitting Detection\n")

    # 리포트 디렉토리 생성
    reports_dir = project_root / "reports" / "phase1"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    data = load_test_data()

    # 2. Walk-Forward Analysis
    wfa_results = run_walk_forward_analysis(data, reports_dir)

    # 3. Robustness Analysis
    robustness_results = run_robustness_analysis(data, reports_dir)

    # 4. Permutation Test
    permutation_results = run_permutation_test(data, reports_dir)

    # 5. 종합 리포트 생성
    generate_phase1_summary(reports_dir, wfa_results, robustness_results, permutation_results)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Phase 1 Execution Complete!")
    logger.info(f"📁 Reports saved to: {reports_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
