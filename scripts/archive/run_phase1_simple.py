"""
Phase 1 간단한 실행 스크립트.

주요 3개 분석 모듈 실행:
1. Walk-Forward Analysis
2. Robustness Analysis
3. Permutation Test
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtester.engine import BacktestConfig  # noqa: E402
from src.backtester.permutation_test import PermutationTester  # noqa: E402
from src.backtester.robustness_analysis import RobustnessAnalyzer  # noqa: E402
from src.backtester.walk_forward_auto import WalkForwardAnalyzer  # noqa: E402
from src.strategies.volatility_breakout.vbo import VanillaVBO  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def create_sample_data() -> pd.DataFrame:
    """샘플 데이터 생성."""
    logger.info("Creating synthetic test data...")
    np.random.seed(42)

    dates = pd.date_range(start="2018-01-01", end="2026-01-07", freq="D")

    # 현실적인 가격 시뮬레이션
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 1000 * np.exp(np.cumsum(returns))

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

    logger.info(f"Created data: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    return data


def main():
    """메인 실행 함수."""
    logger.info("Starting Phase 1: Overfitting Detection")
    logger.info("=" * 70)

    # 데이터 로드/생성
    data = create_sample_data()

    # 출력 디렉토리
    reports_dir = Path("reports/phase1")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Walk-Forward Analysis
    logger.info("\n" + "=" * 70)
    logger.info("1. Walk-Forward Analysis (OOS 성과 측정)")
    logger.info("=" * 70)

    try:
        analyzer = WalkForwardAnalyzer(
            data=data,
            train_period=252 * 2,  # 2년
            test_period=252,  # 1년
            step=63,  # 3개월
        )

        def vbo_factory(params):
            return VanillaVBO(
                sma_period=int(params.get("sma_period", 4)),
                trend_sma_period=int(params.get("trend_sma_period", 8)),
            )

        config = BacktestConfig(initial_capital=1_000_000, fee_rate=0.0005, slippage_rate=0.0005)

        wfa_report = analyzer.run(
            strategy_factory=vbo_factory,
            param_ranges={"sma_period": [3, 4, 5], "trend_sma_period": [7, 8, 9]},
            backtest_config=config,
            verbose=True,
        )

        logger.info(f"  IS 평균수익: {wfa_report.in_sample_avg_return:.2%}")
        logger.info(f"  OOS 평균수익: {wfa_report.out_of_sample_avg_return:.2%}")
        logger.info(f"  과적합비율: {wfa_report.overfitting_ratio:.2%}")

        # HTML 리포트 저장
        report_path = reports_dir / "01_wfa_report.html"
        html_content = analyzer._generate_html(wfa_report)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"  WFA 리포트 저장: {report_path}")

    except Exception as e:
        logger.error(f"WFA 실행 실패: {e}", exc_info=True)
        return 1

    # 2. Robustness Analysis
    logger.info("\n" + "=" * 70)
    logger.info("2. Robustness Analysis (파라미터 안정성)")
    logger.info("=" * 70)

    try:
        analyzer = RobustnessAnalyzer(data, vbo_factory, config)

        robustness_report = analyzer.analyze(
            optimal_params={"sma_period": 4, "trend_sma_period": 8},
            parameter_ranges={"sma_period": [2, 3, 4, 5, 6], "trend_sma_period": [6, 7, 8, 9, 10]},
            verbose=True,
        )

        logger.info(f"  이웃 성공률: {robustness_report.neighbor_success_rate:.1%}")
        logger.info(f"  평균 수익: {robustness_report.mean_return:.2%}")

        # CSV 저장
        csv_path = reports_dir / "02_robustness_results.csv"
        results_df = pd.DataFrame([r.to_dict() for r in robustness_report.results])
        results_df.to_csv(csv_path, index=False)
        logger.info(f"  결과 CSV 저장: {csv_path}")

    except Exception as e:
        logger.error(f"Robustness 실행 실패: {e}", exc_info=True)
        return 1

    # 3. Permutation Test
    logger.info("\n" + "=" * 70)
    logger.info("3. Permutation Test (통계적 유의성)")
    logger.info("=" * 70)

    try:
        tester = PermutationTester(data, lambda: VanillaVBO(), config)

        perm_result = tester.run(
            num_shuffles=100,  # 실제: 1000
            shuffle_columns=["close", "volume"],
            verbose=True,
        )

        logger.info(f"  원본 수익: {perm_result.original_return:.2%}")
        logger.info(f"  Z-score: {perm_result.z_score:.2f}")
        logger.info(f"  p-value: {perm_result.p_value:.4f}")
        logger.info(f"  유의성: {perm_result.interpretation}")

        # 결과 저장
        result_file = reports_dir / "03_permutation_result.txt"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write("Permutation Test Results\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Original Return: {perm_result.original_return:.2%}\n")
            f.write(f"Mean Shuffled Return: {perm_result.mean_shuffled_return:.2%}\n")
            f.write(f"Z-score: {perm_result.z_score:.2f}\n")
            f.write(f"P-value: {perm_result.p_value:.4f}\n")
            f.write(f"Significant: {perm_result.is_statistically_significant}\n")
            f.write(f"Interpretation: {perm_result.interpretation}\n")
        logger.info(f"  결과 저장: {result_file}")

    except Exception as e:
        logger.error(f"Permutation 실행 실패: {e}", exc_info=True)
        return 1

    # 최종 결과 요약
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1 완료: 과적합 검증 결과 요약")
    logger.info("=" * 70)

    results = []

    # WFA 검증
    wfa_pass = wfa_report.overfitting_ratio > 0.3
    results.append(
        f"WFA (OOS/IS > 0.3): {'PASS' if wfa_pass else 'FAIL'} ({wfa_report.overfitting_ratio:.2%})"
    )

    # Robustness 검증
    rob_pass = robustness_report.neighbor_success_rate > 0.7
    results.append(
        f"Robustness (이웃성공률 > 70%): {'PASS' if rob_pass else 'FAIL'} ({robustness_report.neighbor_success_rate:.1%})"
    )

    # Permutation 검증
    perm_pass = perm_result.z_score > 2.0
    results.append(
        f"Permutation (Z-score > 2.0): {'PASS' if perm_pass else 'FAIL'} ({perm_result.z_score:.2f})"
    )

    for result in results:
        logger.info(f"  {result}")

    all_pass = wfa_pass and rob_pass and perm_pass
    logger.info(f"\n전체 결과: {'ALL PASS - 신뢰성 높음' if all_pass else '개선 필요'}")
    logger.info(f"리포트 저장: {reports_dir}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
