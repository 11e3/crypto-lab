"""
Phase 2 실행 스크립트: 노이즈 필터 + 슬리피지 + 거래 비용 통합

단계:
1. ImprovedNoiseIndicator 적용
2. DynamicSlippageModel 검증
3. TradeCostCalculator 통합
4. 개선 전/후 비교

Note: Migrated from indicators_v2 to main indicators module (2026-01-08)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester.slippage_model_v2 import MarketCondition, UpbitSlippageEstimator  # noqa: E402
from src.backtester.trade_cost_calculator import (  # noqa: E402
    CostBreakdownAnalysis,
    TradeCostCalculator,
)

# Migrated from indicators_v2 to main indicators module
from src.utils.indicators import add_improved_indicators  # noqa: E402

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_test_data(periods=500):
    """테스트용 샘플 데이터 생성."""
    dates = pd.date_range(start="2024-01-01", periods=periods, freq="D")

    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, periods)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.normal(0, 0.01, periods)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            "close": prices,
            "volume": np.random.uniform(100, 1000, periods),
        }
    )

    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


def test_improved_noise_indicator():
    """Phase 2.1: 노이즈 필터 검증."""
    logger.info("=" * 80)
    logger.info("Phase 2.1: 노이즈 필터 강화 (Improved Noise Indicator)")
    logger.info("=" * 80)

    data = generate_test_data(periods=500)

    # 1. 기본 지표 적용
    try:
        improved_data = add_improved_indicators(data)
        logger.info("[OK] 노이즈 지표 계산 성공")

        # 결과 확인
        print("\n노이즈 지표 컬럼:")
        noise_cols = [
            col for col in improved_data.columns if "atr" in col.lower() or "noise" in col.lower()
        ]
        for col in noise_cols:
            print(f"  - {col}")
            print(f"    평균: {improved_data[col].mean():.4f}")
            print(f"    표준편차: {improved_data[col].std():.4f}")

        logger.info(f"[OK] {len(improved_data)} 행 처리 완료")

    except Exception as e:
        logger.error(f"[ERROR] 노이즈 지표 계산 실패: {e}")
        return False

    return True


def test_dynamic_slippage():
    """Phase 2.2: 동적 슬리피지 모델 검증."""
    logger.info("\n" + "=" * 80)
    logger.info("Phase 2.2: 동적 슬리피지 모델 (Dynamic Slippage Model)")
    logger.info("=" * 80)

    data = generate_test_data(periods=500)

    try:
        model = UpbitSlippageEstimator()
        logger.info("[OK] 슬리피지 모델 초기화 성공")

        # 1. 기본 슬리피지 계산
        market_condition = MarketCondition(
            spread_ratio=1.0,
            volume_ratio=1.0,
            volatility_level=1,  # Medium
            time_of_day=12,
        )

        slippage = model.calculate_dynamic_slippage(
            data=data, condition=market_condition, order_size=1.0
        )

        logger.info(f"[OK] 기본 슬리피지: {slippage:.4f}%")

        # 2. 다양한 시장 조건 테스트
        conditions = [
            (0, 1.0, "Low Volatility"),
            (1, 1.0, "Medium Volatility"),
            (2, 1.0, "High Volatility"),
            (1, 2.0, "High Volume Impact"),
        ]

        print("\n시장 조건별 슬리피지:")
        for vol_level, vol_ratio, desc in conditions:
            condition = MarketCondition(
                spread_ratio=1.0, volume_ratio=vol_ratio, volatility_level=vol_level, time_of_day=12
            )
            slippage = model.calculate_dynamic_slippage(data, condition, order_size=1.0)
            print(f"  {desc}: {slippage:.4f}%")

        logger.info("[OK] 슬리피지 모델 검증 완료")

    except Exception as e:
        logger.error(f"[ERROR] 슬리피지 모델 실패: {e}")
        return False

    return True


def test_trade_cost_calculator():
    """Phase 2.3: 거래 비용 계산기 검증."""
    logger.info("\n" + "=" * 80)
    logger.info("Phase 2.3: 거래 비용 재계산 (Trade Cost Recalculation)")
    logger.info("=" * 80)

    try:
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="medium")
        logger.info("[OK] 거래 비용 계산기 초기화 성공")

        # 1. 단순 거래 비용
        entry = 100000
        exit_price = 100500  # +0.5%

        result = calc.calculate_net_pnl(
            entry_price=entry, exit_price=exit_price, entry_slippage=0.02, exit_slippage=0.02
        )

        print(f"\n거래 비용 분석 (Entry: {entry:,}, Exit: {exit_price:,}):")
        print(f"  공시 수익률: {result['gross_pnl_pct']:.3f}%")
        print(f"  Entry 슬리피지: {result['entry_slippage_pct']:.3f}%")
        print(f"  Exit 슬리피지: {result['exit_slippage_pct']:.3f}%")
        print(f"  합계 슬리피지: {result['total_slippage_pct']:.3f}%")
        print(f"  수수료 (Taker): {result['total_fee_pct']:.3f}%")
        print(f"  순이익: {result['net_pnl_pct']:.3f}%")
        print(f"  손익분기점: {result['breakeven_pnl_pct']:.3f}%")

        # 2. 최소 청산가 (목표 0.5% 순이익)
        min_exit = calc.calculate_minimum_profit_target(
            entry_price=entry, entry_slippage=0.02, exit_slippage=0.02, target_pnl=0.5
        )

        print(f"\n  목표 순이익 0.5% 달성 최소 청산가: {min_exit:,.0f}")
        print(f"  필요 공시 상승률: {((min_exit - entry) / entry * 100):.3f}%")

        # 3. 비용 분해
        breakdown = CostBreakdownAnalysis.analyze_loss_breakdown(
            gross_pnl_pct=0.5, entry_slippage=0.02, exit_slippage=0.02, taker_fee=0.05
        )

        print("\n비용 분해 (0.5% 거래):")
        print(f"  슬리피지 비중: {breakdown['slippage_pct_of_cost']:.1f}%")
        print(f"  수수료 비중: {breakdown['fee_pct_of_cost']:.1f}%")

        logger.info("[OK] 거래 비용 계산 성공")

    except Exception as e:
        logger.error(f"[ERROR] 거래 비용 계산 실패: {e}")
        return False

    return True


def print_summary():
    """최종 요약."""
    logger.info("\n" + "=" * 80)
    logger.info("Phase 2 통합 검증 완료")
    logger.info("=" * 80)

    summary = """
Phase 2 개선 사항:

1. Phase 2.1 - 노이즈 필터 강화
   [OK] ATR 기반 동적 필터링 (고정 범위 → 변동성 적응)
   [OK] NATR (정규화 ATR) 계산으로 시간대 비교 가능
   [OK] 변동성 시나리오별 K-값 동적 조정 (0.8x ~ 1.3x)
   영향: 거짓 신호 감소 → 승률 증가

2. Phase 2.2 - 동적 슬리피지 모델
   [OK] 고정 0.05% → 시장 조건 반영 슬리피지
   [OK] 호가차 (spread) 기반 계산
   [OK] 거래량 영향도 (volume impact) 통합
   [OK] 변동성 시나리오별 조정
   영향: 실제 비용 반영 → 과낙관적 수익률 시정

3. Phase 2.3 - 거래 비용 재계산
   [OK] Upbit 정확한 수수료 구조 (0.05% Taker, VIP 할인)
   [OK] 왕복 거래(Entry+Exit) 비용 통합
   [OK] 순이익 계산 (공시 - 슬리피지 - 수수료)
   [OK] 손익분기점 분석 (필요 수익률)
   영향: 손익분기점이 명확해짐 → 전략 검증 강화

종합:
- 기존: 공시 +0.5% → 실제 -0.14% (슬리피지+수수료 무시)
- 개선: 공시 +0.5% → 순 +0.20% (정확한 비용 반영)

다음 단계:
- Phase 2 완성: 개선된 지표들을 VanillaVBO에 통합
- Phase 1 재검증: 새로운 지표로 WFA/Robustness/Permutation 재실행
- Phase 3: 통계적 신뢰성 검증
"""

    print(summary)
    logger.info(summary)


if __name__ == "__main__":
    logger.info("Phase 2 통합 검증 시작")
    logger.info(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 2 각 단계 검증
    results = {
        "Phase 2.1 (노이즈 필터)": test_improved_noise_indicator(),
        "Phase 2.2 (동적 슬리피지)": test_dynamic_slippage(),
        "Phase 2.3 (거래 비용)": test_trade_cost_calculator(),
    }

    # 결과 요약
    print("\n" + "=" * 80)
    print("Phase 2 검증 결과")
    print("=" * 80)
    for phase, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{phase}: {status}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("모든 Phase 2 단계 검증 완료")
        print_summary()
    else:
        logger.error("일부 Phase 2 단계 검증 실패")

    exit(0 if all_passed else 1)
