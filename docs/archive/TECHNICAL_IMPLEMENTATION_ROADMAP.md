# 퀀트 피드백 해결 - 기술 구현 로드맵

## 🔧 Phase 1: 과적합 방지 메커니즘 (우선순위 1순위)

### Task 1.1: Walk-Forward Analysis 자동화
**담당**: 백테스팅 엔지니어  
**예상 기간**: 3일  
**검증**: OOS 성과 리포트 자동 생성

```python
# src/backtester/walk_forward_auto.py (신규)

class AutomatedWalkForwardAnalysis:
    """
    자동화된 Walk-forward 분석
    """
    def __init__(self, 
                 data: pd.DataFrame,
                 train_period: int = 252 * 2,  # 2년
                 test_period: int = 252,  # 1년
                 step: int = 63):  # 3개월 롤링
        self.data = data
        self.train_period = train_period
        self.test_period = test_period
        self.step = step
    
    def run(self) -> WalkForwardReport:
        """
        Walk-forward 자동 실행
        
        Returns:
            {
                'in_sample': [성과1, 성과2, ...],
                'out_of_sample': [성과1, 성과2, ...],
                'overfitting_ratio': OOS/IS,  # < 0.7이면 심각한 과적합
                'parameter_stability': parameter_heatmap,
            }
        """
        results = []
        
        for i in range(0, len(self.data) - self.train_period - self.test_period, self.step):
            train_data = self.data.iloc[i:i+self.train_period]
            test_data = self.data.iloc[i+self.train_period:i+self.train_period+self.test_period]
            
            # 1. Training 구간에서 파라미터 최적화
            optimal_params = self._optimize_params(train_data)
            
            # 2. Test 구간에서 OOS 성과 측정
            is_result = self._backtest(train_data, optimal_params)
            oos_result = self._backtest(test_data, optimal_params)
            
            results.append({
                'period': f"{train_data.index[0]:%Y-%m-%d} ~ {test_data.index[-1]:%Y-%m-%d}",
                'in_sample': is_result,
                'out_of_sample': oos_result,
                'params': optimal_params,
            })
        
        return self._aggregate_results(results)

# 실행 스크립트
if __name__ == "__main__":
    # 8년 데이터 로드
    data = pd.read_parquet("data/processed/KRW-BTC.parquet")
    
    # Walk-forward 분석 실행
    wf = AutomatedWalkForwardAnalysis(data)
    report = wf.run()
    
    # 결과 분석
    print(f"In-Sample Avg Return: {report['is_avg']:.2%}")
    print(f"Out-of-Sample Avg Return: {report['oos_avg']:.2%}")
    print(f"Overfitting Ratio: {report['oos_avg']/report['is_avg']:.2%}")
    
    # 기대값: 0.5-0.7 (30-50% 악화는 정상)
    # 문제: < 0.2라면 심각한 과적합
    
    if report['oos_avg']/report['is_avg'] < 0.2:
        raise OverfittingError("심각한 과적합 감지!")
```

**검증 기준**:
- ✅ OOS/IS 비율 > 0.3 (과적합 아님)
- ✅ OOS 수익률이 양수 (손실이 아님)
- ✅ OOS Sharpe > 0 (통계적 의미 있음)

**결과 활용**:
- README에 "OOS 성과: X%" 공개
- 38,331% → 현실 수치로 수정

---

### Task 1.2: Parameter Robustness 분석
**담당**: 백테스팅 엔지니어  
**예상 기간**: 2일

```python
# src/backtester/robustness_analysis.py (신규)

class RobustnessAnalyzer:
    """
    파라미터 감도 분석
    """
    def analyze(self, 
                optimal_params: dict,
                parameter_ranges: dict) -> HeatmapReport:
        """
        최적 파라미터 주변에서 성과 변화 분석
        
        예: sma_period = 4 (최적)
            → [1, 2, 3, 4, 5, 6, 7] 범위에서 각각 백테스트
            → 그래프: 파라미터 vs 성과
        """
        
# 사용 예:
analyzer = RobustnessAnalyzer()

# sma_period = 4 (최적값), ±50% 변화 테스트
report = analyzer.analyze(
    optimal_params={'sma_period': 4, 'noise_period': 8},
    parameter_ranges={
        'sma_period': [2, 3, 4, 5, 6],  # ±50%
        'noise_period': [4, 6, 8, 10, 12],  # ±50%
    }
)

# 결과:
# - 파라미터 변화 시 성과 곡선이 부드러운가?
# - 최적값에서 벗어나면 급격히 나빠지는가?
# - 평평한 곡선 = 과적합 위험 신호
```

**검증 기준**:
- ✅ 파라미터 ±20% 변화 → 성과 ±10% 이내 (안정적)
- ❌ 파라미터 ±20% 변화 → 성과 ±50% 이상 (불안정 = 과적합)

---

### Task 1.3: Synthetic Data Permutation Test
**담당**: 통계 분석가  
**예상 기간**: 3일

```python
# tests/test_overfitting_detection.py (신규)

class TestOverfittingDetection:
    """
    과적합 여부 통계적 검증
    """
    def test_permutation_invariance(self):
        """
        데이터를 무작위로 섞었을 때도 같은 수익이 나오는가?
        → YES라면 과적합 (데이터 피킹)
        → NO라면 정상 (실제 시그널 캡처)
        """
        
        # 1. 원본 데이터 백테스트
        original_result = backtest(original_data, strategy)
        
        # 2. 데이터 셔플 1000회 반복
        shuffled_results = []
        for i in range(1000):
            shuffled_data = original_data.copy()
            shuffled_data['close'] = np.random.permutation(shuffled_data['close'])
            result = backtest(shuffled_data, strategy)
            shuffled_results.append(result.total_return)
        
        # 3. 통계 검증
        mean_shuffled = np.mean(shuffled_results)
        std_shuffled = np.std(shuffled_results)
        z_score = (original_result.total_return - mean_shuffled) / std_shuffled
        
        # 기대값: z_score > 2.0 (5% 유의수준)
        # 의미: 원본 성과가 우연에 비해 통계적으로 유의
        assert z_score > 2.0, "No statistical significance detected (likely overfitting)"
        
        print(f"Z-score: {z_score:.2f} ✓ (통계적으로 유의)")
```

---

## 🔩 Phase 2: 노이즈 비율 및 슬리피지 안정화 (우선순위 1순위)

### Task 2.1: 노이즈 비율 경계 조건 강화
**담당**: 핵심 엔지니어  
**예상 기간**: 2일

```python
# src/utils/indicators_v2.py

import numpy as np
import pandas as pd
from typing import Tuple

def noise_ratio_stable(
    open_: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    min_range: float = 1e-8,
    outlier_percentile: float = 99.9
) -> Tuple[pd.Series, dict]:
    """
    안정화된 노이즈 비율 계산
    
    Args:
        open_, high, low, close: OHLC 데이터
        min_range: 최소 high-low 범위 (극소값 처리)
        outlier_percentile: 이상치 판단 기준 (99.9%)
    
    Returns:
        (noise_ratio, diagnostics)
    """
    # 1. 입력 검증
    assert len(open_) == len(high) == len(low) == len(close), \
        "All inputs must have same length"
    
    # 2. High-Low 범위 계산
    range_hl = high - low
    
    # 3. 이상치 감지 (범위 너무 큼)
    threshold_high = np.percentile(range_hl[range_hl > 0], outlier_percentile)
    mask_extreme_range = range_hl > threshold_high
    
    # 4. 극소값 처리 (range 거의 0)
    range_hl_safe = np.maximum(range_hl, min_range)
    
    # 5. 노이즈 비율 계산
    noise = 1.0 - np.abs(close - open_) / range_hl_safe
    
    # 6. 범위 클리핑 (오류 방지)
    noise = np.clip(noise, 0.0, 1.0)
    
    # 7. 이상치 처리 (extreme range 구간)
    # 옵션 1: Forward fill
    noise_fixed = pd.Series(noise)
    noise_fixed[mask_extreme_range] = np.nan
    noise_fixed = noise_fixed.fillna(method='ffill').fillna(0.5)
    
    # 8. NaN/Inf 최종 처리
    noise_fixed = noise_fixed.fillna(0.5)
    noise_fixed = np.isfinite(noise_fixed).astype(float) * noise_fixed + \
                   (~np.isfinite(noise_fixed)).astype(float) * 0.5
    
    # 9. 진단 정보
    diagnostics = {
        'nan_count': np.isnan(noise).sum(),
        'outlier_count': mask_extreme_range.sum(),
        'zero_range_count': (range_hl == 0).sum(),
        'outlier_ratio': mask_extreme_range.sum() / len(range_hl),
        'mean_noise': float(np.mean(noise_fixed)),
        'std_noise': float(np.std(noise_fixed)),
    }
    
    return noise_fixed, diagnostics
```

### Task 2.2: 동적 슬리피지 모델
**담당**: 리스크 엔지니어  
**예상 기간**: 3일

```python
# src/backtester/slippage_model_v2.py (신규)

class DynamicSlippageModel:
    """
    자본규모 및 시장 상황에 따른 동적 슬리피지
    """
    
    # 기본 설정 (Upbit 기준)
    CONFIG = {
        'base_slippage_bps': 5,  # 0.05% 기본
        'maker_fee_bps': 2,      # 0.02% 메이커 수수료
        'taker_fee_bps': 5,      # 0.05% 테이커 수수료
    }
    
    # 자본 규모별 추가 슬리피지
    CAPITAL_TIERS = {
        'micro': {'max': 5_000_000, 'slippage_bps': 20},      # 추가 0.2%
        'small': {'max': 50_000_000, 'slippage_bps': 10},     # 추가 0.1%
        'medium': {'max': 500_000_000, 'slippage_bps': 5},    # 추가 0.05%
        'large': {'max': float('inf'), 'slippage_bps': 0},    # 추가 없음
    }
    
    # 주문 크기 비율별 슬리피지
    ORDER_SIZE_MULTIPLIERS = {
        0.001: 1.0,    # 0.1% 주문 → 1배
        0.005: 1.5,    # 0.5% 주문 → 1.5배
        0.01: 2.5,     # 1% 주문 → 2.5배
        0.02: 4.0,     # 2% 주문 → 4배
    }
    
    def calculate(self,
                  order_amount: float,
                  portfolio_value: float,
                  daily_volume: float,
                  side: str = 'buy') -> float:
        """
        동적 슬리피지 계산
        
        Args:
            order_amount: 주문 금액
            portfolio_value: 포트폴리오 총액
            daily_volume: 일일 거래량
            side: 'buy' or 'sell'
        
        Returns:
            슬리피지 (% 단위, 예: 0.001 = 0.1%)
        """
        # 1. 기본 수수료
        if side == 'buy':
            base_slippage = self.CONFIG['taker_fee_bps'] / 10000
        else:
            base_slippage = self.CONFIG['taker_fee_bps'] / 10000
        
        # 2. 자본규모별 추가 슬리피지
        for tier_name, tier_config in self.CAPITAL_TIERS.items():
            if portfolio_value <= tier_config['max']:
                capital_slippage = tier_config['slippage_bps'] / 10000
                break
        
        # 3. 주문 크기 비율 (Order Size / Daily Volume)
        order_size_ratio = order_amount / (daily_volume + 1e-10)
        
        # 주문 크기별 승수 계산
        size_multiplier = 1.0
        for ratio_threshold in sorted(self.ORDER_SIZE_MULTIPLIERS.keys()):
            if order_size_ratio <= ratio_threshold:
                size_multiplier = self.ORDER_SIZE_MULTIPLIERS[ratio_threshold]
                break
        else:
            # 2% 초과 → 추가 페널티
            size_multiplier = 5.0 + (order_size_ratio - 0.02) * 10
        
        # 4. 총 슬리피지
        total_slippage = (base_slippage + capital_slippage) * size_multiplier
        
        # 5. 매도 시 추가 (호가창 불리함)
        if side == 'sell':
            total_slippage *= 1.1
        
        # 6. 상한선 설정 (극단적 상황 방지)
        total_slippage = min(total_slippage, 0.05)  # 최대 5%
        
        return total_slippage

# 테스트 케이스
slippage_calc = DynamicSlippageModel()

# Case 1: 소액 자본 (1천만원), 소액 주문
slippage1 = slippage_calc.calculate(
    order_amount=1_000_000,
    portfolio_value=10_000_000,
    daily_volume=100_000_000
)
print(f"Micro capital, small order: {slippage1:.2%}")  # 예상: 0.3-0.5%

# Case 2: 대액 자본 (1억원), 소액 주문
slippage2 = slippage_calc.calculate(
    order_amount=1_000_000,
    portfolio_value=100_000_000,
    daily_volume=100_000_000
)
print(f"Large capital, small order: {slippage2:.2%}")  # 예상: 0.1%
```

---

## 📊 Phase 3: 테스트 강화 (우선순위 2순위)

### Task 3.1: Edge Case 테스트 100개 추가
**담당**: QA 엔지니어  
**예상 기간**: 3일

```python
# tests/unit/test_edge_cases_comprehensive.py (신규)

class TestEdgeCasesComprehensive:
    """
    극한 상황 테스트 (100개 케이스)
    """
    
    # 그룹 1: 데이터 극값 (20개)
    def test_zero_range_single_row(self):
        """High == Low == Close == Open (range = 0)"""
    
    def test_zero_range_multiple_rows(self):
        """연속 5일 같은 가격"""
    
    def test_extreme_gap_up(self):
        """Open << Previous Close (100배 갭)"""
    
    def test_extreme_gap_down(self):
        """Open >> Previous Close"""
    
    def test_nan_and_inf_mixed(self):
        """NaN, Inf, -Inf 섞인 데이터"""
    
    def test_single_row_dataframe(self):
        """1행 데이터 (SMA 계산 불가)"""
    
    def test_empty_dataframe(self):
        """빈 데이터프레임"""
    
    def test_all_zero_volume(self):
        """모든 거래량 0"""
    
    def test_negative_prices(self):
        """음수 가격 (오류 데이터)"""
    
    def test_price_overflow(self):
        """float64 최대값에 가까운 가격"""
    
    # ... 계속 20개
    
    # 그룹 2: 포지션 극값 (20개)
    def test_zero_position_size(self):
        """0 크기 포지션"""
    
    def test_fraction_position(self):
        """0.00001개 매수 (극소 포지션)"""
    
    def test_million_units_position(self):
        """100만 개 매수 (극대 포지션)"""
    
    # ... 계속 18개
    
    # 그룹 3: 슬리피지 극값 (15개)
    def test_zero_daily_volume(self):
        """일일 거래량 0 (division by zero)"""
    
    def test_order_size_exceeds_volume(self):
        """주문액 > 일일 거래량"""
    
    # ... 계속 13개
    
    # 그룹 4: 시간 극값 (20개)
    def test_leap_year_feb29(self):
        """윤년 2월 29일"""
    
    def test_midnight_boundary(self):
        """자정 기준 데이터"""
    
    # ... 계속 18개
    
    # 그룹 5: 통계 극값 (25개)
    def test_infinite_sharpe_ratio(self):
        """모든 일일 수익이 음수 (Sharpe = -∞)"""
    
    def test_division_by_zero_in_metrics(self):
        """분모가 0이 되는 경우"""
    
    # ... 계속 23개
```

### Task 3.2: 부동소수점 정밀도 테스트
**담당**: 수치 해석 전문가  
**예상 기간**: 2일

```python
# tests/unit/test_floating_point_precision.py (신규)

class TestFloatingPointPrecision:
    """
    부동소수점 오차로 인한 거래 손실 방지
    """
    
    def test_cumulative_commission_error(self):
        """
        1000회 거래 시 누적 수수료 오차 검증
        
        이론:
        1000회 × 0.0005 (0.05%) = 0.5
        
        수치 계산:
        반복문으로 1000회 누적할 시 오차 발생 가능
        """
        
        # 직접 계산
        direct = 1000 * 0.0005
        
        # 반복 계산
        iterative = sum([0.0005 for _ in range(1000)])
        
        # 오차 검증
        assert abs(direct - iterative) < 1e-10, \
            f"Cumulative error: {abs(direct - iterative)}"
    
    def test_sma_calculation_accuracy(self):
        """
        SMA 계산 시 부동소수점 오차
        
        예: [1.1, 2.2, 3.3] SMA(2)
        수동: (1.1 + 2.2) / 2 = 1.65
        pandas: 같은 결과
        """
        data = pd.Series([1.1, 2.2, 3.3])
        sma_manual = (data[0] + data[1]) / 2
        sma_pandas = data.rolling(2).mean()[1]
        
        assert abs(sma_manual - sma_pandas) < 1e-14
    
    def test_portfolio_value_calculation_chain(self):
        """
        Buy → Sell → Buy → Sell ... 반복 시 누적 오차
        
        각 거래에서:
        1. 포트폴리오 값 = 현금 + 보유 자산 가치
        2. 소수점 처리 오류 누적
        """
        
        portfolio_value = 1_000_000.0
        price = 50_000.0
        
        for i in range(100):
            # Buy
            quantity = 1.0  # 1 unit
            cost = quantity * price * 1.0005  # 0.05% 수수료
            portfolio_value -= cost
            
            # Sell
            revenue = quantity * price * 0.9995  # 0.05% 수수료
            portfolio_value += revenue
        
        # 100회 왕복 후 원금 회복 확인
        # (가격 불변, 수수료만 손실)
        expected_loss = 1_000_000 * 100 * 0.001  # 0.1% × 100회
        expected_value = 1_000_000 - expected_loss
        
        assert abs(portfolio_value - expected_value) < 1.0, \
            f"Portfolio error too large: {abs(portfolio_value - expected_value)}"
```

---

## 🚀 Phase 4: 운영 체계 (우선순위 3순위)

### Task 4.1: State Machine + Circuit Breaker 구현
**담당**: 시스템 아키텍트  
**예상 기간**: 5일

```python
# src/execution/order_state_machine_impl.py

from enum import Enum, auto
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OrderState(Enum):
    """주문 상태 정의"""
    CREATED = auto()
    PENDING = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    FAILED = auto()
    ERROR = auto()

class OrderStateMachine:
    """
    엄격한 상태 관리 머신
    """
    VALID_TRANSITIONS = {
        OrderState.CREATED: {OrderState.PENDING, OrderState.FAILED},
        OrderState.PENDING: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, 
                           OrderState.CANCELLED, OrderState.ERROR},
        OrderState.PARTIALLY_FILLED: {OrderState.FILLED, OrderState.CANCELLED, 
                                     OrderState.ERROR},
        OrderState.FILLED: set(),  # Terminal
        OrderState.CANCELLED: set(),  # Terminal
        OrderState.FAILED: set(),  # Terminal
        OrderState.ERROR: {OrderState.PENDING},  # Retry가능
    }
    
    def __init__(self, order_id: str, initial_state: OrderState = OrderState.CREATED):
        self.order_id = order_id
        self.state = initial_state
        self.history = [(initial_state, "initialized", datetime.now())]
        logger.info(f"Order {order_id}: {initial_state.name}")
    
    def transition(self, new_state: OrderState, reason: str = "") -> bool:
        """
        상태 전이 시도
        
        Returns:
            True if successful, raises exception otherwise
        """
        if new_state not in self.VALID_TRANSITIONS.get(self.state, set()):
            msg = f"Order {self.order_id}: Invalid transition {self.state.name} → {new_state.name}"
            logger.error(msg)
            raise ValueError(msg)
        
        self.state = new_state
        self.history.append((new_state, reason, datetime.now()))
        logger.info(f"Order {self.order_id}: {new_state.name} ({reason})")
        return True

# Circuit Breaker 구현
class CircuitBreakerState(Enum):
    CLOSED = auto()      # 정상
    OPEN = auto()        # 차단
    HALF_OPEN = auto()   # 복구 중

class CircuitBreaker:
    """
    자동 장애 차단 시스템
    """
    def __init__(self, 
                 failure_threshold: int = 5,
                 success_threshold: int = 2,
                 timeout_sec: float = 60.0):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_sec = timeout_sec
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
        logger.info("Circuit breaker initialized")
    
    def call(self, func, *args, **kwargs):
        """
        Circuit breaker를 통한 함수 호출
        """
        if self.state == CircuitBreakerState.OPEN:
            elapsed = datetime.now() - self.last_failure_time
            if elapsed.total_seconds() > self.timeout_sec:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.warning("Circuit breaker: Attempting recovery")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit is OPEN. Retry after {self.timeout_sec}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """성공 처리"""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker: CLOSED (recovered)")
                self.success_count = 0
    
    def _on_failure(self):
        """실패 처리"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(
                f"Circuit breaker: OPEN "
                f"({self.failure_count} failures detected)"
            )
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
```

---

## 📈 완성도 체크리스트

### Phase 1: 과적합 방지 (2주)
- [ ] Task 1.1: Walk-Forward 자동화 완료
- [ ] Task 1.2: Parameter robustness 리포트 완료
- [ ] Task 1.3: Permutation 테스트 완료
- [ ] 테스트 통과: 모든 OOS 테스트 PASS
- [ ] 문서 작성: Walk-forward 결과 리포트
- [ ] README 수정: 실제 OOS 수익률 공개

### Phase 2: 안정성 (1-2주)
- [ ] Task 2.1: 노이즈 비율 강화 완료
- [ ] Task 2.2: 동적 슬리피지 완료
- [ ] 백테스트 재실행: 새 슬리피지 모델 적용
- [ ] 테스트 통과: 모든 안정성 테스트 PASS
- [ ] 비교 리포트: "기존 vs 새 슬리피지" 결과

### Phase 3: 검증 강화 (1-3주)
- [ ] Task 3.1: 100개 edge case 테스트 작성
- [ ] Task 3.2: 부동소수점 정밀도 테스트 작성
- [ ] 테스트 커버리지: > 85%
- [ ] 테스트 통과: 모든 새 테스트 PASS

### Phase 4: 운영 개선 (3-8주)
- [ ] Task 4.1: State machine + Circuit breaker 구현
- [ ] 모니터링 대시보드 구축
- [ ] 알림 시스템 완성
- [ ] 다중 거래소 지원 시작

---

**작성 날짜**: 2026년 1월 7일  
**상태**: 🟠 구현 준비 중  
**다음 단계**: Phase 1 시작 (Task 1.1)
