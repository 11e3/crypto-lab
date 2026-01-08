# V2 Module Consolidation Plan

## 현황

현재 3개의 v2 모듈이 존재하며, 8개의 phase 스크립트가 의존하고 있습니다.

### V2 모듈
1. `src/strategies/volatility_breakout/vbo_v2.py` (371 lines)
   - Phase 2 개선사항 포함 (ATR 기반 노이즈 필터, 동적 K값, 슬리피지)
   - VanillaVBO의 향상된 버전

2. `src/utils/indicators_v2.py`
   - ImprovedNoiseIndicator: ATR 기반 동적 필터링
   - AdaptiveKValue: 동적 K-값 조정
   - apply_improved_indicators 함수

3. `src/backtester/slippage_model_v2.py`
   - DynamicSlippageModel: 시장 조건 반영 슬리피지
   - MarketCondition 분류
   - UpbitSlippageEstimator

### 의존 스크립트 (8개)
1. `scripts/debug_bootstrap.py`
2. `scripts/real_time_monitor.py`
3. `scripts/run_phase1_real_data.py`
4. `scripts/run_phase1_revalidation.py`
5. `scripts/run_phase2_integration.py`
6. `scripts/run_phase3_statistical_reliability.py`
7. `scripts/test_bootstrap_stability.py`
8. `scripts/test_sl_tp.py`

## 문제점

1. **코드 중복**: v1과 v2 간 중복 로직
2. **유지보수 부담**: 동일한 버그를 두 곳에서 수정
3. **Git 활용 미흡**: 버전 관리는 Git으로 해야 하는데 파일명으로 구분
4. **혼란 가능성**: 개발자가 어느 버전을 사용해야 할지 불명확

## 통합 전략

### 옵션 1: v1에 v2 기능 병합 (권장)

v2의 개선사항을 선택적 플래그로 v1에 통합:

```python
# src/strategies/volatility_breakout/vbo.py
class VanillaVBO(Strategy):
    def __init__(
        self,
        # 기존 파라미터...
        use_improved_noise: bool = False,  # Phase 2.1
        use_adaptive_k: bool = False,
        use_dynamic_slippage: bool = False,  # Phase 2.2
        use_cost_calculator: bool = False,  # Phase 2.3
        # ...
    ):
        # v2 기능을 조건부로 활성화
        if use_improved_noise:
            # ImprovedNoiseIndicator 사용
        else:
            # 기존 NoiseCondition 사용
```

**장점**:
- 단일 파일 유지
- 하위 호환성 보장 (기본값 False)
- v2 기능을 점진적으로 테스트 가능

**단점**:
- 코드 복잡도 증가
- 많은 조건문 필요

### 옵션 2: v2를 메인으로 승격, v1 deprecated

v2를 공식 버전으로 지정하고 v1을 제거:

```python
# vbo_v2.py → vbo.py로 리네임
# vbo.py → vbo_legacy.py 또는 삭제
```

**장점**:
- 명확한 버전 정책
- v2가 검증되었으므로 안전

**단점**:
- 스크립트 대량 업데이트 필요
- v1 의존 코드 깨질 수 있음

### 옵션 3: 점진적 마이그레이션 (실용적)

**Phase 1: Deprecation 마킹**
```python
# vbo_v2.py 상단
import warnings

warnings.warn(
    "vbo_v2 is deprecated and will be removed in v2.0.0. "
    "Use VanillaVBO with use_improved_noise=True instead.",
    DeprecationWarning,
    stacklevel=2
)
```

**Phase 2: 스크립트 업데이트**
- 각 스크립트를 v1 + 플래그로 마이그레이션
- 테스트 통과 확인

**Phase 3: v2 파일 제거**
- 다음 메이저 버전 릴리스 시 제거

## 실행 계획

### Step 1: v1에 v2 기능 통합 (2주)

1. **indicators.py 확장** (3일)
   ```python
   # src/utils/indicators.py에 추가
   def calculate_improved_noise(df, atr_period=14):
       # indicators_v2.py의 로직 이동
       pass
   
   def calculate_adaptive_k(df, ...):
       # indicators_v2.py의 로직 이동
       pass
   ```

2. **vbo.py 확장** (5일)
   - use_improved_noise, use_adaptive_k 파라미터 추가
   - calculate_indicators 메서드 조건부 로직
   - 기존 테스트 통과 확인
   - 새로운 플래그 조합 테스트

3. **문서화** (2일)
   - 각 플래그의 효과 문서화
   - 마이그레이션 가이드 작성

### Step 2: 스크립트 마이그레이션 (1주)

각 스크립트를 순차적으로 업데이트:

```python
# Before
from src.strategies.volatility_breakout.vbo_v2 import VanillaVBO_v2
strategy = VanillaVBO_v2(...)

# After
from src.strategies.volatility_breakout.vbo import VanillaVBO
strategy = VanillaVBO(
    use_improved_noise=True,
    use_adaptive_k=True,
    use_dynamic_slippage=True,
    use_cost_calculator=True,
    ...
)
```

### Step 3: v2 파일 Deprecation (즉시)

각 v2 파일 상단에 경고 추가:

```python
"""
DEPRECATED: This module will be removed in v2.0.0.

Use the main module with feature flags instead:
- vbo_v2.VanillaVBO_v2 → vbo.VanillaVBO(use_improved_noise=True, ...)
- indicators_v2 → indicators with improved functions
- slippage_model_v2 → slippage_model with DynamicSlippageModel
"""
import warnings
warnings.warn("...", DeprecationWarning, stacklevel=2)
```

### Step 4: 제거 (v2.0.0 릴리스 시)

- v2 파일 삭제
- 모든 테스트 통과 확인
- 문서 업데이트

## 리스크 및 완화 방안

### 리스크 1: 기존 스크립트 깨짐
**완화**: 
- Deprecation 기간 설정 (3-6개월)
- 마이그레이션 가이드 제공
- CI에서 deprecation warning 모니터링

### 리스크 2: 성능 저하
**완화**:
- 플래그가 False일 때 기존 로직 그대로 유지
- 벤치마크 테스트 추가
- 프로파일링으로 핫스팟 확인

### 리스크 3: 버그 유입
**완화**:
- 단계별 테스트
- v2 기존 테스트 모두 포팅
- 프로덕션 배포 전 충분한 검증

## 타임라인

```
Week 1-2: v1 확장 + 테스트
Week 3: 스크립트 마이그레이션 (1-4)
Week 4: 스크립트 마이그레이션 (5-8) + 문서화
Week 5: 통합 테스트 + 성능 검증
Week 6: Deprecation 마킹 + PR 리뷰
Week 7+: Deprecation 기간 (3-6개월)
v2.0.0: v2 파일 제거
```

## 현재 상태

- ⏸️ 보류 중
- ✅ 분석 완료
- 📋 실행 대기

**다음 액션**: 팀과 논의 후 옵션 선택 및 일정 확정

## 참고

- Phase 2 개선사항은 검증 완료 (Phase 1-3 completion reports 참조)
- v2 기능은 프로덕션 준비 완료
- 모든 테스트 통과 (948 tests, 86.99% coverage)

---

**작성일**: 2025-01-08  
**작성자**: Code Quality Review  
**상태**: RFC (Request for Comments)
