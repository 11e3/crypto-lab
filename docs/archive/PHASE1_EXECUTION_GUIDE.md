# Phase 1 실행 가이드 - 과적합 검증

**상태**: ✅ 준비 완료  
**실행 명령**: `python scripts/run_phase1_overfitting_detection.py`

---

## 📋 생성된 파일 목록

### 1. 핵심 구현 파일 (3개)

#### [src/backtester/walk_forward_auto.py](src/backtester/walk_forward_auto.py)
- **목적**: Walk-Forward Analysis (WFA) 자동 실행
- **주요 클래스**:
  - `WalkForwardAnalyzer`: WFA 엔진
  - `WFASegment`: 각 구간 (Training + Test)
  - `WFAReport`: 종합 리포트
- **기능**:
  - Training/Test 구간 자동 분할
  - 각 구간에서 파라미터 최적화
  - OOS/IS 비율 계산
  - HTML 리포트 생성
- **검증 기준**:
  - OOS/IS > 0.3 → 정상
  - OOS/IS 0.1-0.3 → 경고
  - OOS/IS < 0.1 → 위험

#### [src/backtester/robustness_analysis.py](src/backtester/robustness_analysis.py)
- **목적**: 파라미터 안정성 분석
- **주요 클래스**:
  - `RobustnessAnalyzer`: 안정성 분석 엔진
  - `RobustnessResult`: 개별 파라미터 조합 결과
  - `RobustnessReport`: 종합 리포트
- **기능**:
  - 최적 파라미터 주변 성과 분포 분석
  - 파라미터별 민감도 계산 (0.0~1.0)
  - 이웃 성공률 계산
  - HTML + CSV 리포트 생성
- **검증 기준**:
  - Neighbor Success Rate > 70% → 강건함
  - Success Rate 50-70% → 보통
  - Success Rate < 50% → 취약함

#### [tests/unit/test_overfitting_detection.py](tests/unit/test_overfitting_detection.py)
- **목적**: Permutation Test (통계적 과적합 검증)
- **주요 클래스**:
  - `PermutationTester`: Permutation test 엔진
  - `PermutationTestResult`: 검정 결과
- **기능**:
  - 원본 데이터 vs 섞인 데이터 비교
  - Z-score, p-value 계산
  - 통계적 유의성 판정
  - 히스토그램 포함 HTML 리포트 생성
- **검증 기준**:
  - Z-score > 2.0 → 통계적으로 유의 (p < 0.05)
  - Z-score 1.0-2.0 → 약하게 유의
  - Z-score < 1.0 → 유의하지 않음

### 2. 실행 스크립트

#### [scripts/run_phase1_overfitting_detection.py](scripts/run_phase1_overfitting_detection.py)
- **실행 방식**: `python scripts/run_phase1_overfitting_detection.py`
- **실행 순서**:
  1. WFA 실행 → `reports/phase1/01_wfa_report.html`
  2. Robustness 분석 → `reports/phase1/02_robustness_report.html`
  3. Permutation test → `reports/phase1/03_permutation_test_report.html`
  4. 종합 리포트 생성 → `reports/phase1/00_phase1_summary.md`
- **실행 시간**: ~30분 (100 WFA segments × 25 robustness combos × 100 shuffles)

---

## 🚀 실행 방법

### 1단계: 환경 확인
```bash
cd c:\workspace\dev\crypto-quant-system

# Python 버전 확인 (3.11+)
python --version

# 필수 패키지 확인
python -c "import pandas, numpy, scipy; print('OK')"
```

### 2단계: 데이터 준비
```bash
# 옵션 1: 실제 데이터 사용
# → data/processed/KRW-BTC.parquet 파일 필수
# → 없으면 자동으로 샘플 데이터 생성

# 옵션 2: 스크립트 자동 생성
# → 8년 치 샘플 데이터로 테스트
```

### 3단계: Phase 1 실행
```bash
# 기본 실행
python scripts/run_phase1_overfitting_detection.py

# 로깅 레벨 변경 (선택사항)
# 스크립트 내 logger.setLevel() 수정 필요
```

### 4단계: 결과 확인
```bash
# 생성된 리포트 확인
ls reports/phase1/

# 브라우저에서 열기
# - reports/phase1/01_wfa_report.html
# - reports/phase1/02_robustness_report.html
# - reports/phase1/03_permutation_test_report.html
# - reports/phase1/00_phase1_summary.md (마크다운)
```

---

## 📊 예상 결과

### Walk-Forward Analysis
```
세그먼트 1 (2018-01 ~ 2020-01):
  IS Return: 250%, OOS Return: 180%, Ratio: 72% ✅ (정상)

세그먼트 2 (2019-01 ~ 2021-01):
  IS Return: 400%, OOS Return: 60%, Ratio: 15% ⚠️ (경고)

세그먼트 3 (2020-01 ~ 2022-01):
  IS Return: -20%, OOS Return: -25%, Ratio: 125% ✅ (정상)

종합:
  - In-Sample Avg: 150%
  - Out-of-Sample Avg: 100%
  - Overfitting Ratio: 67% ✅ (정상)
```

### Robustness Analysis
```
최적 파라미터: sma_period=4, noise_period=8

테스트 범위:
  sma_period: [2, 3, 4, 5, 6]
  noise_period: [6, 7, 8, 9, 10]
  → 25개 조합 테스트

결과:
  - Mean Return: 120% (±30%)
  - Neighbor Success Rate: 85% ✅ (강건함)
  
파라미터별 민감도:
  - sma_period: 0.45 (중간)
  - noise_period: 0.35 (낮음) ✅
```

### Permutation Test
```
원본 데이터: Return = 120%, Sharpe = 1.5
100회 셔플 데이터: Mean Return = 5%, Std = 15%

Z-score = (120% - 5%) / 15% = 7.67
P-value = 0.0001 (매우 유의)

✅ 결론: 전략의 성과는 통계적으로 유의함
```

---

## ⚠️ 주의사항

### 1. 데이터 품질
- **필수**: OHLCV 데이터 (Open, High, Low, Close, Volume)
- **기간**: 최소 3년 (252×3 거래일)
- **빈도**: 일일 봉 (Daily OHLC)

### 2. 파라미터 범위
- 현재 설정 (스크립트에서):
  ```python
  'sma_period': [3, 4, 5]           # 최적값 ±1
  'trend_sma_period': [7, 8, 9]     # 최적값 ±1
  'short_noise_period': [3, 4, 5]   # 최적값 ±1
  'long_noise_period': [7, 8, 9]    # 최적값 ±1
  ```
- **너무 넓으면** → 계산 시간 증가 (exponential)
- **너무 좁으면** → 안정성 검증 불충분

### 3. 계산 시간
- WFA: O(segments × param_combos)
- Robustness: O(param_combos)
- Permutation: O(shuffles)

| 항목 | 설정 | 예상 시간 |
|------|------|---------|
| WFA Segments | 10 | 10분 |
| Robustness Combos | 25 | 5분 |
| Permutation Shuffles | 100 | 5분 |
| **총계** | | **20분** |

실제 데이터: 더 오래 걸릴 수 있음

### 4. 메모리
- 전체 메모리 사용: ~500MB
- 제약 있으면 WFA segments 감소

---

## 🔧 커스터마이징

### 파라미터 범위 변경
```python
# scripts/run_phase1_overfitting_detection.py 수정

# 현재:
param_ranges={
    'sma_period': [3, 4, 5],
    ...
}

# 변경:
param_ranges={
    'sma_period': [2, 3, 4, 5, 6],  # ±2 범위로 확대
    ...
}
```

### WFA 구간 설정 변경
```python
# Walk-Forward 기간 조정
analyzer = WalkForwardAnalyzer(
    data=data,
    train_period=252 * 2,  # 2년 → 3년 가능
    test_period=252,       # 1년 → 6개월 가능
    step=63                # 3개월 → 월간 가능
)
```

### Permutation Test 횟수
```python
# 충분한 통계적 신뢰도를 위해 1000 권장
result = tester.run(
    num_shuffles=100,  # 현재 (테스트용)
    # num_shuffles=1000,  # 실제 운영
)
```

---

## 📈 해석 가이드

### OOS/IS 비율 해석
```
0.5 이상   → ✅ 정상 (과적합 거의 없음)
0.3-0.5    → ✅ 정상 (약간 보수적)
0.1-0.3    → ⚠️ 경고 (중간 과적합)
< 0.1      → ❌ 위험 (심각한 과적합)
음수       → ❌ 재난 (OOS에서 손실)
```

### Neighbor Success Rate 해석
```
> 80%      → ✅ 매우 강건함 (안정적)
70-80%     → ✅ 강건함
50-70%     → ⚠️ 보통 (파라미터 민감함)
< 50%      → ❌ 취약함 (불안정)
```

### Z-score 해석
```
> 3.0      → 🎯 매우 강함 (신호 확실)
2.0-3.0    → ✅ 유의함 (p < 0.05)
1.0-2.0    → ⚠️ 약하게 유의 (우려 있음)
< 1.0      → ❌ 유의하지 않음 (우연)
```

---

## ✅ 검증 체크리스트

실행 후 확인할 항목:

- [ ] 모든 3개 리포트 생성됨 (01, 02, 03)
- [ ] 종합 리포트 생성됨 (00_phase1_summary.md)
- [ ] OOS/IS 비율 > 0.3
- [ ] Neighbor Success Rate > 70%
- [ ] Z-score > 2.0 (p < 0.05)
- [ ] 세 가지 지표가 모두 통과 → Phase 2 진행 가능
- [ ] 하나라도 실패 → 전략/파라미터 재검토 필요

---

## 🎯 다음 단계

### Phase 1 완료 후
1. ✅ 세 가지 리포트 상세 분석
2. ✅ 과적합 여부 최종 판정
3. **Phase 2 진행**: 노이즈 비율 및 슬리피지 안정화

### 실패한 경우
1. ❌ 파라미터 재조정 (더 넓은 범위)
2. ❌ 데이터 기간 변경
3. ❌ 전략 로직 재검토

---

**생성일**: 2026년 1월 7일  
**담당자**: Crypto Quant System Team  
**상태**: 🟢 Ready to Execute
