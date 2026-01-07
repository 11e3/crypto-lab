# 예제

이 디렉토리에는 Upbit Quant System 사용 방법을 보여주는 실용적인 예제가 포함되어 있습니다.

## 📚 사용 가능한 예제

### 1. 기본 백테스트 (`basic_backtest.py`)
기본 설정을 사용한 간단한 백테스트 예제입니다.

**보여주는 내용**:
- 기본 백테스트 실행
- 결과 보기
- 주요 지표 이해

**실행 방법**:
```bash
uv run python examples/basic_backtest.py
```

### 2. 커스텀 전략 (`custom_strategy.py`)
커스텀 거래 전략을 생성하고 테스트합니다.

**보여주는 내용**:
- 전략 커스터마이징
- 커스텀 조건 추가
- 파라미터 튜닝

**실행 방법**:
```bash
uv run python examples/custom_strategy.py
```

### 3. 전략 벤치마크 (`strategy_benchmark.py`) ⭐ NEW
4개 전략 패밀리를 포괄적으로 비교합니다.

**보여주는 내용**:
- 변동성 돌파(VBO) vs 모멘텀 vs 평균회귀 vs 페어트레이딩 비교
- 위험조정수익률(Sharpe, Sortino, Calmar) 분석
- 거래 통계 및 최대낙폭 비교
- 각 전략의 강점과 약점 파악

**실행 방법**:
```bash
python examples/strategy_benchmark.py
```

### 4. 라이브 트레이딩 시뮬레이터 (`live_trading_simulator.py`) ⭐ NEW
실제 자본 없이 페이퍼 모드에서 트레이딩을 시뮬레이션합니다.

**보여주는 내용**:
- 비동기 거래 실행 엔진
- 실시간 가격 시뮬레이션
- 포지션 관리 및 P&L 추적
- 전략 신호 생성 및 실행
- 포트폴리오 성과 분석

**실행 방법**:
```bash
python examples/live_trading_simulator.py
```

**샘플 결과**:
- 초기 자본: 1,000,000
- 최종 자산: 1,254,891
- 수익률: +25.5%
- 거래 수: 4,786건
- 승률: 35.4%

### 5. 실시간 거래 (`live_trading.py`)
실시간 거래 봇을 설정하고 실행합니다 (⚠️ 실제 자본 사용).

**보여주는 내용**:
- 실시간 거래 설정
- 리스크 관리
- 모니터링 설정

**⚠️ 경고**: 이 예제는 실제 돈을 사용합니다. 사용 전에 충분히 테스트하세요.

**실행 방법**:
```bash
python examples/live_trading.py
```

### 5. 성능 분석 (`performance_analysis.py`)
전략 성능을 분석하고 비교합니다.

**보여주는 내용**:
- 성능 지표 계산
- 리스크 분석
- 전략 비교

**실행 방법**:
```bash
python examples/performance_analysis.py
```

### 6. 전략 비교 (`strategy_comparison.py`)
여러 전략을 나란히 비교합니다.

**보여주는 내용**:
- 여러 전략 실행
- 성능 비교
- 리스크-수익 분석

**실행 방법**:
```bash
python examples/strategy_comparison.py
```

### 7. 성능 벤치마크 (`performance_benchmark.py`)
다양한 설정에서 성능을 측정합니다.

**보여주는 내용**:
- 매개변수에 따른 성능 변화
- 최적 설정 찾기
- 일관성 및 견고성 테스트

**실행 방법**:
```bash
python examples/performance_benchmark.py
```

### 8. 포트폴리오 최적화 (`portfolio_optimization.py`) ⭐ NEW
포트폴리오 구성 및 최적화 방법을 보여줍니다.

**보여주는 내용**:
- 평균-분산 최적화(MPT) - Sharpe 비율 최대화
- 리스크 패리티 - 동일 리스크 기여도
- 켈리 기준 - 거래 통계 기반 위치 크기 결정
- 거래비용 모델링 (선형+2차)
- 제약 조건 적용 (최대 위치, 최소/최대 할당)
- 리밸런싱 정책 (달력 기반, 한계값 기반, 동적)

**실행 방법**:
```bash
python examples/portfolio_optimization.py
```

**출력 예시**:
```
MPT (MODERN PORTFOLIO THEORY)
Expected Annual Return: 8.45%
Portfolio Volatility: 12.34%
Sharpe Ratio: 0.68

ALLOCATION
  BTC      : 52.30%
  ETH      : 35.20%
  STAKING  : 12.50%
```

## 🚀 빠른 시작

1. **의존성 설치**:
   ```bash
   pip install -e ".[dev]"
   ```

2. **환경 설정**:
   ```bash
   cp .env.example .env
   # .env 파일을 편집하여 설정 입력
   ```

3. **예제 실행**:
   ```bash
   python examples/basic_backtest.py
   ```

## 📖 학습 경로

**초급**:
1. `basic_backtest.py`로 시작
2. 결과 이해
3. `custom_strategy.py` 시도

**중급**:
1. `performance_analysis.py` 탐색
2. `strategy_benchmark.py`로 전략 비교
3. `strategy_comparison.py`로 추가 분석
4. 전략을 더욱 커스터마이징
5. `performance_benchmark.py`로 파라미터 최적화

**고급**:
1. `live_trading_simulator.py`로 페이퍼 모드 트레이딩 시뮬레이션 ⭐ NEW
   - 비동기 거래 엔진
   - 실시간 가격 시뮬레이션
   - 포지션 관리 및 P&L 추적
   - 전략 검증 및 성과 분석
2. `portfolio_optimization.py` 탐색
   - MPT, 리스크 패리티, 켈리 기준 비교
   - 거래비용 모델링
   - 제약 조건 및 리밸런싱
3. `live_trading.py` 설정 (주의! 실제 자본 사용)
4. 자신만의 포트폴리오 전략 생성

## 💡 팁

- **간단하게 시작**: 복잡한 예제로 넘어가기 전에 기본 예제부터 시작
- **코드 읽기**: 예제는 잘 주석 처리되어 있습니다 - 학습을 위해 읽어보세요
- **실험**: 예제를 수정하여 변경사항이 결과에 미치는 영향을 확인
- **먼저 테스트**: 실거래 전에 항상 백테스팅 수행

## 🐛 문제 해결

**Import 오류**: `pip install -e ".[dev]"`로 의존성을 설치했는지 확인

**데이터 오류**: 시장 데이터를 수집했는지 확인:
```bash
python scripts/generate_sample_data.py
```

**설정 오류**: `.env` 파일과 `config/settings.yaml` 확인

## 📚 관련 문서

- [시작 가이드](../docs/guides/getting_started.md)
- [전략 가이드 및 라이브러리](../docs/guides/strategy_guide.md) ⭐ NEW
- [전략 커스터마이징](../docs/guides/strategy_customization.md)
- [설정 가이드](../docs/guides/configuration.md)
- [아키텍처 문서](../docs/architecture.md)

## 🤝 기여하기

버그를 발견했거나 제안사항이 있으신가요? 이슈를 열거나 PR을 제출해 주세요!
