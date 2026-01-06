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

### 3. 실시간 거래 (`live_trading.py`)
실시간 거래 봇을 설정하고 실행합니다.

**보여주는 내용**:
- 실시간 거래 설정
- 리스크 관리
- 모니터링 설정

**⚠️ 경고**: 이 예제는 실제 돈을 사용합니다. 사용 전에 충분히 테스트하세요.

**실행 방법**:
```bash
uv run python examples/live_trading.py
```

### 4. 성능 분석 (`performance_analysis.py`)
전략 성능을 분석하고 비교합니다.

**보여주는 내용**:
- 성능 지표 계산
- 리스크 분석
- 전략 비교

**실행 방법**:
```bash
uv run python examples/performance_analysis.py
```

### 5. 전략 비교 (`strategy_comparison.py`)
여러 전략을 나란히 비교합니다.

**보여주는 내용**:
- 여러 전략 실행
- 성능 비교
- 리스크-수익 분석

**실행 방법**:
```bash
uv run python examples/strategy_comparison.py
```

## 🚀 빠른 시작

1. **의존성 설치**:
   ```bash
   uv sync --extra dev
   ```

2. **환경 설정**:
   ```bash
   cp .env.example .env
   # .env 파일을 편집하여 설정 입력
   ```

3. **예제 실행**:
   ```bash
   uv run python examples/basic_backtest.py
   ```

## 📖 학습 경로

**초급**:
1. `basic_backtest.py`로 시작
2. 결과 이해
3. `custom_strategy.py` 시도

**중급**:
1. `performance_analysis.py` 탐색
2. `strategy_comparison.py`로 전략 비교
3. 전략을 더욱 커스터마이징

**고급**:
1. `live_trading.py` 설정 (주의!)
2. 자신만의 전략 생성
3. 파라미터 최적화

## 💡 팁

- **간단하게 시작**: 복잡한 예제로 넘어가기 전에 기본 예제부터 시작
- **코드 읽기**: 예제는 잘 주석 처리되어 있습니다 - 학습을 위해 읽어보세요
- **실험**: 예제를 수정하여 변경사항이 결과에 미치는 영향을 확인
- **먼저 테스트**: 실거래 전에 항상 백테스팅 수행

## 🐛 문제 해결

**Import 오류**: `uv sync`로 의존성을 설치했는지 확인

**데이터 오류**: 시장 데이터를 수집했는지 확인:
```bash
uv run upbit-quant collect
```

**설정 오류**: `.env` 파일과 `config/settings.yaml` 확인

## 📚 관련 문서

- [시작 가이드](../docs/guides/getting_started.md)
- [전략 커스터마이징](../docs/guides/strategy_customization.md)
- [설정 가이드](../docs/guides/configuration.md)
- [아키텍처 문서](../docs/architecture.md)

## 🤝 기여하기

버그를 발견했거나 제안사항이 있으신가요? 이슈를 열거나 PR을 제출해 주세요!
