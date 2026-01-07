# Crypto Quant System에 기여하기

기여에 관심을 가져주셔서 감사합니다! 이 문서는 이 프로젝트에 기여하기 위한 가이드라인과 지침을 제공합니다.

## 행동 강령

- 존중하고 포용적이 되기
- 신규 참여자를 환영하고 학습을 돕기
- 건설적인 피드백에 집중하기
- 다양한 관점과 경험을 존중하기

## 기여 방법

### 버그 신고

1. [Issues](https://github.com/your-username/crypto-quant-system/issues)에서 버그가 이미 신고되었는지 확인
2. 그렇지 않은 경우, 다음을 포함하여 새 이슈 생성:
   - 명확한 제목과 설명
   - 재현 단계
   - 예상 동작 vs 실제 동작
   - 환경 세부 정보 (OS, Python 버전 등)
   - 오류 메시지 또는 로그 (해당되는 경우)

### 기능 제안

1. 기존 이슈 및 토론 확인
2. 다음을 포함하여 새 이슈 생성:
   - 기능에 대한 명확한 설명
   - 사용 사례 및 동기
   - 제안된 구현 (아이디어가 있는 경우)

### Pull Request

1. **저장소 포크**
2. **기능 브랜치 생성**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **변경사항 작성**
   - 코딩 표준 준수 (아래 참조)
   - 테스트 작성/업데이트
   - 문서 업데이트

4. **품질 검사 실행**
   
   ```bash
   # 방법 1: 모든 nox 세션 실행 (권장)
   nox
   
   # 또는 개별 세션 실행
   nox -s format    # 코드 포맷팅
   nox -s lint      # 린팅 & 타입 검사
   nox -s tests     # 테스트 실행
   nox -s docs      # 문서 빌드
   ```
   
   **PR 제출 전 체크리스트:**
   - [ ] `nox`가 모든 세션에서 통과됨
   - [ ] 신규 코드에 대한 테스트 추가됨
   - [ ] 테스트 커버리지 80% 이상 유지
   - [ ] 문서 업데이트됨 (해당되는 경우)
   - [ ] 타입 힌트 추가됨
   - [ ] Docstring 추가됨 (Google 스타일)
   - [ ] 변경사항이 관련 이슈를 해결함

5. **변경사항 커밋**
   ```bash
   git commit -m "feat: add new feature"
   ```
   Conventional commit 메시지 사용:
   - `feat:` 새로운 기능
   - `fix:` 버그 수정
   - `docs:` 문서 변경
   - `test:` 테스트 추가/변경
   - `refactor:` 코드 리팩토링
   - `style:` 코드 스타일 변경 (포맷팅 등)

6. **포크에 푸시**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Pull Request 생성**
   - 명확한 설명 제공
   - 관련 이슈 참조
   - 리뷰 및 피드백 대기

## 개발 환경 설정

### 사전 요구사항

- Python 3.14+
- Git

### 초기 설정

```bash
# 포크 클론
git clone https://github.com/your-username/crypto-quant-system.git
cd crypto-quant-system

# 가상 환경 생성 (선택사항, 권장)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix/macOS
source .venv/bin/activate

# 의존성 설치 (개발 종속성 포함)
pip install -e ".[dev,test,docs]"

# Pre-commit 훅 설정 (커밋 전 자동 검사)
pre-commit install
```

### 설정 검증

```bash
# Pre-commit 훅이 모든 파일에 적용되는지 확인
pre-commit run --all-files
```

## 코딩 표준

### Python 스타일

- [PEP 8](https://pep8.org/) 준수
- 모든 함수에 타입 힌트 사용
- 독스트링 작성 (Google 스타일)
- 최대 줄 길이: 100자

### 개발 워크플로우 (nox 사용)

모든 개발 작업은 `nox`를 통해 자동화됩니다. `nox`는 개발 환경을 격리하고 일관된 도구 버전을 관리합니다.

**주요 nox 세션:**

```bash
# 1. 코드 포맷팅 (Black, isort, docformatter)
nox -s format

# 2. 린팅 및 타입 검사 (Ruff, mypy)
nox -s lint

# 3. 엄격한 타입 검사 (mypy --strict)
nox -s type_check

# 4. 테스트 실행 (pytest with coverage)
nox -s tests

# 5. 문서 빌드 (Sphinx)
nox -s docs

# 6. Pre-commit 검사 (모든 hooks 실행)
nox -s pre_commit_check

# 7. 캐시 정리
nox -s clean

# 모든 세션 실행 (권장: PR 전에)
nox
```

**세션별 설명:**

| 세션 | 목적 | 도구 |
|------|------|------|
| `format` | 코드 자동 포맷팅 | Ruff, isort, docformatter |
| `lint` | 린팅 & 타입 검사 | Ruff, mypy |
| `type_check` | 엄격한 타입 검사 | mypy (--strict) |
| `tests` | 테스트 실행 & 커버리지 | pytest |
| `docs` | 문서 생성 | Sphinx |
| `pre_commit_check` | Pre-commit 훅 검증 | pre-commit |
| `clean` | 빌드 아티팩트 정리 | shutil |

### Pre-commit 훅

커밋할 때 자동으로 실행되는 검사:

```yaml
# .pre-commit-config.yaml 참고
- Ruff: 코드 스타일 및 포맷팅
- isort: import 정렬
- Black: 코드 포맷팅 
- mypy: 타입 검사
- docformatter: docstring 포맷팅
- bandit: 보안 검사
- 그 외 10+ 핸들러
```

**Pre-commit 우회 (권장하지 않음):**

```bash
git commit --no-verify
```

### 테스트

테스트는 코드 품질을 보장하는 핵심입니다.

```bash
# 전체 테스트 실행
nox -s tests

# 특정 테스트 파일만 실행
pytest tests/test_strategy.py -v

# 특정 테스트 함수만 실행
pytest tests/test_strategy.py::test_volatility_breakout -v

# 커버리지 리포트 생성
pytest --cov=src --cov-report=html
# htmlcov/index.html에서 리포트 확인
```

**테스트 작성 가이드:**

- 새 기능에 대한 테스트 작성
- 테스트 커버리지 유지 또는 개선 (목표: 80%+)
- 테스트 데이터에 pytest 픽스처 사용
- 명명 규칙: `test_*.py` 파일, `test_*` 함수
- Arrange-Act-Assert 패턴 사용

**예시:**

```python
def test_volatility_breakout_calculate_signals(sample_data):
    """Test signal generation in volatility breakout strategy."""
    # Arrange
    strategy = VolatilityBreakout(volatility_threshold=0.02)
    
    # Act
    signals = strategy.calculate_signals(sample_data)
    
    # Assert
    assert len(signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in signals)
```

### 문서

- 필요시 README.md 업데이트
- 새 함수/클래스에 독스트링 추가
- 관련 문서 파일 업데이트
- 주석을 명확하고 간결하게 유지

## 프로젝트 구조

```
crypto-quant-system/
├── src/              # 소스 코드
├── tests/            # 테스트 파일
├── docs/             # 문서
├── scripts/          # 유틸리티 스크립트
└── deploy/           # 배포 파일
```

## 커밋 메시지 가이드라인

[Conventional Commits](https://www.conventionalcommits.org/) 사용:

```
<type>(<scope>): <subject>

<body>

<footer>
```

예시:
- `feat(strategy): add momentum filter condition`
- `fix(engine): correct equity calculation bug`
- `docs(readme): update installation instructions`
- `test(cache): add cache invalidation tests`

## 리뷰 프로세스

1. 모든 PR은 최소 1명의 리뷰가 필요합니다
2. 유지보수자는 2-3 영업일 내에 리뷰합니다
3. 리뷰 코멘트에 신속하게 대응합니다
4. PR을 집중적이고 합리적인 크기로 유지합니다

## 질문이 있으신가요?

- 질문에 대한 이슈 열기
- 기존 문서 확인
- 유사한 질문에 대한 닫힌 이슈/PR 검토

기여해 주셔서 감사합니다! 🎉
