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
   make check  # 모든 검사 실행 (포맷, 린트, 타입 체크, 테스트)
   ```

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

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) 패키지 관리자

### 설정

```bash
# 포크 클론
git clone https://github.com/your-username/crypto-quant-system.git
cd crypto-quant-system

# 의존성 설치
uv sync --extra dev

# pre-commit 훅 설치
uv run pre-commit install
```

## 코딩 표준

### Python 스타일

- [PEP 8](https://pep8.org/) 준수
- 모든 함수에 타입 힌트 사용
- 독스트링 작성 (Google 스타일)
- 최대 줄 길이: 100자

### 코드 품질 도구

```bash
# 코드 포맷팅
make format

# 린팅
make lint

# 타입 체크
make type-check

# 테스트 실행
make test

# 모든 검사 실행
make check
```

### 테스트

- 새 기능에 대한 테스트 작성
- 테스트 커버리지 유지 또는 개선
- 테스트 데이터에 pytest 픽스처 사용
- 명명 규칙: `test_*.py` 파일, `test_*` 함수

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
