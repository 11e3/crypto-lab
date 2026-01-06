# Sphinx 문서 생성 가이드

이 문서는 Sphinx를 사용한 자동 API 문서 생성 방법을 설명합니다.

## 사전 요구사항

```bash
# 문서 생성에 필요한 의존성 설치
uv sync --extra docs
```

## 문서 빌드

### 기본 빌드

**Windows (PowerShell):**
```powershell
# 프로젝트 루트에서
cd docs
.\build.ps1

# 또는 직접 실행
cd docs
uv run sphinx-build -b html . _build/html
```

**Linux/Mac:**
```bash
# 프로젝트 루트에서
make docs

# 또는 docs 디렉토리에서 직접
cd docs
uv run sphinx-build -b html . _build/html
```

### 로컬 서버로 확인

**Windows (PowerShell):**
```powershell
# 문서 빌드 및 로컬 서버 실행
cd docs
.\serve.ps1

# 브라우저에서 http://localhost:8000 열기
```

**Linux/Mac:**
```bash
# 문서 빌드 및 로컬 서버 실행
make docs-serve

# 브라우저에서 http://localhost:8000 열기
```

## 문서 구조

### 소스 파일

- `conf.py`: Sphinx 설정 파일
- `index.rst`: 메인 인덱스 파일
- `api/index.rst`: API 문서 인덱스
- `api/*.rst`: 각 레이어별 API 문서

### 생성되는 파일

빌드 후 `docs/_build/html/` 디렉토리에 HTML 문서가 생성됩니다.

## 문서 업데이트

### 새 모듈 추가

1. `docs/api/index.rst`에 새 모듈 섹션 추가:

```rst
새 모듈
~~~~~~~

.. automodule:: src.new_module
   :members:
   :undoc-members:
   :show-inheritance:
```

2. 문서 재빌드:

```bash
make docs
```

### docstring 업데이트

소스 코드의 docstring을 업데이트하면 다음 빌드 시 자동으로 반영됩니다.

## 설정 커스터마이징

`docs/conf.py` 파일에서 다음을 커스터마이징할 수 있습니다:

- 프로젝트 정보
- 테마 및 스타일
- 확장 기능
- autodoc 옵션

## 문제 해결

### Import 오류

모듈을 import할 수 없는 경우:

1. `conf.py`의 `sys.path` 확인
2. 필요한 의존성이 설치되어 있는지 확인

### 문서가 생성되지 않음

1. `conf.py`의 설정 확인
2. 소스 파일 경로 확인
3. 빌드 로그 확인

## CI/CD 통합

GitHub Actions에서 문서를 자동으로 빌드하고 배포할 수 있습니다:

```yaml
- name: Build documentation
  run: |
    uv sync --extra docs
    make docs
```

## 참고 자료

- [Sphinx 공식 문서](https://www.sphinx-doc.org/)
- [autodoc 확장](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [Napoleon 확장](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
