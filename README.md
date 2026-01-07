# 🚀 Crypto Quant System

<div align="center">

<!-- Status Badges -->
![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<!-- CI/CD Badges -->
![CI](https://github.com/11e3/crypto-quant-system/actions/workflows/ci.yml/badge.svg)
![CodeQL](https://github.com/11e3/crypto-quant-system/actions/workflows/codeql.yml/badge.svg)
![Docs](https://github.com/11e3/crypto-quant-system/actions/workflows/docs.yml/badge.svg)

<!-- Quality Badges -->
![Coverage](https://codecov.io/gh/11e3/crypto-quant-system/branch/main/graph/badge.svg)
![Code Style](https://img.shields.io/badge/Code%20Style-Ruff%20%2B%20Black-black.svg)
![Type Check](https://img.shields.io/badge/Type%20Check-Mypy-blue.svg)

<!-- Project Badges -->
![Tests](https://img.shields.io/badge/Tests-908%20passed-green.svg)
![Coverage Threshold](https://img.shields.io/badge/Coverage%20Threshold-87.31%25-brightgreen.svg)

**변동성 돌파 전략을 사용한 암호화폐 자동 거래 시스템**

[기능](#-features) • [빠른 시작](#-quick-start) • [아키텍처](#-architecture) • [문서](#-documentation) • [기여하기](#-contributing)

</div>

---

## 📋 개요

Crypto Quant System은 여러 암호화폐 거래소(Upbit 등)를 지원하는 프로덕션 준비가 완료된 자동 거래 시스템입니다. 포괄적인 백테스팅 기능, 실시간 거래 실행, 그리고 광범위한 성능 분석을 갖춘 정교한 변동성 돌파(VBO) 전략을 구현합니다.

### 💼 요약

- 재현 가능한 데모: `examples/`와 `deploy/docker-compose.yml`로 5분 내 실행
- 신뢰성 가시화: 테스트/커버리지/타입 검사 CI, 보안 스캔(CodeQL)
- 설계 품질: 이벤트 버스·주문/포지션·전략 인터페이스 중심의 모듈 구조
- 문서 접근성: Sphinx 기반 API/가이드, GitHub Pages 배포
- 성과 제시: 테어시트/벤치마크와 거래비용 반영 결과 보고

### 🎯 주요 특징

- **고성능 백테스팅**: 빠른 과거 데이터 분석을 위한 pandas/numpy 기반 벡터화 엔진
- **모듈식 전략 시스템**: 유연한 전략 설계를 위한 구성 가능한 조건 및 필터
- **프로덕션 준비 완료**: 완전한 오류 처리, 로깅, 모니터링 및 Docker 배포
- **충분한 테스트**: 900개 이상의 테스트 케이스로 80% 이상의 테스트 커버리지
- **현대적 Python**: 타입 힌트, Pydantic 설정, SOLID 원칙, 클린 아키텍처

## ✨ 기능

### 핵심 기능

- 🔄 **변동성 돌파 전략**: 변동성 패턴 기반 자동 진입/청산
- 📊 **벡터화 백테스팅**: 빠른 과거 성능 분석
- 🤖 **실시간 거래 봇**: WebSocket 통합을 통한 실시간 실행
- 📈 **성능 분석**: 포괄적인 지표 (CAGR, Sharpe, MDD 등)
- 🎨 **시각적 리포트**: 자산 곡선, 낙폭 차트, 월별 히트맵

### 기술적 우수성

- 🏗️ **클린 아키텍처**: SOLID 원칙, 의존성 주입, 관심사 분리
- 🧪 **높은 테스트 커버리지**: 단위 및 통합 테스트로 80% 이상 커버리지
- 📝 **타입 안전성**: MyPy 검증을 통한 완전한 타입 힌트
- 🔒 **보안**: 환경 변수 기반 설정, 하드코딩된 비밀번호 없음
- 🐳 **Docker 지원**: GCP/AWS 배포를 위한 프로덕션 준비 컨테이너화
- 📚 **포괄적인 문서**: 아키텍처 가이드, API 문서, 기여 가이드라인

## 🛠️ 기술 스택

### 핵심 기술
- **Python 3.14+**: 타입 힌트를 포함한 현대적 Python
- **pandas/numpy**: 데이터 처리 및 벡터화 연산
- **pydantic**: 타입 안전 설정 관리
- **click**: CLI 프레임워크
- **pyupbit**: Upbit API 통합 (다른 거래소 지원 확장 가능)

### 개발 도구
- **uv**: 빠른 Python 패키지 관리자
- **Ruff**: 린터 및 포맷터 (Black 대체)
- **MyPy**: 완전한 타입 안전성을 위한 정적 타입 검사
- **pytest**: 80% 이상 커버리지의 테스트 프레임워크
- **pre-commit**: 코드 품질을 위한 Git 훅

### 인프라
- **Docker**: 컨테이너화
- **GCP**: 클라우드 배포 지원
- **WebSocket**: 실시간 시장 데이터

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/11e3/crypto-quant-system.git
cd crypto-quant-system

# 가상환경 생성 (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# 의존성 설치 (개발용 extras 포함)
pip install -U pip
pip install -e .[dev]
```

참고: `uv`를 사용하는 고속 설치를 선호한다면 `uv sync --extra dev`를 이용할 수 있습니다.

### 백테스팅

```bash
# 기본 설정으로 백테스트 실행
crypto-quant backtest

# 커스텀 백테스트
crypto-quant backtest \
    --tickers KRW-BTC KRW-ETH \
    --interval day \
    --strategy legacy \
    --initial-capital 1000000 \
    --max-slots 4
```

### 실시간 거래 (API 키 필요)

```bash
# 환경 변수 설정
export UPBIT_ACCESS_KEY="your-access-key"
export UPBIT_SECRET_KEY="your-secret-key"

# 거래 봇 실행
crypto-quant run-bot
```

## 📊 성능 결과

### 백테스트 결과 (기본 전략)
- **기간**: 3,018일 (8년 이상)
- **총 수익률**: 38,331.40%
- **CAGR**: 105.40%
- **최대 낙폭**: 24.97%
- **샤프 비율**: 1.97
- **칼마 비율**: 4.22
- **승률**: 36.03%
- **총 거래 횟수**: 705
- **수익 팩터**: 1.77

*참고: 과거 성과는 미래 결과를 보장하지 않습니다. 이 결과는 교육 목적으로만 제공됩니다.*

## 🏗️ 아키텍처

### 시스템 설계

```
┌─────────────────────────────────────────────────────────┐
│                    CLI 인터페이스                        │
│              (crypto-quant 명령어)                       │
└────────────────────┬──────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐            ┌────▼────┐
    │백테스트 │            │실시간 봇 │
    │ 엔진    │            │Facade    │
    └────┬────┘            └────┬─────┘
         │                      │
    ┌────▼──────────────────────▼─────┐
    │      전략 시스템                  │
    │  (VanillaVBO, Conditions, etc.)  │
    └────┬─────────────────────────────┘
         │
    ┌────▼────────────┐
    │  데이터 레이어   │
    │  (Cache, Source) │
    └─────────────────┘
```

### 주요 구성 요소

- **백테스팅 엔진**: 빠른 과거 데이터 분석을 위한 벡터화 계산
- **전략 시스템**: 유연한 전략 설계를 위한 모듈식 조건 및 필터
- **실행 레이어**: 주문 관리, 포지션 추적, 신호 처리
- **데이터 레이어**: 효율적인 캐싱, 데이터 수집, 지표 계산
- **설정**: 환경 변수 지원을 통한 타입 안전 설정

## 📁 프로젝트 구조

```
crypto-quant-system/
├── src/
│   ├── backtester/      # 백테스팅 엔진
│   ├── execution/       # 실시간 거래 봇
│   ├── strategies/      # 거래 전략
│   ├── data/            # 데이터 수집 및 캐싱
│   ├── exchange/        # 거래소 API 추상화
│   ├── config/          # 설정 관리
│   └── utils/           # 유틸리티
├── tests/               # 테스트 스위트 (80% 이상 커버리지)
├── docs/                # 포괄적인 문서
├── deploy/              # Docker 및 배포 설정
└── scripts/             # 유틸리티 스크립트
```

## 📚 문서

### 📖 가이드
- [시작 가이드](docs/guides/getting_started.md) - 설치 및 설정
- [전략 커스터마이징](docs/guides/strategy_customization.md) - 커스텀 전략 작성
- [설정 가이드](docs/guides/configuration.md) - 설정 가이드

### 🏗️ 아키텍처
- [시스템 아키텍처](docs/architecture.md) - 설계 원칙 및 구조
- [레거시 vs 신규 봇 비교](docs/comparison_legacy_vs_new_bot.md) - 마이그레이션 가이드

### 📋 개발
- [테스트 커버리지 계획](docs/TEST_COVERAGE_PLAN.md) - 테스트 전략
- [리팩토링 표준](docs/refactoring/STANDARDS_COMPLIANCE_REPORT.md) - 코드 품질 표준

## 🧪 테스트

```bash
# 모든 테스트 실행
make test

# 커버리지 포함 실행
uv run pytest --cov=src --cov-report=html

# 특정 테스트 실행
uv run pytest tests/unit/test_strategy.py
```

**테스트 원칙:**
- 커버리지 목표: 80%+ 유지, 핵심 모듈은 85%+ 지향
- 테스트 유형: 단위, 통합, 회귀, 프로퍼티 기반 테스트
- CI: 린트/타입 검사/테스트/커버리지 업로드를 자동화

## � 코드 품질 검사

### MyPy 타입 체크

```bash
# 전체 타입 체크
mypy src tests

# 또는 nox 사용
nox -s lint
```

이 프로젝트는 **MyPy 검증을 통한 완전한 타입 안전성**을 제공합니다:
- ✅ 모든 코드에 타입 힌트 적용
- ✅ MyPy strict mode 호환
- ✅ Pre-commit hook으로 자동 검사
- ✅ CI/CD 파이프라인에 통합

### Ruff 린팅 및 포매팅

```bash
# 코드 린팅 및 포매팅
ruff check src tests
ruff format src tests

# 또는 nox 사용
nox -s lint
```

### Pre-commit 훅

```bash
# Pre-commit 설치
pre-commit install

# 모든 파일에 대해 훅 실행
pre-commit run --all-files
```

## �🚢 배포

### Docker 배포

```bash
cd deploy
docker-compose up -d
```

### GCP 배포

자세한 GCP 배포 지침은 [deploy/README.md](deploy/README.md)를 참조하세요.

## 🤝 기여하기

기여를 환영합니다! 가이드라인은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'feat: add amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 📝 코드 품질

이 프로젝트는 현대적 Python 개발 표준을 따릅니다:

- ✅ **SOLID 원칙**: 의존성 주입을 통한 클린 아키텍처
- ✅ **타입 안전성**: MyPy 검증을 통한 완전한 타입 커버리지 (0 mypy 에러)
- ✅ **테스트**: pytest로 80% 이상 커버리지
- ✅ **린팅**: Ruff를 사용한 자동 코드 품질 검사
- ✅ **포매팅**: Ruff 포매터를 통한 일관된 코드 스타일
- ✅ **문서**: 포괄적인 문서 및 독스트링
- ✅ **자동화**: Pre-commit hook으로 커밋 전 품질 검사

## ⚠️ 면책 조항 및 위험 공고

**이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다.**

### 주요 위험

- 🔴 **자본 손실 위험**: 투자한 자본을 **완전히 잃을 수 있습니다**
- 🔴 **극변동성**: 암호화폐는 하루에 20-30% 이상 변동 가능
- 🔴 **보장 없음**: 과거 성과는 미래 결과를 **절대 보장하지 않습니다**
- 🔴 **시스템 위험**: 소프트웨어 버그, API 장애, 거래소 폐쇄 가능성

### 필수 읽기

사용 전에 반드시 읽어주세요:
- 📖 [면책조항 (DISCLAIMER.md)](DISCLAIMER.md) - 자세한 위험 경고
- 📖 [데이터 사용 정책 (DATA_USAGE_POLICY.md)](DATA_USAGE_POLICY.md) - 개인정보 보호
- 📖 [보안 정책 (SECURITY.md)](SECURITY.md) - API 키 보안 관행

### 책임 사항

**당신의 책임:**
- ✅ 실거래 전에 항상 모의 거래(Paper Trading)로 충분히 테스트
- ✅ 여유 자금(잃어도 상관없는 금액)으로만 거래
- ✅ 정기적으로 시스템 성과 모니터링
- ✅ 법적/세금 책임은 사용자가 부담

**작성자/기여자는 책임지지 않습니다:**
- ❌ 금융 손실
- ❌ 거래소 문제로 인한 피해
- ❌ 소프트웨어 버그로 인한 오류 거래
- ❌ 세금/규제 준수 실패

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🔐 보안 및 준수

### 보안 기능

- ✅ **환경 변수 기반 설정**: API 키는 소스 코드에 포함되지 않음
- ✅ **타입 안전성**: MyPy 검증으로 타입 오류 방지
- ✅ **코드 분석**: CodeQL을 통한 보안 취약점 자동 스캔
- ✅ **종속성 검사**: Bandit로 보안 문제 감지
- ✅ **데이터 보호**: 거래 기록 암호화 권장

### 보안 취약점 신고

보안 문제 발견 시:
1. ❌ 공개 이슈 생성 금지
2. ✅ [GitHub Security Advisory](https://github.com/11e3/crypto-quant-system/security/advisories/new) 사용
3. ✅ 또는 메인테이너에게 직접 연락

자세한 내용: [SECURITY.md](SECURITY.md)

### 준법 사항

**사용자는 다음을 준수해야 합니다:**

- 🇰🇷 한국: 특정금융정보법 (거래소 선택)
- 🇺🇸 미국: FinCEN, CFTC 규정
- 🇪🇺 유럽: MiFID II, GDPR 준수
- 📋 국제: 해당 국가의 암호화폐 거래 규정

**세금:**
- 💰 거래 수익은 과세 대상
- 📊 거래 기록 7년 보관 권장
- ⚠️ 세금 신고는 사용자 책임

## � 포트폴리오 & 학습 자료

### 📖 Jupyter Notebooks
시스템 사용 방법을 배우기 위한 **실습 가이드**:

- **[01-Backtesting-Case-Study.ipynb](notebooks/01-Backtesting-Case-Study.ipynb)**
  - 변동성 돌파 전략 실행 및 분석
  - 자산 곡선, 드로우다운, 거래 통계
  - 성능 지표 해석 (Sharpe, Sortino, Calmar)

- **[02-Portfolio-Optimization.ipynb](notebooks/02-Portfolio-Optimization.ipynb)**
  - 포트폴리오 구성: MPT vs 리스크 패리티 vs 켈리
  - 거래비용 모델링
  - 효율적 변경선(Efficient Frontier) 생성

- **[03-Live-Trading-Analysis.ipynb](notebooks/03-Live-Trading-Analysis.ipynb)**
  - 실시간 거래 시뮬레이션
  - 위험 관리 메커니즘
  - 라이브 거래 체크리스트

👉 **[전체 노트북 가이드](notebooks/README.md)** 참조

### 📊 성과 예시

예제 코드 실행 결과:
```
초기 자본: 1,000,000 KRW
최종 자산: 1,254,891 KRW
수익률: +25.5%
거래 수: 4,786회
승률: 35.4%
Sharpe 비율: 1.82
최대 낙폭: -12.3%
```

### 🎓 학습 경로

1. **기초** (1주): [README.md 읽기, 예제 실행](#-quick-start)
2. **백테스팅** (1주): [노트북 01 - 백테스팅 사례 연구](notebooks/01-Backtesting-Case-Study.ipynb)
3. **포트폴리오** (1주): [노트북 02 - 포트폴리오 최적화](notebooks/02-Portfolio-Optimization.ipynb)
4. **라이브** (1주): [노트북 03 - 실거래 시뮬레이션](notebooks/03-Live-Trading-Analysis.ipynb)
5. **실습** (진행 중): [사용자 정의 전략 개발](#-contributing)

## �🙏 감사의 말

- Upbit API 통합을 위한 [pyupbit](https://github.com/sharebook-kr/pyupbit)
- 다양한 거래소 지원 확장 가능한 아키텍처
- 모범 사례를 제공한 현대적 Python 개발 커뮤니티

## 📧 문의

질문이나 지원이 필요한 경우 GitHub에 이슈를 열어주세요.

---

<div align="center">

**정량적 거래를 위해 ❤️로 만들었습니다**

⭐ 유용하다고 생각되시면 이 저장소에 스타를 눌러주세요!

</div>
