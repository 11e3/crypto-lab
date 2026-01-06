# 🚀 Crypto Quant System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-495%20passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/Coverage-90%25-success.svg)
![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-black.svg)

**변동성 돌파 전략을 사용한 암호화폐 자동 거래 시스템**

[기능](#-features) • [빠른 시작](#-quick-start) • [아키텍처](#-architecture) • [문서](#-documentation) • [기여하기](#-contributing)

</div>

---

## 📋 개요

Crypto Quant System은 여러 암호화폐 거래소(Upbit 등)를 지원하는 프로덕션 준비가 완료된 자동 거래 시스템입니다. 포괄적인 백테스팅 기능, 실시간 거래 실행, 그리고 광범위한 성능 분석을 갖춘 정교한 변동성 돌파(VBO) 전략을 구현합니다.

### 🎯 주요 특징

- **고성능 백테스팅**: 빠른 과거 데이터 분석을 위한 pandas/numpy 기반 벡터화 엔진
- **모듈식 전략 시스템**: 유연한 전략 설계를 위한 구성 가능한 조건 및 필터
- **프로덕션 준비 완료**: 완전한 오류 처리, 로깅, 모니터링 및 Docker 배포
- **충분한 테스트**: 495개 이상의 테스트 케이스로 90% 이상의 테스트 커버리지
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
- 🧪 **높은 테스트 커버리지**: 단위 및 통합 테스트로 90% 이상 커버리지
- 📝 **타입 안전성**: MyPy 검증을 통한 완전한 타입 힌트
- 🔒 **보안**: 환경 변수 기반 설정, 하드코딩된 비밀번호 없음
- 🐳 **Docker 지원**: GCP/AWS 배포를 위한 프로덕션 준비 컨테이너화
- 📚 **포괄적인 문서**: 아키텍처 가이드, API 문서, 기여 가이드라인

## 🛠️ 기술 스택

### 핵심 기술
- **Python 3.11+**: 타입 힌트를 포함한 현대적 Python
- **pandas/numpy**: 데이터 처리 및 벡터화 연산
- **pydantic**: 타입 안전 설정 관리
- **click**: CLI 프레임워크
- **pyupbit**: Upbit API 통합 (다른 거래소 지원 확장 가능)

### 개발 도구
- **uv**: 빠른 Python 패키지 관리자
- **Ruff**: 린터 및 포맷터 (Black 대체)
- **MyPy**: 정적 타입 검사
- **pytest**: 85% 이상 커버리지의 테스트 프레임워크
- **pre-commit**: 코드 품질을 위한 Git 훅

### 인프라
- **Docker**: 컨테이너화
- **GCP**: 클라우드 배포 지원
- **WebSocket**: 실시간 시장 데이터

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/crypto-quant-system.git
cd crypto-quant-system

# uv 설치 (설치되지 않은 경우)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync --extra dev
```

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
├── tests/               # 테스트 스위트 (85% 이상 커버리지)
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

**테스트 통계:**
- 총 테스트: 495개 이상
- 커버리지: 90% 이상 (목표: 90%)
- 테스트 유형: 단위, 통합, 픽스처

## 🚢 배포

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
- ✅ **타입 힌트**: MyPy 검증을 통한 완전한 타입 커버리지
- ✅ **테스트**: pytest로 90% 이상 커버리지
- ✅ **린팅**: 코드 품질을 위한 Ruff
- ✅ **문서**: 포괄적인 문서 및 독스트링

## ⚠️ 면책 조항

**이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다.**

- 암호화폐 거래는 상당한 손실 위험이 수반됩니다
- 과거 성과는 미래 결과를 보장하지 않습니다
- 실거래 전에 항상 백테스팅으로 충분히 테스트하세요
- 본인의 책임 하에 사용하세요
- 작성자는 어떠한 금융 손실에 대해서도 책임지지 않습니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

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
