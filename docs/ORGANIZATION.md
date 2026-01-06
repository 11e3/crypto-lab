# 문서 구조

이 문서는 저장소 내 모든 마크다운 파일의 구조를 설명합니다.

## 📁 디렉토리 구조

```
docs/
├── portfolio/          # 포트폴리오 공개 문서
│   ├── README.md
│   ├── PORTFOLIO_PUBLICATION_GUIDE.md
│   ├── PUBLICATION_CHECKLIST.md
│   ├── PORTFOLIO_CHECKLIST.md
│   └── PORTFOLIO_PUBLICATION_PLAN.md
│
├── maintenance/        # 유지보수 및 정리 문서
│   ├── README.md
│   ├── CLEANUP_RECOMMENDATIONS.md
│   ├── CLEANUP_SCRIPT.md
│   ├── FILES_TO_DELETE.md
│   └── GITIGNORE_AUDIT.md
│
├── guides/             # 사용자 가이드
│   ├── getting_started.md
│   ├── configuration.md
│   ├── strategy_customization.md
│   └── deprecation_guide.md
│
├── planning/           # 프로젝트 계획 문서
│   ├── TEST_COVERAGE_PLAN.md
│   ├── COVERAGE_PROGRESS.md
│   ├── COVERAGE_90_PERCENT_PLAN.md
│   ├── CONFIGURATION_STANDARD.md
│   └── comparison_legacy_vs_new_bot.md
│
├── refactoring/        # 리팩토링 문서
│   ├── REFACTORING_SUMMARY.md
│   ├── MODERN_PYTHON_STANDARDS_MIGRATION.md
│   └── STANDARDS_COMPLIANCE_REPORT.md
│
├── archive/            # 역사적/아카이브 문서
│   └── [21개의 단계 완료 문서]
│
├── api/                # API 문서
│   └── README.md
│
├── architecture.md     # 시스템 아키텍처
└── README.md           # 문서 인덱스
```

## 📄 루트 레벨 파일

루트에 남아있는 필수 문서 파일:

- `README.md` - 메인 프로젝트 README
- `CONTRIBUTING.md` - 기여 가이드라인
- `SECURITY.md` - 보안 정책

## 🎯 빠른 참조

### 사용자를 위한
- **시작하기**: `docs/guides/getting_started.md`
- **설정**: `docs/guides/configuration.md`
- **전략 커스터마이징**: `docs/guides/strategy_customization.md`
- **아키텍처**: `docs/architecture.md`

### 포트폴리오 공개를 위한
- **메인 가이드**: `docs/portfolio/PORTFOLIO_PUBLICATION_GUIDE.md`
- **빠른 체크리스트**: `docs/portfolio/PUBLICATION_CHECKLIST.md`

### 유지보수를 위한
- **정리 가이드**: `docs/maintenance/CLEANUP_RECOMMENDATIONS.md`
- **삭제할 파일**: `docs/maintenance/FILES_TO_DELETE.md`

## 📝 구조 원칙

1. **루트 레벨**: 필수 파일만 (README, CONTRIBUTING, SECURITY)
2. **docs/portfolio/**: 포트폴리오 공개 관련 모든 문서
3. **docs/maintenance/**: 정리, 감사, 유지보수 가이드
4. **docs/guides/**: 사용자 대상 가이드 및 튜토리얼
5. **docs/planning/**: 프로젝트 계획 및 추적 문서
6. **docs/archive/**: 역사적/아카이브 문서
7. **docs/refactoring/**: 리팩토링 문서

## 🔄 마이그레이션 요약

**`docs/portfolio/`로 이동:**
- `PORTFOLIO_PUBLICATION_GUIDE.md`
- `PUBLICATION_CHECKLIST.md`
- `docs/planning/PORTFOLIO_CHECKLIST.md`
- `docs/planning/PORTFOLIO_PUBLICATION_PLAN.md`

**`docs/maintenance/`로 이동:**
- `CLEANUP_RECOMMENDATIONS.md`
- `CLEANUP_SCRIPT.md`
- `FILES_TO_DELETE.md`
- `GITIGNORE_AUDIT.md`
