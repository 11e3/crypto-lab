# Scripts 디렉토리

이 디렉토리에는 목적별로 구성된 유틸리티 스크립트가 포함되어 있습니다.

## 구조

```
scripts/
├── tools/          # 개발 및 분석 도구
├── backtest/       # 백테스트 관련 스크립트
└── data/           # 데이터 관리 스크립트 (필요한 경우)
```

## 메인 CLI

대부분의 작업은 CLI 명령어를 사용하세요:

```bash
# 데이터 수집
upbit-quant collect --tickers KRW-BTC KRW-ETH

# 백테스팅
upbit-quant backtest --tickers KRW-BTC --interval day

# 거래 봇 실행
upbit-quant run-bot
```

## 개발 도구

`tools/`의 개발 도구는 디버깅 및 분석용입니다:
- 거래 비교
- 지표 비교
- 로직 검증

이들은 메인 CLI의 일부가 아닙니다.
