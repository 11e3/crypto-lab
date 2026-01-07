# 데이터 사용 정책 (Data Usage Policy)

## 개요

이 문서는 Crypto Quant System에서 거래 데이터, 시장 데이터, 그리고 사용자 정보를 어떻게 수집, 저장, 사용하는지 설명합니다.

---

## 1. 수집되는 데이터

### 1.1 시장 데이터 (Market Data)

이 시스템은 거래소 API를 통해 다음 데이터를 수집합니다:

```python
{
    "ticker": "KRW-BTC",           # 거래쌍
    "timestamp": "2025-01-07",     # 시간
    "open": 50000000,              # 시가
    "high": 51000000,              # 고가
    "low": 49000000,               # 저가
    "close": 50500000,             # 종가
    "volume": 100.5,               # 거래량
}
```

**출처**: 거래소 공개 API (Upbit 등)  
**목적**: 가격 분석, 신호 생성, 백테스팅  
**보관**: 로컬 저장소 (`.csv` 또는 Parquet 파일)

### 1.2 거래 데이터 (Trade Data)

실제 거래를 실행한 경우 다음 정보가 기록됩니다:

```python
{
    "trade_id": "12345",
    "timestamp": "2025-01-07 10:30:00",
    "ticker": "KRW-BTC",
    "side": "buy",                  # 매수/매도
    "quantity": 1.0,
    "price": 50000000,
    "fee": 25000,                   # 거래 수수료
    "status": "filled",             # 체결 상태
    "pnl": 500000,                  # 손익
}
```

**저장 위치**: 로컬 데이터베이스 (사용자 관리)  
**접근 권한**: 시스템/사용자만 접근  
**보호 방법**: 암호화 (권장)

### 1.3 포트폴리오 데이터 (Portfolio Data)

시스템이 관리하는 자산 정보:

```python
{
    "date": "2025-01-07",
    "total_equity": 1254891,        # 총 자산
    "cash": 500000,                 # 현금
    "positions": {
        "KRW-BTC": {
            "quantity": 0.025,
            "entry_price": 50000000,
            "current_value": 1250000,
        }
    },
    "daily_pnl": 25000,
}
```

**저장 위치**: 로컬 저장소 또는 클라우드 (사용자 선택)  
**데이터 소유**: 100% 사용자 소유  
**공개 범위**: 사용자가 명시적으로 공개하지 않는 한 비공개

---

## 2. 데이터 저장 및 보호

### 2.1 저장 위치

**기본 (권장):**
```
로컬 컴퓨터
└── ~/.crypto-quant/
    ├── data/
    │   ├── raw/              # OHLCV 데이터
    │   └── processed/        # 처리된 거래 기록
    ├── config/               # 설정 (API 키 제외)
    └── logs/                 # 거래 로그
```

**클라우드 (선택사항):**
- Google Cloud Storage (권장, 암호화)
- AWS S3 (권장, 버전 관리)
- Dropbox (개인용, 암호화됨)

### 2.2 보안 조치

#### ✅ 권장 보안 관행:

**1. API 키 격리**
```bash
# .env 파일 - 절대 공개하지 마세요
UPBIT_ACCESS_KEY="your-key-here"
UPBIT_SECRET_KEY="your-secret-here"

# .gitignore에 추가
echo ".env" >> .gitignore
```

**2. 파일 권한 제한**
```bash
# Linux/macOS
chmod 600 ~/.crypto-quant/.env
chmod 700 ~/.crypto-quant/

# Windows
icacls "%USERPROFILE%\.crypto-quant" /inheritance:r /grant "%USERNAME%:F"
```

**3. 데이터 암호화 (권장)**
```python
from cryptography.fernet import Fernet

# 마스터 키 생성 (한 번만)
master_key = Fernet.generate_key()

# 데이터 암호화
cipher = Fernet(master_key)
encrypted_data = cipher.encrypt(b"sensitive data")

# 데이터 복호화
decrypted_data = cipher.decrypt(encrypted_data)
```

**4. 정기적 백업**
```bash
# 주간 백업 (자동)
0 2 * * 0 tar -czf ~/backup/crypto-quant-$(date +%Y%m%d).tar.gz ~/.crypto-quant/

# 중요 파일만 백업
0 2 * * 0 cp -r ~/.crypto-quant/data ~/backup/
```

**5. 접근 로깅**
```python
import logging

logger = logging.getLogger('data_access')
handler = logging.FileHandler('~/.crypto-quant/logs/access.log')

logger.info(f"File accessed: {filename} at {timestamp}")
```

---

## 3. 데이터 사용 목적

### 3.1 허용된 사용

✅ **개인 투자/거래:**
- 자신의 자본으로 실제 거래 실행
- 전략 백테스팅 및 성과 분석
- 리스크 관리 및 포지션 추적

✅ **개선 및 최적화:**
- 거래 성과 분석
- 파라미터 튜닝
- 시스템 개선

✅ **학습 및 연구:**
- 금융 기술 학습
- 거래 전략 개발
- 성과 분석 기술 습득

✅ **법적 준수:**
- 세금 신고를 위한 거래 기록 유지
- 감시 기관 보고 (요청 시)
- 분쟁 해결 증거

### 3.2 금지된 사용

❌ **상업적 이용 (허가 없음):**
- 다른 사람에게 거래 신호 판매
- 관리형 펀드 운영
- 거래 자산운용 서비스 제공

❌ **데이터 공유:**
- 거래 기록을 제3자와 공유
- 포트폴리오 정보 공개 (동의 없음)
- API 키 공개

❌ **머신러닝 학습:**
- 공개되지 않은 개인 거래 데이터로 모델 학습
- 차별적 거래 신호 개발 및 판매

❌ **사기/조작:**
- 시스템을 이용한 사기 거래
- 시장 조작
- 무단 자동 거래

---

## 4. 제3자 데이터 공유

### 4.1 거래소 (필수)

**공유되는 정보:**
- 거래 주문 (필수)
- 거래 쌍, 수량, 가격, 시간

**법적 근거:** 거래소 서비스 이용약관

**사용자 통제:**
- API 키 관리 (사용자가 관리)
- 언제든 거래소와의 연결 해제 가능

### 4.2 클라우드 저장소 (선택)

**공유되는 정보:** 사용자가 직접 업로드한 데이터만

**공급자:**
- Google Cloud (Google Privacy Policy 준수)
- AWS (AWS Privacy Policy 준수)
- Dropbox (Dropbox Privacy Policy 준수)

**사용자 통제:**
- 업로드 여부 선택
- 언제든 삭제 가능
- 암호화 설정 가능

### 4.3 분석 도구 (선택)

**공유되지 않는 정보:**
- API 키
- 개인 거래 기록
- 포트폴리오 정보

**공유될 수 있는 정보:**
- 집계된 성과 지표 (선택)
- 시장 데이터 (공개 정보)

---

## 5. 데이터 보유 기간

### 5.1 보유 정책

| 데이터 유형 | 보유 기간 | 목적 | 삭제 방법 |
|-----------|---------|------|---------|
| OHLCV 데이터 | 무제한 | 백테스팅 | 수동 삭제 |
| 거래 기록 | 7년 | 세금 신고 | 자동 또는 수동 |
| 로그 파일 | 1개월 | 문제 진단 | 자동 순환 |
| 캐시 데이터 | 7일 | 성능 | 자동 삭제 |
| API 키 | 사용 중 | 거래소 연결 | 즉시 재설정 |

### 5.2 데이터 삭제

**수동 삭제:**
```bash
# 특정 기간 데이터 삭제
rm -rf ~/.crypto-quant/data/raw/2020-*

# 전체 초기화 (주의!)
rm -rf ~/.crypto-quant/
```

**자동 정리:**
```python
import os
from datetime import datetime, timedelta

# 30일 이상 된 로그 삭제
log_dir = os.path.expanduser("~/.crypto-quant/logs")
cutoff_date = datetime.now() - timedelta(days=30)

for log_file in os.listdir(log_dir):
    file_path = os.path.join(log_dir, log_file)
    if os.path.getmtime(file_path) < cutoff_date.timestamp():
        os.remove(file_path)
```

---

## 6. 사용자 권리

### 6.1 접근권 (Right to Access)

**당신의 권리:**
- ✅ 수집된 모든 데이터 조회 가능
- ✅ 거래 기록 내보내기
- ✅ 포트폴리오 스냅샷 저장

**실행 방법:**
```bash
# 모든 데이터 내보내기
python -c "
import os
import shutil
from datetime import datetime

backup_name = f'crypto-quant-backup-{datetime.now():%Y%m%d}.zip'
shutil.make_archive(backup_name, 'zip', os.path.expanduser('~/.crypto-quant'))
print(f'Backup created: {backup_name}')
"
```

### 6.2 삭제권 (Right to Delete)

**당신의 권리:**
- ✅ 개인 거래 데이터 삭제 (세금 신고 후)
- ✅ 포트폴리오 기록 삭제
- ✅ 로컬 데이터 전부 삭제

**주의:**
- ⚠️ 세금 신고를 위해 거래 기록은 최소 7년 보관 권장
- ⚠️ 클라우드 데이터는 각 공급자의 정책을 따름

### 6.3 이동권 (Right to Data Portability)

**당신의 권리:**
- ✅ 데이터를 다른 도구로 옮기기 가능
- ✅ 표준 포맷 (CSV, JSON) 제공

**지원 포맷:**
```python
# CSV 내보내기
df.to_csv('trades.csv', index=False)

# JSON 내보내기
import json
with open('portfolio.json', 'w') as f:
    json.dump(portfolio_data, f, indent=2)

# Parquet 내보내기
df.to_parquet('trades.parquet')
```

---

## 7. 거래소 데이터 정책

### 7.1 Upbit API 준수

이 시스템은 Upbit API 이용약관을 따릅니다:

- ✅ API 요청 제한 준수 (Rate Limiting)
- ✅ 공개 API만 사용 (비공개 데이터 없음)
- ✅ 저작권 준수 (Upbit 데이터 재판매 금지)

**Upbit 데이터 이용약관 참고:**
- [Upbit API Docs](https://docs.upbit.com/)
- [Upbit Terms](https://upbit.com/terms)

### 7.2 데이터 재사용 제한

❌ **금지 사항:**
- Upbit 데이터를 다른 사이트/서비스에 재판매
- Upbit 데이터를 기반으로 상업 서비스 개발
- API 데이터를 무단 재배포

✅ **허용 사항:**
- 개인적 거래 목적의 사용
- 교육 목적의 사용
- 오픈소스 프로젝트에서의 비상업 사용

---

## 8. 준법 선언

### 8.1 관련 법령

이 정책은 다음을 준수합니다:

🇰🇷 **한국:**
- 개인정보보호법
- 정보통신망법
- 금융투자법

🇺🇸 **미국:**
- GDPR (해당 시 적용)
- CCPA (캘리포니아)
- SEC 규정

🇪🇺 **유럽:**
- GDPR (General Data Protection Regulation)

### 8.2 정부 요청 대응

법적 요청을 받은 경우:

- ⚠️ 법원 명령이 없는 한 데이터 공개 안 함
- ⚠️ 사용자에게 요청 내용 통보 (법적으로 가능한 경우)
- ⚠️ 정당한 법적 절차만 인정

---

## 9. 데이터 보안 사고

### 9.1 사고 대응 절차

**만약 데이터가 유출되었다면:**

1. **즉시 조치**
   - API 키 재설정
   - 클라우드 동기화 중단
   - 거래소에 사고 신고

2. **검사**
   - 로그 파일 분석
   - 비인가 거래 확인
   - 영향받는 데이터 범위 파악

3. **복구**
   - 백업에서 복원
   - 데이터 무결성 검증
   - 시스템 재구성

### 9.2 신고 및 지원

**거래소 고객 지원:**
- Upbit: support@upbit.com
- [거래소별 신고 채널]

**법 집행 기관:**
- 한국: 경찰청 사이버수사대
- 국제: FBI IC3 (미국)

---

## 10. 정책 변경

### 10.1 변경 공지

**이 정책이 변경되면:**
- ✅ GitHub Releases에서 공지
- ✅ 메이저 변경은 새 버전 번호 할당
- ✅ 이전 버전 정책도 보관 유지

### 10.2 동의

이 시스템을 사용함으로써 당신은 이 정책에 동의합니다.

**최신 버전:** [GitHub 저장소](https://github.com/11e3/crypto-quant-system)

---

## 11. 문의

데이터 정책에 대한 질문:

- 📧 Issues: https://github.com/11e3/crypto-quant-system/issues
- 💬 Discussions: https://github.com/11e3/crypto-quant-system/discussions

---

**최종 업데이트:** 2025년 1월  
**정책 버전:** 1.0  
**효력:** 즉시

