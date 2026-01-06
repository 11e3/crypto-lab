# 테스트 픽스처

이 디렉토리에는 테스트 스위트를 위한 재사용 가능한 테스트 픽스처와 샘플 데이터가 포함되어 있습니다.

## 구조

```
fixtures/
├── data/              # 샘플 OHLCV 및 시장 데이터 생성기
│   └── sample_ohlcv.py
├── config/            # 테스트 설정 파일
│   └── test_settings.yaml
├── mock_exchange.py   # Mock Exchange 구현
└── README.md          # 이 파일
```

## 데이터 픽스처

### `sample_ohlcv.py`

테스트를 위한 현실적인 OHLCV 데이터를 생성하는 함수를 제공합니다:

- `generate_ohlcv_data()`: 구성 가능한 파라미터로 표준 OHLCV 데이터 생성
- `generate_trending_data()`: 명확한 트렌드(상승/하락)가 있는 데이터 생성
- `generate_volatile_data()`: 스트레스 테스트를 위한 고변동성 데이터 생성
- `generate_multiple_tickers_data()`: 여러 티커에 대한 데이터를 동시에 생성

**사용법:**
```python
from tests.fixtures.data.sample_ohlcv import generate_ohlcv_data

# 100일치 데이터 생성
df = generate_ohlcv_data(periods=100, base_price=50_000_000.0, seed=42)
```

## 설정 픽스처

### `test_settings.yaml`

`config/settings.yaml`의 구조를 반영하지만 다음을 포함하는 테스트 설정 파일:
- 테스트 안전 값 (실제 API 키 없음)
- 외부 서비스 비활성화 (Telegram)
- 더 빠른 테스트를 위한 최소 티커 목록

**사용법:**
```python
from pathlib import Path
from src.config.loader import get_config

test_config_path = Path(__file__).parent / "fixtures" / "config" / "test_settings.yaml"
config = get_config(test_config_path)
```

## Mock Exchange

### `mock_exchange.py`

테스트를 위한 `Exchange` 인터페이스의 완전한 mock 구현:

- 메모리 내 상태 관리 (잔액, 주문, 가격)
- 구성 가능한 실패 모드
- 현실적인 주문 실행 시뮬레이션

**사용법:**
```python
from tests.fixtures.mock_exchange import MockExchange

exchange = MockExchange()
exchange.set_balance("KRW", 1_000_000.0)
exchange.set_price("KRW-BTC", 50_000_000.0)
```

## Pytest 픽스처

모든 픽스처는 `conftest.py`를 통해 테스트에서 자동으로 사용 가능합니다:

- `sample_ohlcv_data`: 표준 OHLCV DataFrame (100 기간)
- `trending_ohlcv_data`: 트렌드가 있는 OHLCV DataFrame
- `volatile_ohlcv_data`: 변동성이 큰 OHLCV DataFrame
- `multiple_tickers_data`: 여러 티커에 대한 DataFrame 딕셔너리
- `mock_exchange`: MockExchange 인스턴스
- `vbo_strategy`: VanillaVBO 전략 인스턴스
- `sample_balance`: 샘플 Balance 객체
- `sample_order`: 샘플 Order 객체
- `sample_ticker`: 샘플 Ticker 객체
- `test_config_path`: 테스트 설정 파일 경로

**사용법:**
```python
def test_my_function(sample_ohlcv_data, mock_exchange):
    # 픽스처를 직접 사용
    df = sample_ohlcv_data
    exchange = mock_exchange
    # ... 테스트 코드
```
