# Exchange Layer API

거래소 레이어는 거래소와의 상호작용을 추상화합니다.

## 모듈 개요

- `src.exchange.base`: 거래소 인터페이스
- `src.exchange.upbit`: Upbit 거래소 구현
- `src.exchange.types`: 거래소 관련 타입 정의

## 핵심 인터페이스

### `Exchange` (추상 클래스)

모든 거래소 구현의 기본 인터페이스입니다.

#### 메서드

##### `get_balance(currency: str = "KRW") -> Balance`

잔고를 조회합니다.

**매개변수:**
- `currency`: 통화 코드 (기본값: "KRW")

**반환값:**
- `Balance`: 잔고 정보

**예외:**
- `ExchangeError`: 거래소 오류

##### `get_current_price(ticker: str) -> float`

현재가를 조회합니다.

**매개변수:**
- `ticker`: 티커 (예: "KRW-BTC")

**반환값:**
- `float`: 현재가

**예외:**
- `ExchangeError`: 거래소 오류

##### `buy_market_order(ticker: str, amount: float) -> Order`

시장가 매수 주문을 실행합니다.

**매개변수:**
- `ticker`: 티커
- `amount`: 매수 수량

**반환값:**
- `Order`: 주문 객체

**예외:**
- `InsufficientBalanceError`: 잔고 부족
- `ExchangeError`: 거래소 오류

##### `sell_market_order(ticker: str, amount: float) -> Order`

시장가 매도 주문을 실행합니다.

**매개변수:**
- `ticker`: 티커
- `amount`: 매도 수량

**반환값:**
- `Order`: 주문 객체

**예외:**
- `InsufficientBalanceError`: 잔고 부족
- `ExchangeError`: 거래소 오류

##### `get_ohlcv(ticker: str, interval: str, count: int = 200) -> pd.DataFrame`

OHLCV 데이터를 조회합니다.

**매개변수:**
- `ticker`: 티커
- `interval`: 시간 간격 ("day", "minute240" 등)
- `count`: 조회할 데이터 개수

**반환값:**
- `pd.DataFrame`: OHLCV 데이터

**예외:**
- `ExchangeError`: 거래소 오류

##### `get_ticker_info(ticker: str) -> Ticker`

티커 정보를 조회합니다.

**매개변수:**
- `ticker`: 티커

**반환값:**
- `Ticker`: 티커 정보 객체

## 구현 클래스

### `UpbitExchange`

Upbit 거래소의 구현 클래스입니다.

#### 생성자

```python
UpbitExchange(
    access_key: str | None = None,
    secret_key: str | None = None
)
```

**매개변수:**
- `access_key`: Upbit API Access Key (환경 변수에서 자동 로드 가능)
- `secret_key`: Upbit API Secret Key (환경 변수에서 자동 로드 가능)

**예외:**
- `ValueError`: API 키가 설정되지 않은 경우

**예시:**
```python
from src.exchange import UpbitExchange

# 환경 변수에서 자동 로드
exchange = UpbitExchange()

# 또는 직접 지정
exchange = UpbitExchange(
    access_key="your-access-key",
    secret_key="your-secret-key"
)
```

#### 메서드

`Exchange` 인터페이스의 모든 메서드를 구현합니다.

##### 추가 메서드

##### `get_orderbook(ticker: str) -> dict`

호가창 정보를 조회합니다.

**매개변수:**
- `ticker`: 티커

**반환값:**
- `dict`: 호가창 정보 (매수/매도 호가 목록)

## 데이터 타입

### `Balance`

잔고 정보를 나타내는 데이터 클래스입니다.

```python
@dataclass
class Balance:
    currency: str  # 통화 코드
    balance: float  # 잔고
    locked: float  # 주문 중인 금액
    available: float  # 사용 가능한 금액
```

### `Order`

주문 정보를 나타내는 데이터 클래스입니다.

```python
@dataclass
class Order:
    uuid: str  # 주문 UUID
    ticker: str  # 티커
    side: str  # "bid" (매수) or "ask" (매도)
    order_type: str  # "market" or "limit"
    price: float  # 주문 가격
    amount: float  # 주문 수량
    executed_amount: float  # 체결 수량
    state: str  # 주문 상태
    created_at: datetime  # 주문 생성 시간
```

### `Ticker`

티커 정보를 나타내는 데이터 클래스입니다.

```python
@dataclass
class Ticker:
    market: str  # 마켓 코드
    trade_price: float  # 현재가
    trade_volume: float  # 거래량
    acc_trade_volume_24h: float  # 24시간 누적 거래량
    acc_trade_price_24h: float  # 24시간 누적 거래대금
    highest_price_52w: float  # 52주 최고가
    lowest_price_52w: float  # 52주 최저가
    prev_closing_price: float  # 전일 종가
    change: str  # 전일 대비 ("RISE", "FALL", "EVEN")
    change_rate: float  # 전일 대비 등락률
```

## 사용 예제

### 기본 사용

```python
from src.exchange import UpbitExchange

# 거래소 인스턴스 생성
exchange = UpbitExchange()

# 현재가 조회
price = exchange.get_current_price("KRW-BTC")
print(f"BTC 현재가: {price:,.0f}원")

# 잔고 조회
balance = exchange.get_balance("KRW")
print(f"KRW 잔고: {balance.available:,.0f}원")

# OHLCV 데이터 조회
df = exchange.get_ohlcv("KRW-BTC", interval="day", count=100)
print(df.head())
```

### 주문 실행

```python
from src.exchange import UpbitExchange

exchange = UpbitExchange()

# 시장가 매수
order = exchange.buy_market_order("KRW-BTC", amount=0.001)
print(f"매수 주문: {order.uuid}")

# 시장가 매도
order = exchange.sell_market_order("KRW-BTC", amount=0.001)
print(f"매도 주문: {order.uuid}")
```

### 오류 처리

```python
from src.exchange import UpbitExchange
from src.exceptions.exchange import InsufficientBalanceError, ExchangeError

exchange = UpbitExchange()

try:
    order = exchange.buy_market_order("KRW-BTC", amount=1000.0)
except InsufficientBalanceError:
    print("잔고가 부족합니다.")
except ExchangeError as e:
    print(f"거래소 오류: {e}")
```

### 커스텀 거래소 구현

```python
from src.exchange.base import Exchange
from src.exchange.types import Balance, Order, Ticker
import pandas as pd

class MyCustomExchange(Exchange):
    """커스텀 거래소 구현 예제"""
    
    def get_balance(self, currency: str = "KRW") -> Balance:
        # 구현...
        pass
    
    def get_current_price(self, ticker: str) -> float:
        # 구현...
        pass
    
    def buy_market_order(self, ticker: str, amount: float) -> Order:
        # 구현...
        pass
    
    def sell_market_order(self, ticker: str, amount: float) -> Order:
        # 구현...
        pass
    
    def get_ohlcv(self, ticker: str, interval: str, count: int = 200) -> pd.DataFrame:
        # 구현...
        pass
    
    def get_ticker_info(self, ticker: str) -> Ticker:
        # 구현...
        pass
```

## 관련 문서

- [시작 가이드](../guides/getting_started.md)
- [아키텍처 문서](../architecture.md)
