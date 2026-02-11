# Execution Layer API

실행 레이어는 실시간 거래 봇의 핵심 컴포넌트를 제공합니다.

## 모듈 개요

- `src.execution.bot_facade`: 거래 봇 Facade (주 진입점)
- `src.execution.order_manager`: 주문 관리
- `src.execution.position_manager`: 포지션 관리
- `src.execution.signal_handler`: 신호 처리
- `src.execution.event_bus`: 이벤트 버스 (Pub-Sub 패턴)

## 핵심 클래스

### `TradingBotFacade`

거래 봇의 Facade 패턴 구현으로, 복잡한 시스템을 단순화된 인터페이스로 제공합니다.

#### 생성자

```python
TradingBotFacade(
    exchange: Exchange | None = None,
    position_manager: PositionManager | None = None,
    order_manager: OrderManager | None = None,
    signal_handler: SignalHandler | None = None,
    strategy: VBOV1 | None = None,
    config_path: Path | None = None
)
```

**매개변수:**
- `exchange`: 거래소 인스턴스 (None이면 자동 생성)
- `position_manager`: 포지션 관리자 인스턴스 (None이면 자동 생성)
- `order_manager`: 주문 관리자 인스턴스 (None이면 자동 생성)
- `signal_handler`: 신호 처리기 인스턴스 (None이면 자동 생성)
- `strategy`: 전략 인스턴스 (None이면 자동 생성)
- `config_path`: 설정 파일 경로 (기본값: `config/settings.yaml`)

**예외:**
- `ValueError`: Upbit API 키가 설정되지 않은 경우

#### 메서드

##### `run() -> None`

거래 봇을 실행합니다. 무한 루프로 실행되며, 일일 리셋과 신호 처리를 수행합니다.

**예시:**
```python
from src.execution.bot_facade import TradingBotFacade
from pathlib import Path

bot = TradingBotFacade(config_path=Path("config/settings.yaml"))
bot.run()  # 봇 실행 (무한 루프)
```

##### `stop() -> None`

거래 봇을 안전하게 중지합니다.

##### `get_positions() -> dict[str, Position]`

현재 보유 중인 모든 포지션을 반환합니다.

**반환값:**
- `dict[str, Position]`: 티커를 키로 하는 포지션 딕셔너리

##### `get_balance() -> Balance`

현재 잔고를 조회합니다.

**반환값:**
- `Balance`: 잔고 정보

### `OrderManager`

주문 실행을 관리하는 클래스입니다.

#### 생성자

```python
OrderManager(
    exchange: Exchange,
    position_manager: PositionManager,
    event_bus: EventBus | None = None
)
```

**매개변수:**
- `exchange`: 거래소 인스턴스
- `position_manager`: 포지션 관리자 인스턴스
- `event_bus`: 이벤트 버스 (선택)

#### 메서드

##### `execute_buy(ticker: str, amount: float) -> Order | None`

매수 주문을 실행합니다.

**매개변수:**
- `ticker`: 거래할 티커
- `amount`: 매수 수량

**반환값:**
- `Order | None`: 성공 시 주문 객체, 실패 시 None

**예외:**
- `InsufficientBalanceError`: 잔고 부족
- `ExchangeError`: 거래소 오류

##### `execute_sell(ticker: str, amount: float) -> Order | None`

매도 주문을 실행합니다.

**매개변수:**
- `ticker`: 거래할 티커
- `amount`: 매도 수량

**반환값:**
- `Order | None`: 성공 시 주문 객체, 실패 시 None

### `PositionManager`

포지션 추적 및 관리를 담당하는 클래스입니다.

#### 생성자

```python
PositionManager(
    exchange: Exchange,
    max_slots: int = 4
)
```

**매개변수:**
- `exchange`: 거래소 인스턴스
- `max_slots`: 최대 보유 종목 수

#### 메서드

##### `add_position(ticker: str, amount: float, entry_price: float) -> Position`

새 포지션을 추가합니다.

**매개변수:**
- `ticker`: 티커
- `amount`: 수량
- `entry_price`: 진입 가격

**반환값:**
- `Position`: 생성된 포지션 객체

##### `remove_position(ticker: str) -> Position | None`

포지션을 제거합니다.

**매개변수:**
- `ticker`: 티커

**반환값:**
- `Position | None`: 제거된 포지션 또는 None

##### `get_position(ticker: str) -> Position | None`

특정 티커의 포지션을 조회합니다.

##### `get_all_positions() -> dict[str, Position]`

모든 포지션을 반환합니다.

##### `has_slot() -> bool`

새 포지션을 추가할 수 있는 슬롯이 있는지 확인합니다.

**반환값:**
- `bool`: 슬롯 사용 가능 여부

##### `is_holding(ticker: str) -> bool`

특정 티커를 보유 중인지 확인합니다.

### `SignalHandler`

거래 신호를 처리하는 클래스입니다.

#### 생성자

```python
SignalHandler(
    strategy: Strategy,
    position_manager: PositionManager,
    order_manager: OrderManager,
    exchange: Exchange
)
```

**매개변수:**
- `strategy`: 거래 전략
- `position_manager`: 포지션 관리자
- `order_manager`: 주문 관리자
- `exchange`: 거래소

#### 메서드

##### `process_signals(ticker: str, df: pd.DataFrame) -> None`

신호를 처리하고 필요한 경우 주문을 실행합니다.

**매개변수:**
- `ticker`: 처리할 티커
- `df`: OHLCV 및 지표 데이터

### `EventBus`

Pub-Sub 패턴을 구현한 이벤트 버스입니다.

#### 함수

##### `get_event_bus() -> EventBus`

전역 이벤트 버스 인스턴스를 반환합니다.

#### 메서드

##### `subscribe(event_type: type, handler: Callable) -> None`

이벤트 구독을 등록합니다.

**매개변수:**
- `event_type`: 구독할 이벤트 타입
- `handler`: 이벤트 핸들러 함수

**예시:**
```python
from src.execution.event_bus import get_event_bus
from src.execution.events import OrderFilledEvent

def on_order_filled(event: OrderFilledEvent):
    print(f"주문 체결: {event.ticker}")

bus = get_event_bus()
bus.subscribe(OrderFilledEvent, on_order_filled)
```

##### `publish(event: Event) -> None`

이벤트를 발행합니다.

**매개변수:**
- `event`: 발행할 이벤트 객체

## 이벤트 타입

### `OrderFilledEvent`

주문이 체결되었을 때 발생하는 이벤트입니다.

```python
@dataclass
class OrderFilledEvent:
    ticker: str
    order_type: str  # "buy" or "sell"
    amount: float
    price: float
    timestamp: datetime
```

### `PositionOpenedEvent`

포지션이 열렸을 때 발생하는 이벤트입니다.

### `PositionClosedEvent`

포지션이 닫혔을 때 발생하는 이벤트입니다.

## 사용 예제

### 기본 봇 실행

```python
from src.execution.bot_facade import TradingBotFacade
from pathlib import Path

# 설정 파일 사용
bot = TradingBotFacade(config_path=Path("config/settings.yaml"))
bot.run()
```

### 커스텀 컴포넌트로 봇 생성

```python
from src.execution.bot_facade import TradingBotFacade
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.signal_handler import SignalHandler
from src.exchange import UpbitExchange
from src.strategies.volatility_breakout import VBOV1

# 컴포넌트 생성
exchange = UpbitExchange(access_key="...", secret_key="...")
position_manager = PositionManager(exchange, max_slots=4)
order_manager = OrderManager(exchange, position_manager)
strategy = VBOV1()
signal_handler = SignalHandler(strategy, position_manager, order_manager, exchange)

# Facade 생성
bot = TradingBotFacade(
    exchange=exchange,
    position_manager=position_manager,
    order_manager=order_manager,
    signal_handler=signal_handler,
    strategy=strategy
)

bot.run()
```

### 이벤트 구독

```python
from src.execution.event_bus import get_event_bus
from src.execution.events import OrderFilledEvent, PositionOpenedEvent

def on_order_filled(event: OrderFilledEvent):
    print(f"주문 체결: {event.ticker} @ {event.price}")

def on_position_opened(event: PositionOpenedEvent):
    print(f"포지션 오픈: {event.ticker}")

bus = get_event_bus()
bus.subscribe(OrderFilledEvent, on_order_filled)
bus.subscribe(PositionOpenedEvent, on_position_opened)

# 봇 실행
bot = TradingBotFacade()
bot.run()
```

## 관련 문서

- [시작 가이드](../guides/getting_started.md)
- [아키텍처 문서](../architecture.md)
