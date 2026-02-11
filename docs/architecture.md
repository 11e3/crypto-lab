# 시스템 아키텍처

## 개요

crypto-lab은 모듈화된 아키텍처를 사용하여 거래 전략을 백테스팅하고 실시간으로 실행할 수 있는 시스템입니다.

## 아키텍처 다이어그램

> **참고**: 더 자세한 다이어그램은 [Architecture Diagrams](architecture_diagrams.md) 문서를 참고하세요.

```
┌─────────────────────────────────────────────────────────────┐
│                    TradingBotFacade                         │
│  (Facade Pattern - 단순화된 인터페이스 제공)                    │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼──────┐
│ EventBus    │  │ Managers    │
│ (Pub-Sub)   │  │             │
└──────┬──────┘  └──────┬──────┘
       │                │
       │      ┌─────────┼─────────┐
       │      │         │         │
┌──────▼──────▼──┐ ┌───▼───┐ ┌───▼──────┐
│ OrderManager   │ │Position│ │ Signal  │
│                │ │Manager │ │ Handler │
└──────┬─────────┘ └───┬───┘ └────┬─────┘
       │               │           │
       └───────┬───────┴───────────┘
               │
       ┌───────▼────────┐
       │   Exchange    │
       │  (Interface)  │
       └───────┬───────┘
               │
       ┌───────▼────────┐
       │ UpbitExchange  │
       │  (Implementation)
       └────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Strategy Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Strategy   │  │  Conditions  │  │   Filters    │       │
│  │  (Base)      │  │              │  │              │       │
│  └──────┬───────┘  └──────────────┘  └──────────────┘       │
│         │                                                   │
│  ┌──────▼───────┐                                           │
│  │    VBOV1     │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  DataSource  │  │   Cache      │  │  Collector    │      │
│  │  (Interface) │  │              │  │               │      │
│  └──────┬───────┘  └──────────────┘  └──────────────┘       │
│         │                                                   │
│  ┌──────▼───────┐                                           │
│  │UpbitDataSource│                                          │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Backtest Engine                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Engine     │  │   Report     │  │   Metrics    │       │
│  │ (Vectorized) │  │  Generator   │  │  Calculator  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## 핵심 컴포넌트

### 1. Exchange Layer (`src/exchange/`)

**목적**: 거래소 추상화 계층

- **`Exchange`**: 거래소 인터페이스 (ABC)
  - `get_balance()`: 잔고 조회
  - `get_current_price()`: 현재가 조회
  - `buy_market_order()`: 시장가 매수
  - `sell_market_order()`: 시장가 매도
  - `get_ohlcv()`: OHLCV 데이터 조회

- **`UpbitExchange`**: Upbit 거래소 구현
  - `pyupbit` 라이브러리 래핑
  - 에러 처리 및 재시도 로직

### 2. Execution Layer (`src/execution/`)

**목적**: 거래 실행 및 포지션 관리

- **`TradingBotFacade`**: Facade 패턴으로 복잡한 시스템을 단순화
  - 일일 리셋
  - 타겟 가격 재계산
  - 이벤트 버스 초기화

- **`OrderManager`**: 주문 관리
  - 주문 생성 및 추적
  - 주문 상태 확인
  - 이벤트 발행

- **`PositionManager`**: 포지션 관리
  - 포지션 추가/제거
  - PnL 계산
  - 이벤트 발행

- **`SignalHandler`**: 신호 처리
  - 진입/청산 신호 확인
  - 전략 메트릭 계산
  - 이벤트 발행

- **`EventBus`**: 이벤트 버스 (Pub-Sub 패턴)
  - 이벤트 구독/발행
  - 핸들러 등록

- **Handlers**:
  - `TradeHandler`: 거래 이벤트 처리
  - `NotificationHandler`: 알림 이벤트 처리

### 3. Strategy Layer (`src/strategies/`)

**목적**: 거래 전략 정의 및 실행

- **`Strategy`**: 전략 기본 클래스 (ABC)
  - `calculate_indicators()`: 지표 계산
  - `generate_signals()`: 신호 생성
  - `check_entry()`: 진입 조건 확인
  - `check_exit()`: 청산 조건 확인

- **`VBOV1`**: 변동성 돌파 전략
  - 고정 K=0.5, SMA 기반 추세 필터
  - BTC MA20 시장 필터
  - 시가 매도 (exit_price_base 컨벤션)

- **Conditions & Filters**: 모듈화된 조건 및 필터
  - 재사용 가능한 컴포넌트
  - 전략 조합 가능

### 4. Data Layer (`src/data/`)

**목적**: 데이터 수집 및 관리

- **`DataSource`**: 데이터 소스 인터페이스 (ABC)
  - `get_ohlcv()`: OHLCV 데이터 조회
  - `collect_candles()`: 캔들 데이터 수집

- **`UpbitDataSource`**: Upbit 데이터 소스 구현
  - 증분 업데이트
  - Parquet 형식 저장

- **`IndicatorCache`**: 지표 캐싱
  - 계산된 지표 저장
  - 재사용으로 성능 향상

- **`UpbitDataCollector`**: 데이터 수집기
  - 여러 티커/인터벌 일괄 수집
  - CLI 통합

### 5. Backtest Layer (`src/backtester/`)

**목적**: 백테스팅 엔진 및 리포트 생성

- **`VectorizedBacktestEngine`**: 벡터화된 백테스트 엔진
  - Pandas/NumPy 기반 고성능 시뮬레이션
  - 사전 계산된 신호 사용
  - 슬리피지 및 수수료 모델링

- **`BacktestReport`**: 리포트 생성
  - 성능 메트릭 계산
  - 차트 생성 (Equity Curve, MDD, 히트맵)
  - HTML 리포트 생성

### 6. Configuration (`src/config/`)

**목적**: 설정 관리

- **`ConfigLoader`**: YAML 설정 로더
  - 환경 변수 오버라이드
  - 타입 안전한 설정 접근

- **`constants.py`**: 상수 정의
  - 경로 상수
  - 기본값 상수

### 7. Utilities (`src/utils/`)

**목적**: 공통 유틸리티

- **`logger.py`**: 로깅 시스템
  - 구조화된 로깅
  - 컨텍스트 로깅
  - 성능 로깅

- **`indicators.py`**: 기술 지표
  - SMA, 노이즈 비율 등

- **`telegram.py`**: Telegram 알림

## 설계 원칙

### 1. 의존성 역전 원칙 (DIP)

- 구체적인 구현이 아닌 추상 인터페이스에 의존
- `Exchange`, `DataSource`, `Strategy` 인터페이스 사용

### 2. 단일 책임 원칙 (SRP)

- 각 클래스는 하나의 책임만 가짐
- `OrderManager`, `PositionManager`, `SignalHandler` 분리

### 3. 개방-폐쇄 원칙 (OCP)

- 확장에는 열려있고 수정에는 닫혀있음
- 새로운 전략, 거래소, 데이터 소스 추가 용이

### 4. Facade 패턴

- `TradingBotFacade`로 복잡한 시스템을 단순화
- 클라이언트는 Facade만 사용

### 5. 이벤트 기반 아키텍처

- Pub-Sub 패턴으로 느슨한 결합
- 이벤트 핸들러로 확장 가능

## 데이터 흐름

### 백테스팅 흐름

```
1. 데이터 로드 (Parquet)
   ↓
2. 전략 지표 계산
   ↓
3. 신호 생성
   ↓
4. 벡터화된 시뮬레이션
   ↓
5. 메트릭 계산
   ↓
6. 리포트 생성
```

### 실시간 거래 흐름

```
1. WebSocket으로 가격 업데이트 수신
   ↓
2. SignalHandler가 진입/청산 신호 확인
   ↓
3. OrderManager가 주문 생성
   ↓
4. PositionManager가 포지션 업데이트
   ↓
5. 이벤트 발행 (TradeHandler, NotificationHandler)
   ↓
6. Telegram 알림 전송
```

## 확장성

### 새로운 거래소 추가

1. `Exchange` 인터페이스 구현
2. `UpbitExchange`와 동일한 패턴으로 구현
3. `TradingBotFacade`에 주입

### 새로운 전략 추가

1. `Strategy` 인터페이스 구현
2. `calculate_indicators()`, `generate_signals()` 구현
3. `VBOV1`와 동일한 패턴

### 새로운 데이터 소스 추가

1. `DataSource` 인터페이스 구현
2. `get_ohlcv()`, `collect_candles()` 구현
3. `SignalHandler`에 주입

## 성능 최적화

1. **벡터화**: Pandas/NumPy 사용으로 루프 최소화
2. **캐싱**: 계산된 지표를 `data/processed/`에 저장
3. **사전 계산**: 백테스트 전 모든 신호 계산
4. **이벤트 기반**: 비동기 처리로 블로킹 최소화
