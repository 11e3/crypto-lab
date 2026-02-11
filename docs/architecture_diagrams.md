# Architecture Diagrams

This document contains detailed architecture diagrams for the Upbit Quant System.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   CLI Commands│  │  Python API │  │ Jupyter Notebook│       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼────────────────┼────────────────┼──────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
          ┌───────────────────┴───────────────────┐
          │                                       │
    ┌─────▼──────┐                        ┌──────▼──────┐
    │ Backtest   │                        │ Live Trading│
    │  Engine    │                        │    Bot      │
    └─────┬──────┘                        └──────┬──────┘
          │                                       │
          └───────────────────┬──────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Strategy System     │
                    │  (VBOV1, etc.)  │
                    └───────────┬───────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
    ┌─────▼──────┐      ┌───────▼──────┐    ┌────────▼──────┐
    │   Data     │      │  Exchange    │    │  Execution     │
    │   Layer    │      │   Layer      │    │   Layer       │
    └────────────┘      └──────────────┘    └───────────────┘
```

## Backtesting Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Backtest Engine                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              VectorizedBacktestEngine                      │  │
│  │                                                            │  │
│  │  1. Load Data (from cache or source)                      │  │
│  │  2. Calculate Indicators (vectorized)                     │  │
│  │  3. Generate Signals (entry/exit)                          │  │
│  │  4. Simulate Trades (vectorized)                          │  │
│  │  5. Calculate Metrics                                      │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                        │                                         │
│  ┌─────────────────────▼─────────────────────────────────────┐  │
│  │                  BacktestResult                            │  │
│  │  - Trades                                                  │  │
│  │  - Equity Curve                                            │  │
│  │  - Performance Metrics                                     │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                        │                                         │
│  ┌─────────────────────▼─────────────────────────────────────┐  │
│  │                  Report Generator                          │  │
│  │  - HTML Reports                                            │  │
│  │  - Charts (Equity, Drawdown, Monthly)                     │  │
│  │  - Performance Tables                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Live Trading Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TradingBotFacade                               │
│  (Facade Pattern - Simplified Interface)                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    EventBus                                │  │
│  │              (Pub-Sub Pattern)                              │  │
│  │                                                            │  │
│  │  Events:                                                   │  │
│  │  - Market Data Updates                                     │  │
│  │  - Strategy Signals                                        │  │
│  │  - Order Executions                                        │  │
│  │  - Position Changes                                        │  │
│  └──────┬─────────────────────────────────────────────────────┘  │
│         │                                                        │
│  ┌──────┴─────────────────────────────────────────────────────┐  │
│  │                    Managers                                │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │
│  │  │   Order      │  │  Position    │  │   Signal     │    │  │
│  │  │  Manager     │  │  Manager    │  │   Handler    │    │  │
│  │  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘    │  │
│  │         │                  │                 │            │  │
│  │         └──────────────────┼─────────────────┘            │  │
│  │                            │                              │  │
│  │                    ┌────────▼────────┐                     │  │
│  │                    │   Exchange     │                     │  │
│  │                    │  (Interface)  │                     │  │
│  │                    └────────┬───────┘                     │  │
│  │                             │                             │  │
│  │                    ┌────────▼────────┐                   │  │
│  │                    │  UpbitExchange  │                   │  │
│  │                    │ (Implementation)│                   │  │
│  │                    └──────────────────┘                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Strategy System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Strategy Base                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Strategy (ABC)                         │  │
│  │                                                            │  │
│  │  - name: str                                              │  │
│  │  - entry_conditions: ConditionGroup                       │  │
│  │  - exit_conditions: ConditionGroup                       │  │
│  │                                                            │  │
│  │  Methods:                                                 │  │
│  │  - should_enter() -> bool                                 │  │
│  │  - should_exit() -> bool                                  │  │
│  │  - calculate_target_price() -> float                      │  │
│  └──────┬─────────────────────────────────────────────────────┘  │
│         │                                                        │
│  ┌──────┴─────────────────────────────────────────────────────┐  │
│  │              Concrete Strategies                           │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │
│  │  │         VBOV1             │                    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Condition System                          │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │
│  │  │SMABreakout   │  │   Trend      │  │    Noise     │    │  │
│  │  │ Condition    │  │  Condition   │  │  Condition   │    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌──────────────┐                      │  │
│  │  │  Breakout     │  │   Whipsaw   │                      │  │
│  │  │  Condition    │  │   Exit      │                      │  │
│  │  └──────────────┘  └──────────────┘                      │  │
│  │                                                            │  │
│  │  All conditions implement:                                │  │
│  │  - check(data: DataFrame) -> Series[bool]                │  │
│  │  - name: str                                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Collection                              │
│                                                                  │
│  Upbit API ──► UpbitDataCollector ──► Cache (Parquet)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Access Layer                             │
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │   Cache      │◄────────┤ DataSource   │                     │
│  │  (Parquet)   │         │  (Interface) │                     │
│  └──────┬───────┘         └──────┬───────┘                     │
│         │                         │                             │
│         │                         ▼                             │
│         │              ┌───────────────────┐                     │
│         │              │ UpbitDataSource  │                     │
│         │              └───────────────────┘                     │
│         │                                                         │
│         └───────────────────┬─────────────────────┐               │
│                             │                     │               │
│                    ┌────────▼────────┐   ┌────────▼────────┐     │
│                    │  Backtest       │   │  Live Trading   │     │
│                    │  Engine         │   │  Bot            │     │
│                    └─────────────────┘   └─────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Component Interaction Sequence

### Backtest Flow

```
User
  │
  ├─► CLI/API: run_backtest()
  │
  ├─► BacktestEngine
  │     │
  │     ├─► DataSource.get_ohlcv()
  │     │     │
  │     │     └─► Cache or Upbit API
  │     │
  │     ├─► Strategy.should_enter()
  │     │     │
  │     │     └─► Conditions.check()
  │     │
  │     ├─► Strategy.should_exit()
  │     │     │
  │     │     └─► Conditions.check()
  │     │
  │     ├─► Simulate trades (vectorized)
  │     │
  │     └─► Calculate metrics
  │
  └─► BacktestResult
        │
        └─► Generate report
```

### Live Trading Flow

```
Market Data
  │
  ├─► WebSocket/API
  │
  ├─► EventBus.publish("market_data")
  │
  ├─► SignalHandler
  │     │
  │     ├─► Strategy.should_enter()
  │     │
  │     └─► Strategy.should_exit()
  │
  ├─► OrderManager
  │     │
  │     ├─► Validate order
  │     │
  │     └─► Exchange.buy_market_order()
  │
  ├─► PositionManager
  │     │
  │     └─► Update positions
  │
  └─► Notifications (Telegram, etc.)
```

## Layer Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                           │
│  - CLI Commands                                                  │
│  - Python API                                                    │
│  - Jupyter Notebooks                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│  - Backtest Engine                                               │
│  - Trading Bot Facade                                            │
│  - Report Generator                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Domain Layer                                  │
│  - Strategy System                                                │
│  - Conditions & Filters                                          │
│  - Trading Logic                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  - Data Access (Cache, API)                                      │
│  - Exchange Integration                                           │
│  - Event Bus                                                     │
│  - Order/Position Management                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Design Patterns Used

1. **Facade Pattern**: `TradingBotFacade` simplifies complex system
2. **Strategy Pattern**: Different VBO strategies (Vanilla, Minimal, Strict)
3. **Observer Pattern**: EventBus for pub-sub communication
4. **Dependency Injection**: Constructor injection throughout
5. **Repository Pattern**: DataSource abstraction for data access
6. **Factory Pattern**: Strategy creation helpers
