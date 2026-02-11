# Strategy Library Guide

This guide explains the strategy interface and provides examples for building and testing strategies.

## Strategy Interface

All strategies inherit from the `Strategy` base class, which defines the contract for signal generation:

```python
from abc import abstractmethod
from src.strategies.base import Strategy, Signal, SignalType

class MyStrategy(Strategy):
    """Custom strategy implementation."""

    def generate_signals(self, ohlcv_data: pd.DataFrame, ticker: str) -> list[Signal]:
        """
        Generate buy/sell signals from OHLCV data.

        Args:
            ohlcv_data: DataFrame with columns [date, open, high, low, close, volume]
            ticker: Trading pair (e.g., "KRW-BTC")

        Returns:
            List of Signal objects with timing and strength
        """
        signals = []
        # Your signal generation logic here
        return signals
```

## Built-in Strategy: VBOV1

### Volatility Breakout V1 — Trend Following
**Type**: Momentum-based, breakout entry

**Principle**: Buy when price breaks above intraday volatility range with BTC MA filter; sell at open when trend reverses.

**Files**:
- Implementation: [src/strategies/volatility_breakout/vbo_v1.py](../../src/strategies/volatility_breakout/vbo_v1.py)

**Example Usage**:
```python
from src.strategies.volatility_breakout.vbo_v1 import VBOV1

strategy = VBOV1(
    name="VBOV1",
    ma_short=5,       # Short-term SMA for exit signal
    btc_ma=10,         # BTC market filter MA period
    data_dir=DATA_DIR,
    interval="day",
)
```

**Key Parameters**:
- `ma_short`: Short-term trend SMA for exit (default: 5)
- `btc_ma`: BTC market filter MA period (default: 10)
- `noise_ratio`: Fixed K value for breakout threshold (default: 0.5)

**Key Behaviors**:
- Fixed K=0.5 (no adaptive noise ratio)
- Exit at open price (`exit_price_base` convention)
- Exit signal uses `shift(1)` to prevent look-ahead bias
- BTC MA filter: only enters when BTC close > BTC MA

**Best For**: Trending crypto markets with clear breakout signals

---

## Creating a Custom Strategy

### Template

```python
from src.strategies.base import Strategy, Signal, SignalType
import pandas as pd

class MyCustomStrategy(Strategy):
    """My custom trading strategy."""

    def __init__(self, short_ma: int = 5, long_ma: int = 20):
        super().__init__(name="MyCustom")
        self.short_ma = short_ma
        self.long_ma = long_ma

    def generate_signals(self, ohlcv_data: pd.DataFrame, ticker: str) -> list[Signal]:
        """Generate signals from OHLCV data."""
        signals = []

        ohlcv_data['short_ma'] = ohlcv_data['close'].rolling(self.short_ma).mean()
        ohlcv_data['long_ma'] = ohlcv_data['close'].rolling(self.long_ma).mean()

        for idx, row in ohlcv_data.iterrows():
            if pd.isna(row['short_ma']) or pd.isna(row['long_ma']):
                continue

            if row['short_ma'] > row['long_ma']:
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    ticker=ticker,
                    price=row['close'],
                    date=row.name,
                    strength=0.8
                ))
            elif row['short_ma'] < row['long_ma']:
                signals.append(Signal(
                    signal_type=SignalType.SELL,
                    ticker=ticker,
                    price=row['close'],
                    date=row.name,
                    strength=0.8
                ))

        return signals
```

### Auto-Registration

When you create a `Strategy` subclass in `src/strategies/`, the `StrategyRegistry` automatically:
1. Discovers the class
2. Extracts `__init__` parameters (type, default, min/max)
3. Generates dashboard UI (sliders, inputs)

No manual registration needed.

### Testing Your Strategy

```python
from src.backtester.models import BacktestConfig
from src.backtester.engine.vectorized import VectorizedBacktestEngine

config = BacktestConfig(
    initial_capital=10_000_000,
    fee_rate=0.0005,
    max_slots=3,
)

strategy = MyCustomStrategy()
engine = VectorizedBacktestEngine(config)
result = engine.run(strategy, data_files)

print(f"CAGR: {result.cagr:.2f}%")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"MDD: {result.mdd:.2f}%")
```

---

## Best Practices

1. **Avoid Overfitting**: Use out-of-sample testing and Walk-Forward Analysis
2. **Account for Costs**: Include realistic fee rates and slippage
3. **Risk Management**: Set max drawdown and position limits
4. **Robustness**: Test across multiple tickers and timeframes
5. **Documentation**: Document all parameters and assumptions

---

## See Also

- [Architecture](../architecture.md) - System architecture
- [Backtester Modules](backtester_modules.md) - Engine details
