# Strategy Library Guide

This guide explains the strategy interface and provides examples for each strategy family.

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

## Strategy Families

### 1. Volatility Breakout (VBO) - Trend Following
**Type**: Momentum-based, breakout entry

**Principle**: Buy when price breaks above intraday volatility range; sell when it falls below.

**Files**:
- Implementation: [src/strategies/volatility_breakout/](../src/strategies/volatility_breakout/)
- Example: [examples/strategy_benchmark.py](strategy_benchmark.py)

**Example Usage**:
```python
from src.strategies.volatility_breakout import VanillaVBO, MinimalVBO, StrictVBO

# Standard configuration
strategy = VanillaVBO()

# Minimal configuration (fewer conditions)
strategy = MinimalVBO()

# Strict configuration (more filters)
strategy = StrictVBO()
```

**Key Parameters**:
- `sma_period`: Short-term trend SMA (e.g., 5 days)
- `trend_sma_period`: Long-term trend SMA (e.g., 10 days)
- `short_noise_period`: Intraday volatility period (e.g., 5 days)
- `long_noise_period`: Multi-day volatility period (e.g., 10 days)

**Best For**: Trending markets with clear breakout signals

---

### 2. Momentum - Acceleration-Based
**Type**: Momentum indicator, directional

**Principle**: Buy when price acceleration is strong; sell when momentum weakens.

**Files**:
- Implementation: [src/strategies/momentum/](../src/strategies/momentum/)
- Example: [examples/strategy_benchmark.py](strategy_benchmark.py)

**Example Usage**:
```python
from src.strategies.momentum import Momentum

strategy = Momentum()
signals = strategy.generate_signals(ohlcv_data, "KRW-BTC")
```

**Key Conditions**:
- **Momentum Signal**: ROC (Rate of Change) > threshold
- **Trend Filter**: SMA crossover (fast > slow)
- **Confirmation**: Increasing volume on breakout

**Best For**: Markets with strong directional moves and momentum continuation

---

### 3. Mean Reversion - Oscillator-Based
**Type**: Oscillator-based, reversal

**Principle**: Buy when price is oversold (low relative to recent range); sell when overbought.

**Files**:
- Implementation: [src/strategies/mean_reversion/](../src/strategies/mean_reversion/)
- Example: [examples/strategy_benchmark.py](strategy_benchmark.py)

**Example Usage**:
```python
from src.strategies.mean_reversion import MeanReversion

strategy = MeanReversion()
signals = strategy.generate_signals(ohlcv_data, "KRW-BTC")
```

**Key Conditions**:
- **RSI Oversold**: RSI < 30 (buy signal)
- **Bollinger Band Reversal**: Price < lower band
- **Z-Score Extreme**: Price > 2 std deviations from mean

**Best For**: Range-bound markets and counter-trend reversals

---

### 4. Pair Trading - Statistical Arbitrage
**Type**: Relative value, spread-based

**Principle**: Trade the spread between two correlated assets (e.g., BTC-ETH).

**Files**:
- Implementation: [src/strategies/pair_trading/](../src/strategies/pair_trading/)
- Example: [examples/strategy_benchmark.py](strategy_benchmark.py)

**Example Usage**:
```python
from src.strategies.pair_trading import PairTrading

# Note: Use 2 correlated tickers
config = BacktestConfig(tickers=["KRW-BTC", "KRW-ETH"], ...)
strategy = PairTrading()
signals = strategy.generate_signals(ohlcv_data, "KRW-BTC")
```

**Key Concepts**:
- **Cointegration**: Check if two assets move together long-term
- **Spread**: Log-price ratio between the pair
- **Z-Score**: Normalized spread for buy/sell signals

**Best For**: Hedged trading and market-neutral strategies

---

## Creating a Custom Strategy

### Template

```python
from src.strategies.base import Strategy, Signal, SignalType, Condition
import pandas as pd

class MyCustomStrategy(Strategy):
    """My custom trading strategy."""
    
    def __init__(self):
        super().__init__(name="MyCustom")
        # Initialize parameters
        self.short_ma = 5
        self.long_ma = 20
    
    def generate_signals(self, ohlcv_data: pd.DataFrame, ticker: str) -> list[Signal]:
        """Generate signals from OHLCV data."""
        signals = []
        
        # Add technical indicators
        ohlcv_data['short_ma'] = ohlcv_data['close'].rolling(self.short_ma).mean()
        ohlcv_data['long_ma'] = ohlcv_data['close'].rolling(self.long_ma).mean()
        
        # Generate signals
        for idx, row in ohlcv_data.iterrows():
            if pd.isna(row['short_ma']) or pd.isna(row['long_ma']):
                continue
            
            # Buy signal: short MA > long MA
            if row['short_ma'] > row['long_ma'] and (
                idx == 0 or ohlcv_data.iloc[idx-1]['short_ma'] <= ohlcv_data.iloc[idx-1]['long_ma']
            ):
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    ticker=ticker,
                    price=row['close'],
                    date=row.name,
                    strength=0.8
                ))
            
            # Sell signal: short MA < long MA
            elif row['short_ma'] < row['long_ma'] and (
                idx == 0 or ohlcv_data.iloc[idx-1]['short_ma'] >= ohlcv_data.iloc[idx-1]['long_ma']
            ):
                signals.append(Signal(
                    signal_type=SignalType.SELL,
                    ticker=ticker,
                    price=row['close'],
                    date=row.name,
                    strength=0.8
                ))
        
        return signals
```

### Testing Your Strategy

```python
from src.backtester import BacktestConfig, run_backtest

# Create config
config = BacktestConfig(
    tickers=["KRW-BTC"],
    interval="day",
    initial_capital=1000000.0,
    fee_rate=0.0005,
    max_slots=1
)

# Backtest
strategy = MyCustomStrategy()
result = run_backtest(config, strategy)

# Print results
print(f"Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
print(f"Max DD: {result.metrics.mdd_pct:.2f}%")
print(f"Win Rate: {result.metrics.win_rate_pct:.2f}%")
```

---

## Comparison Framework

Use [strategy_benchmark.py](strategy_benchmark.py) to compare multiple strategies:

```bash
python examples/strategy_benchmark.py
```

**Output includes**:
- Performance metrics (return, CAGR, Sharpe, Sortino, Calmar)
- Risk metrics (max drawdown, volatility, VaR)
- Trade statistics (win rate, profit factor)
- Ranking by Sharpe ratio

---

## Best Practices

1. **Avoid Overfitting**: Use out-of-sample testing
2. **Account for Costs**: Include realistic fee rates and slippage
3. **Risk Management**: Set max drawdown and position limits
4. **Robustness**: Test across multiple tickers and timeframes
5. **Documentation**: Document all parameters and assumptions

---

## See Also

- [examples/custom_strategy.py](custom_strategy.py) - Build a custom strategy from scratch
- [examples/performance_analysis.py](performance_analysis.py) - Detailed performance analysis
- [docs/architecture.md](../docs/architecture.md) - System architecture
