"""Bootstrap 디버깅: 개별 샘플 검증

Note: Migrated from VanillaVBO_v2 to VanillaVBO with feature flags (2026-01-08)
"""

from pathlib import Path

import pandas as pd

from src.backtester.bootstrap_analysis import BootstrapAnalyzer
from src.backtester.engine import BacktestConfig
from src.strategies.volatility_breakout.vbo import VanillaVBO

data_path = Path("c:/workspace/dev/crypto-quant-system/data/raw/KRW-BTC_day.parquet")
df = pd.read_parquet(data_path)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

boot = BootstrapAnalyzer(
    data=df,
    strategy_factory=lambda: VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        use_improved_noise=True,
        use_adaptive_k=True,
        min_hold_periods=3,
    ),
    backtest_config=BacktestConfig(
        initial_capital=100.0,
        fee_rate=0.0005,
        slippage_rate=0.0005,
        max_slots=4,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
    ),
    ticker="KRW-BTC",
    interval="day",
)

print("원본 데이터:")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df.index[0]} ~ {df.index[-1]}")
print(f"  Close range: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

print("\nResampled 데이터 (3개 샘플):")
for i in range(3):
    resampled = boot._resample_data(df, block_size=30)
    print(f"\n  Sample {i + 1}:")
    print(f"    Shape: {resampled.shape}")
    print(f"    Date range: {resampled.index[0]} ~ {resampled.index[-1]}")
    print(f"    Close range: {resampled['close'].min():.2f} ~ {resampled['close'].max():.2f}")
    print(
        f"    OHLC valid: H>={resampled[['open', 'close']].max(axis=1).min():.0f}, L<={resampled[['open', 'close']].min(axis=1).max():.0f}"
    )

    # 전략 실행
    strategy = VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        use_improved_noise=True,
        use_adaptive_k=True,
        min_hold_periods=3,
    )
    result = boot._simple_backtest(resampled, strategy)
    print(f"    Return: {result.total_return * 100:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
