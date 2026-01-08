"""
Stop-Loss / Take-Profit 구현 검증 스크립트

Note: Migrated from VanillaVBO to VanillaVBO with feature flags (2026-01-08)
"""

from pathlib import Path

import pandas as pd

from src.backtester import BacktestConfig, run_backtest
from src.strategies.volatility_breakout.vbo import VanillaVBO


def test_baseline():
    """기준 테스트: SL/TP 없음"""
    print("=" * 80)
    print("BASELINE: No Stop-Loss / No Take-Profit")
    print("=" * 80)

    strategy = VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        use_improved_noise=True,
        use_adaptive_k=True,
    )

    config = BacktestConfig(
        initial_capital=100.0,
        fee_rate=0.0005,
        slippage_rate=0.0005,
        max_slots=4,
        stop_loss_pct=None,
        take_profit_pct=None,
    )

    result = run_backtest(
        strategy=strategy,
        tickers=["KRW-BTC"],
        interval="day",
        config=config,
        data_dir=Path("c:/workspace/dev/crypto-quant-system/data/processed"),
    )

    trades_df = pd.DataFrame(
        [
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "pnl_pct": t.pnl_pct,
                "is_whipsaw": t.is_whipsaw,
                "is_stop_loss": t.is_stop_loss,
                "is_take_profit": t.is_take_profit,
                "exit_reason": t.exit_reason,
            }
            for t in result.trades
            if t.exit_date is not None
        ]
    )

    print(f"\n총 거래 수: {len(trades_df)}")
    print(f"승률: {(trades_df['pnl_pct'] > 0).mean() * 100:.2f}%")
    print(f"평균 수익률: {trades_df['pnl_pct'].mean():.2f}%")
    print(f"최대 손실: {trades_df['pnl_pct'].min():.2f}%")
    print(f"최대 이익: {trades_df['pnl_pct'].max():.2f}%")
    print(f"Whipsaw 거래: {trades_df['is_whipsaw'].sum()}")
    print(f"SL 발동: {trades_df['is_stop_loss'].sum()}")
    print(f"TP 발동: {trades_df['is_take_profit'].sum()}")
    print(f"\nExit Reason 분포:\n{trades_df['exit_reason'].value_counts()}")

    return trades_df


def test_with_sl_tp():
    """SL/TP 활성화 테스트"""
    print("\n" + "=" * 80)
    print("WITH SL/TP: stop_loss=5%, take_profit=15%")
    print("=" * 80)

    strategy = VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        use_improved_noise=True,
        use_adaptive_k=True,
    )

    config = BacktestConfig(
        initial_capital=100.0,
        fee_rate=0.0005,
        slippage_rate=0.0005,
        max_slots=4,
        stop_loss_pct=0.05,  # 5% 손절
        take_profit_pct=0.15,  # 15% 익절
    )

    result = run_backtest(
        strategy=strategy,
        tickers=["KRW-BTC"],
        interval="day",
        config=config,
        data_dir=Path("c:/workspace/dev/crypto-quant-system/data/processed"),
    )

    trades_df = pd.DataFrame(
        [
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "pnl_pct": t.pnl_pct,
                "is_whipsaw": t.is_whipsaw,
                "is_stop_loss": t.is_stop_loss,
                "is_take_profit": t.is_take_profit,
                "exit_reason": t.exit_reason,
            }
            for t in result.trades
            if t.exit_date is not None
        ]
    )

    print(f"\n총 거래 수: {len(trades_df)}")
    print(f"승률: {(trades_df['pnl_pct'] > 0).mean() * 100:.2f}%")
    print(f"평균 수익률: {trades_df['pnl_pct'].mean():.2f}%")
    print(f"최대 손실: {trades_df['pnl_pct'].min():.2f}%")
    print(f"최대 이익: {trades_df['pnl_pct'].max():.2f}%")
    print(f"Whipsaw 거래: {trades_df['is_whipsaw'].sum()}")
    print(f"SL 발동: {trades_df['is_stop_loss'].sum()}")
    print(f"TP 발동: {trades_df['is_take_profit'].sum()}")
    print(f"\nExit Reason 분포:\n{trades_df['exit_reason'].value_counts()}")

    return trades_df


def compare_results(baseline_df, sl_tp_df):
    """결과 비교"""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n거래 수 변화: {len(baseline_df)} → {len(sl_tp_df)}")
    print(
        f"승률 변화: {(baseline_df['pnl_pct'] > 0).mean() * 100:.1f}% → {(sl_tp_df['pnl_pct'] > 0).mean() * 100:.1f}%"
    )
    print(
        f"평균 수익률 변화: {baseline_df['pnl_pct'].mean():.2f}% → {sl_tp_df['pnl_pct'].mean():.2f}%"
    )
    print(f"최대 손실 개선: {baseline_df['pnl_pct'].min():.2f}% → {sl_tp_df['pnl_pct'].min():.2f}%")
    print(f"최대 이익 제한: {baseline_df['pnl_pct'].max():.2f}% → {sl_tp_df['pnl_pct'].max():.2f}%")


if __name__ == "__main__":
    baseline = test_baseline()
    sl_tp = test_with_sl_tp()
    compare_results(baseline, sl_tp)
