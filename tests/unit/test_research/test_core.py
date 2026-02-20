from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.core.data import load_parquet_ohlcv_by_symbol
from src.research.core.metrics import (
    build_equal_weight_portfolio,
    compute_equity_trade_metrics,
    compute_yearly_return_and_sharpe,
)


def test_load_parquet_ohlcv_by_symbol_reads_interval_files(tmp_path: Path) -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1],
            "close": [1.1, 1.2, 1.3],
            "volume": [10.0, 20.0, 30.0],
        },
        index=index,
    )
    frame.to_parquet(tmp_path / "KRW-BTC_day.parquet")
    frame.to_parquet(tmp_path / "KRW-ETH_day.parquet")
    frame.to_parquet(tmp_path / "KRW-BTC_week.parquet")

    loaded = load_parquet_ohlcv_by_symbol(tmp_path, interval="day")

    assert sorted(loaded.keys()) == ["KRW-BTC", "KRW-ETH"]
    pd.testing.assert_frame_equal(loaded["KRW-BTC"], frame, check_freq=False)


def test_load_parquet_ohlcv_by_symbol_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_parquet_ohlcv_by_symbol(tmp_path, interval="day")


def test_compute_equity_trade_metrics_includes_trade_stats() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="D")
    equity = pd.Series([1.0, 1.1, 1.05, 1.2, 1.25], index=index)

    metrics = compute_equity_trade_metrics(
        equity=equity,
        trade_pnls=[0.10, -0.03, 0.08],
        trade_holding_days=[3, 2, 5],
    )

    assert metrics["NumTrades"] == 3.0
    assert metrics["WinRate"] == pytest.approx(2 / 3)
    assert metrics["PF"] > 1.0
    assert metrics["CAGR"] > 0.0


def test_equal_weight_portfolio_and_yearly_metrics() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    equity_a = pd.Series([1.0, 1.1, 1.2, 1.3], index=index)
    equity_b = pd.Series([1.0, 0.95, 1.0, 1.05], index=index)

    portfolio = build_equal_weight_portfolio({"A": equity_a, "B": equity_b})

    assert portfolio is not None
    assert len(portfolio.returns) == 3
    assert portfolio.equity.iloc[-1] > 0

    yearly_returns, yearly_sharpe = compute_yearly_return_and_sharpe(portfolio.returns)
    assert list(yearly_returns.index) == [2024]
    assert list(yearly_sharpe.index) == [2024]
