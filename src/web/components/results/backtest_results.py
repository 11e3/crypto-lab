"""Unified backtest results display component.

Normalizes BacktestResult and BtBacktestResult into a common format
and renders metrics, charts, and trade history.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from src.web.components.charts.equity_curve import render_equity_curve
from src.web.components.charts.underwater import render_underwater_curve
from src.web.components.charts.yearly_bar import render_yearly_bar_chart
from src.web.components.metrics.metrics_display import render_metrics_cards
from src.web.services.metrics_calculator import calculate_extended_metrics

if TYPE_CHECKING:
    from src.backtester.models import BacktestResult
    from src.web.services.bt_backtest_runner import BtBacktestResult

__all__ = ["UnifiedBacktestResult", "render_backtest_results"]


@dataclass
class UnifiedBacktestResult:
    """Adapter that normalizes BacktestResult and BtBacktestResult."""

    equity: np.ndarray
    dates: np.ndarray
    trade_returns: list[float]
    trades_df: pd.DataFrame
    title: str
    trade_count: int

    @classmethod
    def from_event_driven(cls, result: BacktestResult) -> UnifiedBacktestResult:
        """Create from event-driven BacktestResult."""
        equity = np.array(result.equity_curve)
        dates = (
            np.array(result.dates)
            if hasattr(result, "dates") and result.dates is not None and len(result.dates) > 0
            else np.arange(len(equity))
        )
        trade_returns = [t.pnl_pct / 100 for t in result.trades if t.pnl_pct is not None]

        trades_df = (
            pd.DataFrame(
                [
                    {
                        "Ticker": t.ticker,
                        "Entry Date": str(t.entry_date),
                        "Entry Price": f"{t.entry_price:,.0f}",
                        "Exit Date": str(t.exit_date) if t.exit_date else "-",
                        "Exit Price": f"{t.exit_price:,.0f}" if t.exit_price else "-",
                        "P&L": f"{t.pnl:,.0f}",
                        "P&L %": f"{t.pnl_pct:.2f}%",
                    }
                    for t in result.trades
                ]
            )
            if result.trades
            else pd.DataFrame()
        )

        return cls(
            equity=equity,
            dates=dates,
            trade_returns=trade_returns,
            trades_df=trades_df,
            title="Backtest Results",
            trade_count=len(result.trades),
        )

    @classmethod
    def from_bt(
        cls, result: BtBacktestResult, strategy_name: str = "bt_VBO"
    ) -> UnifiedBacktestResult:
        """Create from bt library BtBacktestResult."""
        equity = np.array(result.equity_curve)
        dates = (
            np.array(result.dates)
            if result.dates is not None and len(result.dates) > 0
            else np.arange(len(equity))
        )
        trade_returns = [float(t["return_pct"]) / 100 for t in result.trades if t.get("return_pct")]

        if result.trades:
            trades_df = pd.DataFrame(result.trades)
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.strftime(
                "%Y-%m-%d"
            )
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.strftime("%Y-%m-%d")
            trades_df["entry_price"] = trades_df["entry_price"].apply(lambda x: f"{x:,.0f}")
            trades_df["exit_price"] = trades_df["exit_price"].apply(lambda x: f"{x:,.0f}")
            trades_df["pnl"] = trades_df["pnl"].apply(lambda x: f"{x:,.0f}")
            trades_df["return_pct"] = trades_df["return_pct"].apply(lambda x: f"{x:.2f}%")
            trades_df = trades_df.rename(
                columns={
                    "symbol": "Ticker",
                    "entry_date": "Entry Date",
                    "exit_date": "Exit Date",
                    "entry_price": "Entry Price",
                    "exit_price": "Exit Price",
                    "pnl": "P&L",
                    "return_pct": "P&L %",
                }
            )
        else:
            trades_df = pd.DataFrame()

        display_name = "bt VBO Regime" if strategy_name == "bt_VBO_Regime" else "bt VBO"

        return cls(
            equity=equity,
            dates=dates,
            trade_returns=trade_returns,
            trades_df=trades_df,
            title=f"{display_name} Backtest Results",
            trade_count=len(result.trades),
        )


def render_backtest_results(result: UnifiedBacktestResult) -> None:
    """Render unified backtest results with metrics, charts, and trade history."""
    st.subheader(f"📊 {result.title}")

    # Calculate extended metrics (cached in session state)
    cache_key = f"metrics_{hashlib.md5(result.equity.tobytes()).hexdigest()}"

    # Limit cached metrics to prevent unbounded memory growth
    metrics_keys = [
        k for k in st.session_state if k.startswith("metrics_") or k.startswith("bt_metrics_")
    ]
    if len(metrics_keys) > 20:
        del st.session_state[metrics_keys[0]]

    if cache_key not in st.session_state:
        st.session_state[cache_key] = calculate_extended_metrics(
            equity=result.equity,
            trade_returns=result.trade_returns,
            dates=result.dates,
        )

    extended_metrics = st.session_state[cache_key]

    # Metrics cards
    render_metrics_cards(extended_metrics)

    st.divider()

    # All results on a single scrollable page
    render_equity_curve(result.dates, result.equity)
    render_underwater_curve(result.dates, result.equity)

    st.divider()

    render_yearly_bar_chart(result.dates, result.equity)

    st.divider()

    _render_trade_history(result)


def _render_trade_history(result: UnifiedBacktestResult) -> None:
    """Render trade history tab."""
    if result.trades_df.empty:
        st.info("No trades executed.")
        return

    st.markdown(f"### Trade History ({result.trade_count:,} trades)")

    show_count = st.selectbox(
        "Show trades", options=[10, 25, 50, 100, "All"], index=1, key="unified_trade_count"
    )
    display_df = (
        result.trades_df if show_count == "All" else result.trades_df.tail(int(str(show_count)))
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
