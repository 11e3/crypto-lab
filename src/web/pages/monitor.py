"""Bot Monitor page.

Real-time monitoring of crypto-bot via GCS logs.
Part of the Crypto Quant Ecosystem.

Features:
- Live positions display
- Trade history from GCS logs
- PnL summary (daily/weekly/monthly)
- Error alerts
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.data.storage import GCSStorage

logger = get_logger(__name__)

__all__ = ["render_monitor_page"]


def _check_gcs_availability() -> bool:
    """Check if GCS is available and show status."""
    try:
        from src.data.storage import is_gcs_available

        return is_gcs_available()
    except ImportError:
        return False


def _get_storage() -> GCSStorage | None:
    """Get GCS storage instance."""
    from src.data.storage import get_gcs_storage

    return get_gcs_storage()


def _render_gcs_not_configured() -> None:
    """Render message when GCS is not configured."""
    st.warning("GCS not configured. Set up GCS_BUCKET environment variable.")

    with st.expander("Setup Instructions", expanded=True):
        st.markdown(
            """
### GCS Configuration

1. **Create a GCS bucket** for your quant data:
   ```bash
   gsutil mb gs://your-quant-bucket
   ```

2. **Set up authentication**:
   ```bash
   # Option 1: Service account
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

   # Option 2: User credentials
   gcloud auth application-default login
   ```

3. **Configure environment**:
   ```bash
   # Add to .env file
   GCS_BUCKET=your-quant-bucket
   ```

4. **Install dependencies**:
   ```bash
   pip install google-cloud-storage
   ```

### Expected Bucket Structure

```
gs://your-quant-bucket/
├── logs/
│   └── {account}/
│       ├── trades_2025-01-16.csv
│       └── positions.json
├── models/
│   └── regime_classifier_v1.pkl
└── data/
    └── processed/
        └── *.parquet
```

### Demo Mode

You can use demo data by creating local files in `data/bot_logs/`:
```
data/bot_logs/
└── Main/
    ├── trades_2025-01-16.csv
    └── positions.json
```
            """
        )

    # Check for demo mode
    _render_demo_mode()


def _render_demo_mode() -> None:
    """Render demo mode with local data."""
    from pathlib import Path

    demo_dir = Path("data/bot_logs")

    if demo_dir.exists():
        st.info("Demo mode: Using local data from data/bot_logs/")
        # You could load local files here for demo


def _render_account_selector(accounts: list[str]) -> str:
    """Render account selector."""
    if not accounts:
        accounts = ["jh"]

    return st.selectbox(
        "Account",
        options=accounts,
        index=0,
        help="Select trading account to monitor",
    )


def _render_positions_card(positions: dict[str, Any]) -> None:
    """Render current positions card."""
    st.subheader("Current Positions")

    if not positions:
        st.info("No open positions")
        return

    # Convert to DataFrame for display
    if isinstance(positions, dict):
        if "positions" in positions:
            positions_data = positions["positions"]
        else:
            positions_data = [
                {"symbol": k, **v} if isinstance(v, dict) else {"symbol": k, "amount": v}
                for k, v in positions.items()
            ]
    else:
        positions_data = positions

    if not positions_data:
        st.info("No open positions")
        return

    df = pd.DataFrame(positions_data)

    # Style the dataframe
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )

    # Summary metrics
    if "unrealized_pnl" in df.columns:
        total_pnl = df["unrealized_pnl"].sum()
        pnl_color = "green" if total_pnl >= 0 else "red"
        st.markdown(f"**Total Unrealized PnL:** :{pnl_color}[{total_pnl:,.0f} KRW]")


def _render_trade_history(trades_df: pd.DataFrame) -> None:
    """Render trade history table."""
    st.subheader("Trade History")

    if trades_df.empty:
        st.info("No trades found for selected date")
        return

    # Format columns if they exist
    display_df = trades_df.copy()

    if "price" in display_df.columns:
        display_df["price"] = display_df["price"].apply(lambda x: f"{x:,.0f}")

    if "amount" in display_df.columns:
        display_df["amount"] = display_df["amount"].apply(lambda x: f"{x:.6f}")

    if "pnl" in display_df.columns:
        display_df["pnl"] = display_df["pnl"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )


def _calculate_pnl_summary(
    storage: GCSStorage,
    account: str,
    days: int = 30,
) -> pd.DataFrame:
    """Calculate PnL summary for the last N days."""
    summary_data = []

    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        try:
            trades_df = storage.get_bot_logs(date_str, account)

            if not trades_df.empty and "pnl" in trades_df.columns:
                daily_pnl = trades_df["pnl"].sum()
                trade_count = len(trades_df)
            else:
                daily_pnl = 0
                trade_count = 0

            summary_data.append(
                {
                    "date": date_str,
                    "pnl": daily_pnl,
                    "trades": trade_count,
                }
            )

        except Exception as e:
            logger.debug(f"No data for {date_str}: {e}")

    if not summary_data:
        return pd.DataFrame()

    return pd.DataFrame(summary_data)


def _render_pnl_summary(pnl_df: pd.DataFrame) -> None:
    """Render PnL summary charts and metrics."""
    st.subheader("PnL Summary")

    if pnl_df.empty:
        st.info("No PnL data available")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total_pnl = pnl_df["pnl"].sum()
    total_trades = pnl_df["trades"].sum()
    winning_days = (pnl_df["pnl"] > 0).sum()
    losing_days = (pnl_df["pnl"] < 0).sum()

    with col1:
        pnl_delta = f"{total_pnl:+,.0f}" if total_pnl != 0 else "0"
        st.metric("Total PnL", f"{total_pnl:,.0f} KRW", delta=pnl_delta)

    with col2:
        st.metric("Total Trades", f"{total_trades:,}")

    with col3:
        win_rate = (
            winning_days / (winning_days + losing_days) * 100
            if (winning_days + losing_days) > 0
            else 0
        )
        st.metric("Win Days", f"{winning_days}", delta=f"{win_rate:.1f}%")

    with col4:
        avg_daily = pnl_df["pnl"].mean() if not pnl_df.empty else 0
        st.metric("Avg Daily PnL", f"{avg_daily:,.0f} KRW")

    # PnL chart
    if not pnl_df.empty:
        pnl_df = pnl_df.sort_values("date")
        pnl_df["cumulative_pnl"] = pnl_df["pnl"].cumsum()

        # Cumulative PnL chart
        st.line_chart(
            pnl_df.set_index("date")["cumulative_pnl"],
            width="stretch",
        )


def _render_alerts(storage: GCSStorage, account: str) -> None:
    """Render alerts and notifications."""
    st.subheader("Alerts")

    # Check for recent errors or warnings
    # This would typically check a separate alerts/errors log

    # Placeholder alert examples
    alerts: list[dict[str, str]] = []

    # Check if positions file is stale
    try:
        positions = storage.get_bot_positions(account)
        if positions and "updated_at" in positions:
            last_update = datetime.fromisoformat(str(positions["updated_at"]))
            if (datetime.now() - last_update).total_seconds() > 3600:  # 1 hour
                alerts.append(
                    {
                        "type": "warning",
                        "message": f"Position data is stale (last update: {last_update})",
                    }
                )
    except Exception as e:
        logger.debug(f"Failed to check position staleness: {e}")

    if not alerts:
        st.success("No alerts")
    else:
        for alert in alerts:
            if alert["type"] == "error":
                st.error(alert["message"])
            elif alert["type"] == "warning":
                st.warning(alert["message"])
            else:
                st.info(alert["message"])


def render_monitor_page() -> None:
    """Render bot monitor page."""
    st.header("Bot Monitor")

    # Check GCS availability
    if not _check_gcs_availability():
        _render_gcs_not_configured()
        return

    storage = _get_storage()
    if storage is None:
        _render_gcs_not_configured()
        return

    # Monitor controls (inline, not sidebar)
    with st.expander("⚙️ Monitor Settings", expanded=True):
        accounts = storage.list_accounts()
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

        with ctrl_col1:
            account = _render_account_selector(accounts)

        with ctrl_col2:
            log_dates = storage.list_bot_log_dates(account, limit=30)
            if log_dates:
                selected_date = st.selectbox(
                    "Trade History Date",
                    options=log_dates,
                    index=0,
                    help="Select date to view trade history",
                )
            else:
                selected_date = datetime.now().strftime("%Y-%m-%d")

        with ctrl_col3:
            if st.button("🔄 Refresh Data", width="stretch"):
                st.cache_data.clear()
                st.rerun()

    # Main content
    col1, col2 = st.columns([1, 1])

    # Left column: Positions and Alerts
    with col1:
        try:
            positions = storage.get_bot_positions(account)
            _render_positions_card(positions)
        except Exception as e:
            st.error(f"Error loading positions: {e}")

        st.divider()

        _render_alerts(storage, account)

    # Right column: Trade History
    with col2:
        try:
            trades_df = storage.get_bot_logs(selected_date, account)
            _render_trade_history(trades_df)
        except Exception as e:
            st.error(f"Error loading trades: {e}")

    st.divider()

    # PnL Summary (full width)
    try:
        pnl_df = _calculate_pnl_summary(storage, account, days=30)
        _render_pnl_summary(pnl_df)
    except Exception as e:
        st.error(f"Error calculating PnL summary: {e}")


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Bot Monitor", layout="wide")
    render_monitor_page()
