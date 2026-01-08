"""
Real-time monitoring with Upbit live data integration.

Features:
- Fetch latest OHLCV data from Upbit
- Run monitoring on fresh data
- Enhanced Slack alerts with trade metrics
- Automatic scheduling support
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.backtester import BacktestConfig, run_backtest  # noqa: E402
from src.data.collector_factory import DataCollectorFactory  # noqa: E402
from src.monitoring.alerts import format_issues, to_console, to_file  # noqa: E402
from src.strategies.volatility_breakout.vbo import VanillaVBO  # noqa: E402

logger = logging.getLogger(__name__)


class UpbitLiveMonitor:
    """Real-time monitoring with Upbit live data."""

    def __init__(self, config_path: Path | None = None, output_dir: Path | None = None):
        self.config_path = (
            config_path or Path(__file__).parent.parent / "config" / "monitoring.yaml"
        )
        self.output_dir = output_dir or Path(__file__).parent.parent / "reports"
        self.config = self._load_config()
        self.collector_factory = DataCollectorFactory()

    def _load_config(self) -> dict:
        """Load monitoring configuration."""
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {"thresholds": {"min_win_rate": 0.30, "min_sharpe": 0.5, "max_max_drawdown": -0.25}}

    def fetch_live_data(self, tickers: list[str], interval: str = "day") -> dict[str, pd.DataFrame]:
        """Fetch latest data from Upbit."""
        print(f"[{self._now()}] Fetching live data for {tickers}...")

        data = {}
        collector = self.collector_factory.create(exchange_name="upbit")

        for ticker in tickers:
            try:
                # First collect latest data
                count = collector.collect(ticker=ticker, interval=interval)  # type: ignore[arg-type]
                logger.info(f"Collected {count} new candles for {ticker}")

                # Then load the parquet file
                parquet_path = collector._get_parquet_path(ticker, interval)  # type: ignore[arg-type]
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)

                    # Ensure proper index
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date")
                    elif "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.rename(columns={"timestamp": "date"}).set_index("date")

                    data[ticker] = df.sort_index()
                    print(f"  {ticker}: {len(df)} candles (latest: {df.index[-1]})")
                else:
                    print(f"  {ticker}: No data file found")

            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                import traceback

                traceback.print_exc()

        return data

    def run_backtest(self, tickers: list[str]) -> dict[str, float]:
        """Run backtest on live data."""
        print(f"[{self._now()}] Running backtest...")

        strategy = VanillaVBO(
            sma_period=4,
            trend_sma_period=8,
            use_improved_noise=True,
            use_adaptive_k=True,
            min_hold_periods=3,
        )

        config = BacktestConfig(
            initial_capital=100.0,
            fee_rate=0.0005,
            slippage_rate=0.0005,
            max_slots=4,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
        )

        try:
            result = run_backtest(strategy=strategy, tickers=tickers, interval="day", config=config)

            return {
                "total_return": result.total_return,
                "sharpe_ratio": getattr(result, "sharpe_ratio", 0.0),
                "mdd": getattr(result, "mdd", 0.0),
                "total_trades": len(result.trades) if result.trades else 0,
                "winning_trades": sum(1 for t in result.trades if t.pnl > 0)
                if result.trades
                else 0,
                "win_rate": sum(1 for t in result.trades if t.pnl > 0) / len(result.trades)
                if result.trades
                else 0,
                "last_trade_date": 0.0,
                "total_commission": sum(getattr(t, "commission_cost", 0) for t in result.trades)
                if result.trades
                else 0,
                "total_slippage": sum(getattr(t, "slippage_cost", 0) for t in result.trades)
                if result.trades
                else 0,
            }
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {}

    def check_thresholds(self, metrics: dict) -> list[tuple]:
        """Check monitoring thresholds."""
        thresholds = self.config.get("thresholds", {})
        violations = []

        if "min_win_rate" in thresholds and metrics.get("win_rate", 0) < thresholds["min_win_rate"]:
            violations.append(("win_rate", metrics["win_rate"], thresholds["min_win_rate"]))

        if "min_sharpe" in thresholds and metrics.get("sharpe_ratio", 0) < thresholds["min_sharpe"]:
            violations.append(("sharpe_ratio", metrics["sharpe_ratio"], thresholds["min_sharpe"]))

        if (
            "max_max_drawdown" in thresholds
            and metrics.get("mdd", 0) < thresholds["max_max_drawdown"]
        ):
            violations.append(("max_drawdown", metrics["mdd"], thresholds["max_max_drawdown"]))

        return violations

    def format_slack_alert(
        self, metrics: dict[str, float], violations: list[tuple[str, float, float]]
    ) -> dict[str, object]:
        """Format comprehensive Slack alert."""
        timestamp = self._now()
        last_trade = metrics.get("last_trade_date", "N/A")

        # Build blocks for Slack message
        blocks = []

        # Header
        if violations:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Monitoring Alert - {timestamp}*\n{len(violations)} threshold violation(s) detected",
                    },
                }
            )
        else:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Monitoring Status - {timestamp}*\nAll thresholds OK",
                    },
                }
            )

        # Performance metrics
        metrics_text = f"""*Performance Metrics:*
- Return: {metrics.get("total_return", 0) * 100:.2f}%
- Sharpe: {metrics.get("sharpe_ratio", 0):.2f}
- MDD: {metrics.get("mdd", 0) * 100:.2f}%
- Win Rate: {metrics.get("win_rate", 0) * 100:.1f}%
- Trades: {metrics.get("total_trades", 0)} (Won: {metrics.get("winning_trades", 0)})
- Last Trade: {last_trade}
- Costs: Commission {metrics.get("total_commission", 0):.4f} + Slippage {metrics.get("total_slippage", 0):.4f}"""

        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": metrics_text}})

        # Violations (if any)
        if violations:
            violation_text = "*Violations:*\n"
            for key, val, threshold in violations:
                violation_text += f"- {key}: {val:.4f} (threshold: {threshold:.4f})\n"

            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": violation_text}})

        return {"blocks": blocks}

    def monitor(self, tickers: list[str], webhook_url: str | None = None) -> None:
        """Run complete monitoring cycle."""
        print("\n" + "=" * 80)
        print(f"Real-time Monitoring Started: {self._now()}")
        print("=" * 80)

        # 1. Fetch live data
        data = self.fetch_live_data(tickers)
        if not data:
            print("No data fetched, aborting monitoring")
            return

        # 2. Run backtest
        metrics = self.run_backtest(tickers)
        if not metrics:
            print("Backtest failed, aborting monitoring")
            return

        # 3. Check thresholds
        violations = self.check_thresholds(metrics)

        # 4. Generate alerts
        output_dir = self.output_dir or Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Console output
        to_console(f"Monitoring complete: {len(violations)} violations")
        if violations:
            print(format_issues(violations))
        print(f"\nMetrics: {json.dumps(metrics, indent=2, default=str)}")

        # File output
        alert_file = output_dir / "monitoring_alerts.log"
        if violations:
            to_file(format_issues(violations), str(alert_file))

        # Slack output
        if webhook_url:
            slack_message = self.format_slack_alert(metrics, violations)
            try:
                # Send Slack message with blocks
                import urllib.request as urlreq

                data_bytes = json.dumps(slack_message).encode("utf-8")
                req = urlreq.Request(
                    webhook_url, data=data_bytes, headers={"Content-Type": "application/json"}
                )
                with urlreq.urlopen(req, timeout=10) as resp:
                    logger.info(f"Slack alert sent (status: {resp.status})")
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

        # Save metrics
        metrics_file = output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"\n??Monitoring complete. Output: {output_dir}")

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time crypto strategy monitoring")
    parser.add_argument(
        "--tickers", nargs="+", default=["KRW-BTC", "KRW-ETH"], help="Tickers to monitor"
    )
    parser.add_argument("--config", type=str, help="Monitoring config YAML path")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--slack", type=str, help="Slack webhook URL")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    monitor = UpbitLiveMonitor(
        config_path=Path(args.config) if args.config else None,
        output_dir=Path(args.output) if args.output else None,
    )

    monitor.monitor(args.tickers, webhook_url=args.slack)

