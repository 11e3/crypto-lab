import argparse
import json
import os

import pandas as pd

from src.monitoring import alerts
from src.monitoring.checks import evaluate_thresholds
from src.monitoring.metrics import compute_performance_from_trades, to_dict

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Risk
    "max_max_drawdown": -0.25,  # drawdown is negative number (e.g., -0.3)
    # Return/quality
    "min_cagr": 0.05,
    "min_sharpe": 0.40,
    "min_win_rate": 0.30,
}


def load_thresholds(cfg_path: str | None) -> dict[str, float]:
    if not cfg_path:
        return DEFAULT_THRESHOLDS.copy()
    if not os.path.exists(cfg_path):
        return DEFAULT_THRESHOLDS.copy()
    # Try YAML first, fallback to JSON
    try:
        import yaml  # type: ignore

        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return DEFAULT_THRESHOLDS.copy()

    thresholds = data.get("thresholds", {}) if isinstance(data, dict) else {}
    merged = DEFAULT_THRESHOLDS.copy()
    for k, v in thresholds.items():
        try:
            merged[str(k)] = float(v)
        except Exception:
            continue
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Monitoring Runner")
    parser.add_argument(
        "--trades", default=os.path.join("reports", "engine_trades.csv"), help="Path to trades CSV"
    )
    parser.add_argument(
        "--config",
        default=os.path.join("config", "monitoring.yaml"),
        help="Path to monitoring config (YAML or JSON)",
    )
    parser.add_argument(
        "--outdir", default="reports", help="Directory to write monitoring summaries and alerts log"
    )
    parser.add_argument(
        "--slack_webhook",
        default=os.environ.get("SLACK_WEBHOOK_URL", ""),
        help="Slack Incoming Webhook URL",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load trades
    trades_path = args.trades
    if not os.path.exists(trades_path):
        raise SystemExit(f"Trades file not found: {trades_path}")
    trades = pd.read_csv(trades_path)

    # Compute metrics
    perf = compute_performance_from_trades(trades)
    metrics_dict = to_dict(perf)

    # Save summary
    summary_json = os.path.join(args.outdir, "monitoring_summary.json")
    summary_txt = os.path.join(args.outdir, "monitoring_summary.txt")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    with open(summary_txt, "w", encoding="utf-8") as f:
        lines = [f"{k}: {v}" for k, v in metrics_dict.items()]
        f.write("\n".join(lines))

    # Load thresholds and evaluate
    thresholds = load_thresholds(args.config)
    issues = evaluate_thresholds(metrics_dict, thresholds)

    if issues:
        msg = alerts.format_issues(issues)
        alerts.to_console(msg)
        alerts.to_file(msg, os.path.join(args.outdir, "alerts.log"))
        if args.slack_webhook:
            alerts.to_slack(msg, args.slack_webhook)
        # Non-zero exit to indicate breach (optional policy)
        # raise SystemExit(2)
    else:
        print("Monitoring OK: no threshold breaches detected.")


if __name__ == "__main__":
    main()
