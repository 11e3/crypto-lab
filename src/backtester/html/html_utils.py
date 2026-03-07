"""
HTML report utility functions.

Provides helper functions for HTML report generation including:
- Strategy parameter extraction
- Config parameter extraction
- Risk metrics HTML generation
- Monthly returns calculation
"""

from src.backtester.html.html_returns import calculate_monthly_returns_for_html
from src.backtester.html.html_risk import generate_risk_metrics_html
from src.backtester.report_pkg.report import BacktestReport


def extract_strategy_params(strategy_obj: object, tickers: list[str] | None = None) -> str:
    """Extract strategy parameters from strategy instance using parameter_schema()."""
    if not strategy_obj:
        return ""

    params: list[tuple[str, str]] = []

    # Add tickers if provided
    if tickers:
        params.append(("Tickers", ", ".join(tickers)))

    # Dynamically extract parameters from parameter_schema()
    if hasattr(strategy_obj, "parameter_schema"):
        schema = strategy_obj.parameter_schema()
        for name in schema:
            if hasattr(strategy_obj, name):
                value = getattr(strategy_obj, name)
                label = name.replace("_", " ").title()
                if isinstance(value, float):
                    params.append((label, f"{value:.4g}"))
                else:
                    params.append((label, str(value)))

    if not params:
        return ""

    rows = "\n".join(
        f'<tr><td class="param-name">{name}</td><td class="param-value">{value}</td></tr>'
        for name, value in params
    )

    return f"""
        <div class="section">
            <h2 class="section-title">Strategy Parameters</h2>
            <table class="params-table">
                <tbody>{rows}</tbody>
            </table>
        </div>
    """


def extract_config_params(
    config: object,
    result: BacktestReport | None = None,
    tickers: list[str] | None = None,
) -> str:
    """Extract backtest configuration parameters."""
    if not config:
        return ""

    params: list[tuple[str, str]] = []

    if hasattr(config, "initial_capital"):
        params.append(("Initial Capital", f"₩{config.initial_capital:,.0f}"))
    if hasattr(config, "fee_rate"):
        params.append(("Fee Rate", f"{config.fee_rate * 100:.3f}%"))
    if hasattr(config, "slippage_rate"):
        params.append(("Slippage Rate", f"{config.slippage_rate * 100:.3f}%"))
    if hasattr(config, "max_slots"):
        params.append(("Max Positions", str(config.max_slots)))
    if hasattr(config, "position_sizing"):
        params.append(("Position Sizing", str(config.position_sizing)))

    # Stop loss and take profit
    if hasattr(config, "stop_loss_pct") and config.stop_loss_pct is not None:
        params.append(("Stop Loss", f"{config.stop_loss_pct * 100:.1f}%"))
    if hasattr(config, "take_profit_pct") and config.take_profit_pct is not None:
        params.append(("Take Profit", f"{config.take_profit_pct * 100:.1f}%"))
    if hasattr(config, "trailing_stop_pct") and config.trailing_stop_pct is not None:
        params.append(("Trailing Stop", f"{config.trailing_stop_pct * 100:.1f}%"))

    # Date range from result if available
    if result and hasattr(result, "dates") and len(result.dates) > 0:
        params.append(("Start Date", str(result.dates[0])))
        params.append(("End Date", str(result.dates[-1])))
        params.append(("Total Days", str(len(result.dates))))

    # Tickers if provided
    if tickers:
        params.append(("Universe", f"{len(tickers)} tickers"))

    if not params:
        return ""

    rows = "\n".join(
        f'<tr><td class="param-name">{name}</td><td class="param-value">{value}</td></tr>'
        for name, value in params
    )

    return f"""
        <div class="section">
            <h2 class="section-title">Backtest Configuration</h2>
            <table class="params-table">
                <tbody>{rows}</tbody>
            </table>
        </div>
    """


# Re-export generate_risk_metrics_html from html_risk
__all__ = [
    "extract_strategy_params",
    "extract_config_params",
    "generate_risk_metrics_html",
    "calculate_monthly_returns_for_html",
]
