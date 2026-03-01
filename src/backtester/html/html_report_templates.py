"""
HTML templates for backtest report generation.

Contains CSS styles and HTML template fragments.
"""

from typing import Any

CSS_STYLES = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
        background-color: #f5f5f5; color: #333; line-height: 1.6; padding: 20px;
    }
    .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px;
        border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; margin-bottom: 10px; font-size: 2em; }
    .subtitle { color: #7f8c8d; margin-bottom: 30px; }
    .section { margin-bottom: 40px; }
    .section-title { font-size: 1.5em; color: #34495e; margin-bottom: 20px;
        padding-bottom: 10px; border-bottom: 2px solid #ecf0f1; }
    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px; margin-bottom: 30px; }
    .metric-card { background: #f8f9fa; padding: 15px; border-radius: 6px;
        border-left: 4px solid #3498db; }
    .metric-label { font-size: 0.9em; color: #7f8c8d; margin-bottom: 5px; }
    .metric-value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
    .chart-container { margin-bottom: 30px; background: white; padding: 20px;
        border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }
    th { background-color: #34495e; color: white; font-weight: 600; }
    tr:hover { background-color: #f8f9fa; }
    .positive { color: #27ae60; }
    .negative { color: #e74c3c; }
    .config-section { background: #e8f4f8; padding: 20px; border-radius: 6px;
        border-left: 4px solid #3498db; margin-bottom: 30px; }
    .config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
    .config-item { background: white; padding: 10px; border-radius: 4px; }
    .config-label { font-size: 0.85em; color: #7f8c8d; margin-bottom: 3px; }
    .config-value { font-size: 1.1em; font-weight: 600; color: #2c3e50; }
"""


def get_css_styles() -> str:
    """Get CSS styles for the HTML report."""
    return CSS_STYLES


def get_metrics_html(m: Any) -> str:
    """Generate metrics grid HTML."""
    pos_total = "positive" if m.total_return_pct > 0 else "negative"
    pos_cagr = "positive" if m.cagr_pct > 0 else "negative"
    return f"""<div class="metrics-grid">
        <div class="metric-card"><div class="metric-label">Total Return</div>
            <div class="metric-value {pos_total}">{m.total_return_pct:,.2f}%</div></div>
        <div class="metric-card"><div class="metric-label">CAGR</div>
            <div class="metric-value {pos_cagr}">{m.cagr_pct:.2f}%</div></div>
        <div class="metric-card"><div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">-{m.mdd_pct:.2f}%</div></div>
        <div class="metric-card"><div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{m.sharpe_ratio:.2f}</div></div>
        <div class="metric-card"><div class="metric-label">Sortino Ratio</div>
            <div class="metric-value">{m.sortino_ratio:.2f}</div></div>
        <div class="metric-card"><div class="metric-label">Calmar Ratio</div>
            <div class="metric-value">{m.calmar_ratio:.2f}</div></div>
        <div class="metric-card"><div class="metric-label">Total Trades</div>
            <div class="metric-value">{m.total_trades:,}</div></div>
        <div class="metric-card"><div class="metric-label">Win Rate</div>
            <div class="metric-value">{m.win_rate_pct:.2f}%</div></div>
        <div class="metric-card"><div class="metric-label">Profit Factor</div>
            <div class="metric-value">{m.profit_factor:.2f}</div></div>
    </div>"""


def get_trade_stats_html(m: Any) -> str:
    """Generate trade statistics table HTML."""
    return f"""<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
        <tr><td>Total Trades</td><td>{m.total_trades:,}</td></tr>
        <tr><td>Winning Trades</td><td class="positive">{m.winning_trades:,}</td></tr>
        <tr><td>Losing Trades</td><td class="negative">{m.losing_trades:,}</td></tr>
        <tr><td>Win Rate</td><td>{m.win_rate_pct:.2f}%</td></tr>
        <tr><td>Profit Factor</td><td>{m.profit_factor:.2f}</td></tr>
        <tr><td>Avg Profit</td><td class="positive">{m.avg_profit_pct:.2f}%</td></tr>
        <tr><td>Avg Loss</td><td class="negative">{m.avg_loss_pct:.2f}%</td></tr>
        <tr><td>Avg Trade</td><td>{m.avg_trade_pct:.2f}%</td></tr>
    </tbody></table>"""


__all__ = ["get_css_styles", "get_metrics_html", "get_trade_stats_html", "CSS_STYLES"]
