"""
HTML report generator for backtest results.

Generates interactive HTML reports with:
- Performance metrics tables
- Interactive charts (using Plotly or static images)
- Risk metrics
- Trade statistics
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester.report import BacktestReport, calculate_metrics, calculate_monthly_returns
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_html_report(
    report: BacktestReport,
    save_path: Path | str,
    strategy_obj=None,
    config=None,
    tickers: list[str] | None = None,
) -> None:
    """
    Generate HTML report from BacktestReport.

    Args:
        report: BacktestReport instance
        save_path: Path to save HTML file
        strategy_obj: Strategy instance for parameter extraction
        config: BacktestConfig instance
    """
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    m = report.metrics

    # Extract strategy parameters
    strategy_params_html = _extract_strategy_params(strategy_obj, tickers=tickers)
    
    # Extract backtest configuration
    config_html = _extract_config_params(config, result=report, tickers=tickers)

    # Prepare data for charts
    # Convert dates to ISO format strings for JavaScript
    dates_list = [d.isoformat() if isinstance(d, date) else str(d) for d in report.dates]
    dates_str = "[" + ", ".join([f"'{d}'" for d in dates_list]) + "]"
    equity_values = report.equity_curve.tolist()
    # Filter out NaN values for drawdown and ensure all values are valid
    drawdown_curve_clean = report.metrics.drawdown_curve.copy()
    drawdown_curve_clean = np.where(np.isnan(drawdown_curve_clean), 0.0, drawdown_curve_clean)
    drawdown_curve_clean = np.where(np.isinf(drawdown_curve_clean), 0.0, drawdown_curve_clean)
    drawdown_values = (-drawdown_curve_clean).tolist()
    # Ensure all values are finite numbers
    drawdown_values = [v if np.isfinite(v) else 0.0 for v in drawdown_values]
    
    # Find MDD index (where maximum drawdown occurs)
    mdd_idx = int(np.argmax(drawdown_curve_clean))
    mdd_date = dates_list[mdd_idx] if mdd_idx < len(dates_list) else dates_list[-1]
    # mdd_drawdown_value should be the actual drawdown value at MDD point (positive percentage)
    mdd_drawdown_value = drawdown_curve_clean[mdd_idx] if mdd_idx < len(drawdown_curve_clean) else report.metrics.mdd_pct

    # Monthly returns heatmap data
    monthly_returns = calculate_monthly_returns_for_html(report.equity_curve, report.dates)

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report: {report.strategy_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2em;
        }}
        .subtitle {{
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 1.5em;
            color: #34495e;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .chart-container {{
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #34495e;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .risk-section {{
            background: #fff3cd;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #ffc107;
        }}
        .config-section {{
            background: #e8f4f8;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
            margin-bottom: 30px;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .config-item {{
            background: white;
            padding: 10px;
            border-radius: 4px;
        }}
        .config-label {{
            font-size: 0.85em;
            color: #7f8c8d;
            margin-bottom: 3px;
        }}
        .config-value {{
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
        }}
        .yearly-returns-section {{
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .yearly-returns-section h3 {{
            margin-bottom: 15px;
            color: #34495e;
        }}
        .yearly-table {{
            width: 100%;
            max-width: 400px;
            border-collapse: collapse;
        }}
        .yearly-table th {{
            background-color: #34495e;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        .yearly-table td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .yearly-table tr:hover {{
            background-color: #e8f4f8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report: {report.strategy_name}</h1>
        <div class="subtitle">
            Period: {m.start_date} to {m.end_date} ({m.total_days:,} days)
        </div>

        <!-- Strategy Configuration (Top Section) -->
        <div class="section" style="margin-top: 20px;">
            <h2 class="section-title">Strategy Configuration</h2>
            {strategy_params_html}
            {config_html}
        </div>

        <!-- Performance Metrics -->
        <div class="section">
            <h2 class="section-title">Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if m.total_return_pct > 0 else 'negative'}">
                        {m.total_return_pct:,.2f}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">CAGR</div>
                    <div class="metric-value {'positive' if m.cagr_pct > 0 else 'negative'}">
                        {m.cagr_pct:.2f}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">
                        {m.mdd_pct:.2f}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">
                        {m.sharpe_ratio:.2f}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-value">
                        {m.sortino_ratio:.2f}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Calmar Ratio</div>
                    <div class="metric-value">
                        {m.calmar_ratio:.2f}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">
                        {m.total_trades:,}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">
                        {m.win_rate_pct:.2f}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">
                        {m.profit_factor:.2f}
                    </div>
                </div>
            </div>
        </div>

        <!-- Risk Metrics -->
        {_generate_risk_metrics_html(report.risk_metrics) if hasattr(report, 'risk_metrics') and report.risk_metrics else ''}

        <!-- Equity Curve Chart -->
        <div class="section">
            <h2 class="section-title">Equity Curve</h2>
            <div class="chart-container">
                <div id="equity-chart"></div>
            </div>
        </div>

        <!-- Drawdown Chart -->
        <div class="section">
            <h2 class="section-title">Drawdown</h2>
            <div class="chart-container">
                <div id="drawdown-chart"></div>
            </div>
        </div>

        <!-- Monthly Returns Heatmap -->
        <div class="section">
            <h2 class="section-title">Monthly Returns Heatmap</h2>
            <div class="chart-container">
                <div id="heatmap-chart"></div>
            </div>
        </div>

        <!-- Trade Statistics -->
        <div class="section">
            <h2 class="section-title">Trade Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Total Trades</td>
                        <td>{m.total_trades:,}</td>
                    </tr>
                    <tr>
                        <td>Winning Trades</td>
                        <td class="positive">{m.winning_trades:,}</td>
                    </tr>
                    <tr>
                        <td>Losing Trades</td>
                        <td class="negative">{m.losing_trades:,}</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td>{m.win_rate_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>{m.profit_factor:.2f}</td>
                    </tr>
                    <tr>
                        <td>Avg Profit</td>
                        <td class="positive">{m.avg_profit_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Avg Loss</td>
                        <td class="negative">{m.avg_loss_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Avg Trade</td>
                        <td>{m.avg_trade_pct:.2f}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Equity Curve Chart
        var equityTrace = {{
            x: {dates_str},
            y: {equity_values},
            type: 'scatter',
            mode: 'lines',
            name: 'Equity',
            line: {{color: '#3498db', width: 2}},
            fill: 'tozeroy',
            fillcolor: 'rgba(52, 152, 219, 0.2)'
        }};

        var drawdownTrace = {{
            x: {dates_str},
            y: {drawdown_values},
            type: 'scatter',
            mode: 'lines',
            name: 'Drawdown',
            yaxis: 'y2',
            line: {{color: '#e74c3c', width: 1}},
            fill: 'tozeroy',
            fillcolor: 'rgba(231, 76, 60, 0.3)'
        }};

        var filteredDrawdownForEquity = {drawdown_values}.filter(x => !isNaN(x) && isFinite(x));
        var maxDrawdown = filteredDrawdownForEquity.length > 0 ? Math.max(...filteredDrawdownForEquity) : 0;
        var equityLayout = {{
            title: 'Equity Curve with Drawdown',
            xaxis: {{title: 'Date'}},
            yaxis: {{
                title: 'Equity',
                side: 'left',
                type: 'log'
            }},
            yaxis2: {{
                title: 'Drawdown (%)',
                overlaying: 'y',
                side: 'right',
                range: [maxDrawdown * 1.2, 5]
            }},
            hovermode: 'x unified',
            showlegend: true
        }};

        Plotly.newPlot('equity-chart', [equityTrace, drawdownTrace], equityLayout, {{responsive: true}});

        // Drawdown Chart
        var drawdownOnlyTrace = {{
            x: {dates_str},
            y: {drawdown_values},
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(231, 76, 60, 0.5)',
            line: {{color: '#e74c3c', width: 1}}
        }};

        var filteredDrawdown = {drawdown_values}.filter(x => !isNaN(x) && isFinite(x));
        var maxDrawdown2 = filteredDrawdown.length > 0 ? Math.max(...filteredDrawdown) : 0;
        var mddValue = {report.metrics.mdd_pct:.2f};
        
        var drawdownLayout = {{
            title: 'Drawdown Over Time (MDD: ' + mddValue.toFixed(2) + '%)',
            xaxis: {{title: 'Date'}},
            yaxis: {{
                title: 'Drawdown (%)',
                range: [-100, 0]
            }},
            hovermode: 'x unified',
            showlegend: false,
            annotations: [{{
                x: '{mdd_date}',
                y: -{mdd_drawdown_value:.2f},
                text: 'MDD: ' + mddValue.toFixed(2) + '%',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1.5,
                arrowwidth: 2,
                arrowcolor: '#e74c3c',
                bgcolor: 'rgba(231, 76, 60, 0.8)',
                bordercolor: '#c0392b',
                borderwidth: 1,
                font: {{color: 'white', size: 12}},
                ax: 0,
                ay: 40,
                xanchor: 'center',
                yanchor: 'top',
                xshift: 0,
                yshift: 10
            }}]
        }};

        // Ensure drawdown chart div exists before plotting
        var drawdownChartDiv = document.getElementById('drawdown-chart');
        if (drawdownChartDiv) {{
            Plotly.newPlot('drawdown-chart', [drawdownOnlyTrace], drawdownLayout, {{responsive: true}});
        }}

        // Monthly Returns Heatmap
        var heatmapZ = {json.dumps(monthly_returns['values'])};
        var heatmapX = {json.dumps(monthly_returns['months'])};
        var heatmapY = {json.dumps(monthly_returns['years'])};
        var heatmapText = {json.dumps(monthly_returns['text'])};
        
        // Find min and max values for symmetric color scale
        var allValues = heatmapZ.flat().filter(x => !isNaN(x));
        var minVal = Math.min(...allValues);
        var maxVal = Math.max(...allValues);
        var absMax = Math.max(Math.abs(minVal), Math.abs(maxVal));
        
        var heatmapData = {{
            z: heatmapZ,
            x: heatmapX,
            y: heatmapY,
            type: 'heatmap',
            colorscale: [
                [0.0, '#8b0000'],      // Dark red for very negative (deepest)
                [0.2, '#c0392b'],     // Dark red
                [0.4, '#e74c3c'],     // Red
                [0.45, '#ec7063'],    // Light red
                [0.5, '#fadbd8'],     // Very light red/pink at zero
                [0.55, '#d5f4e6'],    // Very light green at zero
                [0.6, '#82e0aa'],     // Light green
                [0.8, '#27ae60'],     // Green
                [1.0, '#1e8449']      // Dark green for very positive (deepest)
            ],
            zmin: -absMax,  // Symmetric range
            zmax: absMax,   // Symmetric range
            zmid: 0,        // Center at 0
            colorbar: {{title: 'Return (%)'}},
            text: heatmapText,
            texttemplate: '%{{text}}',
            textfont: {{size: 11, color: '#000'}},
            textmode: 'text',
            hovertemplate: 'Year: %{{y}}<br>Month: %{{x}}<br>Return: %{{z:.2f}}%<extra></extra>',
            showscale: true
        }};

        var heatmapLayout = {{
            title: 'Monthly Returns Heatmap (%)',
            xaxis: {{title: 'Month'}},
            yaxis: {{title: 'Year'}},
            height: 400
        }};

        Plotly.newPlot('heatmap-chart', [heatmapData], heatmapLayout, {{responsive: true}});
        
        // Add yearly returns bar chart
        var yearlyReturns = {json.dumps(monthly_returns.get('yearly_returns', []))};
        var yearlyLabels = {json.dumps(monthly_returns.get('yearly_labels', []))};
        
        if (yearlyReturns && yearlyReturns.length > 0 && yearlyLabels && yearlyLabels.length > 0) {{
            // Extract years and values for bar chart
            var years = [];
            var values = [];
            var colors = [];
            
            for (var i = 0; i < yearlyLabels.length; i++) {{
                if (yearlyLabels[i] && !isNaN(yearlyReturns[i])) {{
                    var parts = yearlyLabels[i].split(': ');
                    if (parts.length >= 1) {{
                        years.push(parts[0]);
                        values.push(yearlyReturns[i]);
                        colors.push(yearlyReturns[i] >= 0 ? '#27ae60' : '#e74c3c');
                    }}
                }}
            }}
            
            if (years.length > 0) {{
                var yearlyBarData = {{
                    x: years,
                    y: values,
                    type: 'bar',
                    marker: {{color: colors}},
                    text: values.map(v => v.toFixed(1) + '%'),
                    textposition: 'outside',
                    textfont: {{size: 11, color: '#000'}}
                }};
                
                var yearlyBarLayout = {{
                    title: 'Yearly Returns (%)',
                    xaxis: {{title: 'Year'}},
                    yaxis: {{title: 'Return (%)'}},
                    height: 400,
                    showlegend: false
                }};
                
                // Create a new div for yearly returns chart
                var heatmapContainer = document.getElementById('heatmap-chart');
                if (heatmapContainer && heatmapContainer.parentElement) {{
                    var yearlyChartDiv = document.createElement('div');
                    yearlyChartDiv.id = 'yearly-returns-chart';
                    yearlyChartDiv.style.marginTop = '30px';
                    heatmapContainer.parentElement.appendChild(yearlyChartDiv);
                    Plotly.newPlot('yearly-returns-chart', [yearlyBarData], yearlyBarLayout, {{responsive: true}});
                }}
            }}
        }}
    </script>
</body>
</html>
"""

    save_path.write_text(html_content, encoding="utf-8")
    logger.info(f"HTML report saved to: {save_path}")


def _extract_strategy_params(strategy_obj, tickers: list[str] | None = None) -> str:
    """Extract strategy parameters from strategy instance."""
    if not strategy_obj:
        return ""
    
    params = []
    
    # Add tickers if provided
    if tickers:
        params.append(("Tickers", ", ".join(tickers)))
    
    # Common VBO parameters
    if hasattr(strategy_obj, "sma_period"):
        params.append(("SMA Period", str(strategy_obj.sma_period)))
    if hasattr(strategy_obj, "trend_sma_period"):
        params.append(("Trend SMA Period", str(strategy_obj.trend_sma_period)))
    if hasattr(strategy_obj, "short_noise_period"):
        params.append(("Short Noise Period", str(strategy_obj.short_noise_period)))
    if hasattr(strategy_obj, "long_noise_period"):
        params.append(("Long Noise Period", str(strategy_obj.long_noise_period)))
    if hasattr(strategy_obj, "exclude_current"):
        params.append(("Exclude Current", str(strategy_obj.exclude_current)))
    
    # Momentum strategy parameters
    if hasattr(strategy_obj, "lookback_period"):
        params.append(("Lookback Period", str(strategy_obj.lookback_period)))
    if hasattr(strategy_obj, "momentum_threshold"):
        params.append(("Momentum Threshold", f"{strategy_obj.momentum_threshold:.2f}"))
    
    # Mean reversion parameters
    if hasattr(strategy_obj, "zscore_threshold"):
        params.append(("Z-Score Threshold", f"{strategy_obj.zscore_threshold:.2f}"))
    if hasattr(strategy_obj, "lookback_window"):
        params.append(("Lookback Window", str(strategy_obj.lookback_window)))
    
    # Pair trading parameters
    if hasattr(strategy_obj, "spread_lookback"):
        params.append(("Spread Lookback", str(strategy_obj.spread_lookback)))
    if hasattr(strategy_obj, "entry_zscore"):
        params.append(("Entry Z-Score", f"{strategy_obj.entry_zscore:.2f}"))
    if hasattr(strategy_obj, "exit_zscore"):
        params.append(("Exit Z-Score", f"{strategy_obj.exit_zscore:.2f}"))
    
    # Entry/Exit conditions
    if hasattr(strategy_obj, "entry_conditions"):
        entry_names = [c.name for c in strategy_obj.entry_conditions.conditions]
        params.append(("Entry Conditions", ", ".join(entry_names) if entry_names else "None"))
    if hasattr(strategy_obj, "exit_conditions"):
        exit_names = [c.name for c in strategy_obj.exit_conditions.conditions]
        params.append(("Exit Conditions", ", ".join(exit_names) if exit_names else "None"))
    
    if not params:
        return "<div class='config-section'><p>No strategy parameters available.</p></div>"
    
    params_html = """
                <div class="config-grid">
    """
    
    for label, value in params:
        params_html += f"""
                    <div class="config-item">
                        <div class="config-label">{label}</div>
                        <div class="config-value">{value}</div>
                    </div>
        """
    
    params_html += """
                </div>
    """
    
    return params_html


def _extract_config_params(config, result=None, tickers: list[str] | None = None) -> str:
    """Extract backtest configuration parameters."""
    if not config:
        # Try to get from result
        if result and hasattr(result, "config") and result.config:
            config = result.config
        else:
            return ""
    
    params = []
    
    # Add tickers if provided (in config section as well for visibility)
    if tickers:
        params.append(("Tickers", ", ".join(tickers)))
    
    if hasattr(config, "initial_capital"):
        params.append(("Initial Capital", f"{config.initial_capital:,.0f}"))
    if hasattr(config, "fee_rate"):
        params.append(("Fee Rate", f"{config.fee_rate*100:.4f}%"))
    if hasattr(config, "slippage_rate"):
        params.append(("Slippage Rate", f"{config.slippage_rate*100:.4f}%"))
    if hasattr(config, "max_slots"):
        params.append(("Max Slots", str(config.max_slots)))
    if hasattr(config, "position_sizing"):
        params.append(("Position Sizing", str(config.position_sizing)))
    if hasattr(config, "use_cache"):
        params.append(("Use Cache", str(config.use_cache)))
    
    # Add interval from result if available
    if result and hasattr(result, "interval"):
        params.append(("Data Interval", str(result.interval)))
    
    if not params:
        return ""
    
    if not params:
        return "<div class='config-section'><p>No configuration parameters available.</p></div>"
    
    config_html = """
                <div class="config-grid">
    """
    
    for label, value in params:
        config_html += f"""
                    <div class="config-item">
                        <div class="config-label">{label}</div>
                        <div class="config-value">{value}</div>
                    </div>
        """
    
    config_html += """
                </div>
    """
    
    return config_html


def _generate_risk_metrics_html(risk_metrics) -> str:
    """Generate HTML for risk metrics section."""
    if not risk_metrics:
        return ""

    html = """
        <div class="section">
            <h2 class="section-title">Risk Metrics</h2>
            <div class="risk-section">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">VaR (95%)</div>
                        <div class="metric-value negative">
                            {:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">CVaR (95%)</div>
                        <div class="metric-value negative">
                            {:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">VaR (99%)</div>
                        <div class="metric-value negative">
                            {:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">CVaR (99%)</div>
                        <div class="metric-value negative">
                            {:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Portfolio Volatility</div>
                        <div class="metric-value">
                            {:.2f}%
                        </div>
                    </div>
    """.format(
        risk_metrics.var_95 * 100,
        risk_metrics.cvar_95 * 100,
        risk_metrics.var_99 * 100,
        risk_metrics.cvar_99 * 100,
        risk_metrics.portfolio_volatility * 100,
    )

    if risk_metrics.avg_correlation is not None and not np.isnan(risk_metrics.avg_correlation):
        html += """
                    <div class="metric-card">
                        <div class="metric-label">Avg Correlation</div>
                        <div class="metric-value">
                            {:.3f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Correlation</div>
                        <div class="metric-value">
                            {:.3f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Min Correlation</div>
                        <div class="metric-value">
                            {:.3f}
                        </div>
                    </div>
        """.format(
            risk_metrics.avg_correlation,
            risk_metrics.max_correlation,
            risk_metrics.min_correlation,
        )

    if risk_metrics.max_position_pct is not None and risk_metrics.max_position_pct > 0:
        html += """
                    <div class="metric-card">
                        <div class="metric-label">Max Position %</div>
                        <div class="metric-value">
                            {:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Position HHI</div>
                        <div class="metric-value">
                            {:.3f}
                        </div>
                    </div>
        """.format(
            risk_metrics.max_position_pct * 100,
            risk_metrics.position_concentration,
        )

    html += """
                </div>
            </div>
        </div>
    """

    return html


def calculate_monthly_returns_for_html(
    equity_curve: np.ndarray,
    dates: np.ndarray,
) -> dict:
    """Calculate monthly and yearly returns for heatmap in HTML format."""
    df = pd.DataFrame({"date": dates, "equity": equity_curve})
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Resample to monthly
    monthly = df["equity"].resample("ME").last()
    monthly_returns = monthly.pct_change() * 100

    # Resample to yearly
    yearly = df["equity"].resample("YE").last()
    yearly_returns = yearly.pct_change() * 100

    # Create monthly pivot table
    monthly_df = pd.DataFrame(
        {
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values,
        }
    )

    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ][: len(pivot.columns)]

    # Prepare monthly data for Plotly
    years = [str(y) for y in pivot.index.tolist()]
    months = pivot.columns.tolist()
    values = pivot.values.tolist()
    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in values]

    # Prepare yearly data
    yearly_data = []
    yearly_labels = []
    for year_idx, year in enumerate(pivot.index):
        year_return = yearly_returns[yearly_returns.index.year == year]
        if len(year_return) > 0 and not np.isnan(year_return.iloc[0]):
            yearly_data.append(year_return.iloc[0])
            yearly_labels.append(f"{year}: {year_return.iloc[0]:.1f}%")
        else:
            # Calculate from monthly returns if yearly not available
            year_monthly = monthly_returns[monthly_returns.index.year == year]
            if len(year_monthly) > 0:
                # Compound monthly returns
                year_total = (1 + year_monthly / 100).prod() - 1
                yearly_data.append(year_total * 100)
                yearly_labels.append(f"{year}: {year_total * 100:.1f}%")
            else:
                yearly_data.append(np.nan)
                yearly_labels.append("")

    return {
        "years": years,
        "months": months,
        "values": values,
        "text": text,
        "yearly_returns": yearly_data,
        "yearly_labels": yearly_labels,
    }
