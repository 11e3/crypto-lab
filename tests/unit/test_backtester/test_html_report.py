"""
Unit tests for the HTML report generator.
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import BacktestConfig, Trade
from src.backtester.html_report import generate_html_report
from src.backtester.report import BacktestReport
from src.risk.metrics import PortfolioRiskMetrics


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Create a list of sample trades."""
    return [
        Trade(
            ticker="KRW-BTC",
            entry_date=datetime.date(2024, 1, 5),
            entry_price=50000.0,
            exit_date=datetime.date(2024, 1, 10),
            exit_price=51000.0,
            pnl=1000.0,
            pnl_pct=2.0,
            amount=1.0,
        ),
        Trade(
            ticker="KRW-ETH",
            entry_date=datetime.date(2024, 1, 8),
            entry_price=3000.0,
            exit_date=datetime.date(2024, 1, 12),
            exit_price=2900.0,
            pnl=-100.0,
            pnl_pct=-3.33,
            amount=1.0,
        ),
    ]


@pytest.fixture
def sample_backtest_report(sample_trades: list[Trade]) -> BacktestReport:
    """Create a sample BacktestReport instance."""
    dates = pd.to_datetime(pd.date_range("2024-01-01", periods=20, freq="D")).date
    equity_curve = np.linspace(1_000_000, 1_200_000, 20)
    report = BacktestReport(
        equity_curve=equity_curve,
        dates=dates,
        trades=sample_trades,
        strategy_name="Test Strategy",
        initial_capital=1_000_000,
    )
    # Add mock risk metrics
    report.risk_metrics = PortfolioRiskMetrics(
        var_95=0.01,
        cvar_95=0.015,
        var_99=0.02,
        cvar_99=0.025,
        portfolio_volatility=0.1,
        avg_correlation=0.5,
        max_correlation=0.8,
        min_correlation=0.2,
        max_position_pct=0.25,
        position_concentration=0.125,
    )
    return report


@pytest.fixture
def sample_backtest_config() -> BacktestConfig:
    """Create a sample BacktestConfig instance."""
    return BacktestConfig(
        initial_capital=1_000_000,
        fee_rate=0.001,
        slippage_rate=0.0005,
        max_slots=5,
        use_cache=True,
        position_sizing="equal",
    )


def test_generate_html_report(
    sample_backtest_report: BacktestReport,
    sample_backtest_config: BacktestConfig,
    tmp_path: Path,
) -> None:
    """Test generating a full HTML report."""
    save_path = tmp_path / "report.html"
    tickers = ["KRW-BTC", "KRW-ETH"]

    generate_html_report(
        report=sample_backtest_report,
        save_path=save_path,
        strategy_obj=None,  # Not testing strategy param extraction in detail here
        config=sample_backtest_config,
        tickers=tickers,
    )

    assert save_path.exists()
    html_content = save_path.read_text(encoding="utf-8")

    # Check for key sections and values
    assert "<!DOCTYPE html>" in html_content
    assert "<title>Backtest Report: Test Strategy</title>" in html_content
    assert "<h1>Backtest Report: Test Strategy</h1>" in html_content

    # Check for a metric
    assert "Total Return" in html_content
    # Total return is (1.2M / 1M - 1) * 100 = 20%
    assert "20.00%" in html_content

    # Check for config
    assert "Strategy Configuration" in html_content
    assert "Initial Capital" in html_content
    assert "1,000,000" in html_content
    assert "Tickers" in html_content
    assert "KRW-BTC, KRW-ETH" in html_content

    # Check for risk metrics
    assert "Risk Metrics" in html_content
    assert "VaR (95%)" in html_content
    assert "1.00%" in html_content

    # Check for chart placeholders
    assert '<div id="equity-chart"></div>' in html_content
    assert '<div id="drawdown-chart"></div>' in html_content
    assert '<div id="heatmap-chart"></div>' in html_content

    # Check for trade stats
    assert "Trade Statistics" in html_content
    assert "Total Trades" in html_content
    assert "<td>2</td>" in html_content
    assert "Winning Trades" in html_content
    assert '<td class="positive">1</td>' in html_content
    assert "Losing Trades" in html_content
    assert '<td class="negative">1</td>' in html_content


def test_generate_html_report_no_trades(
    sample_backtest_config: BacktestConfig, tmp_path: Path
) -> None:
    """Test generating a report with no trades."""
    dates = pd.to_datetime(pd.date_range("2024-01-01", periods=20, freq="D")).date
    equity_curve = np.linspace(1_000_000, 1_000_000, 20)  # Flat equity
    report = BacktestReport(
        equity_curve=equity_curve,
        dates=dates,
        trades=[],
        strategy_name="No Trade Strategy",
        initial_capital=1_000_000,
    )

    save_path = tmp_path / "no_trade_report.html"
    generate_html_report(report=report, save_path=save_path, config=sample_backtest_config)

    assert save_path.exists()
    html_content = save_path.read_text(encoding="utf-8")

    assert "<h1>Backtest Report: No Trade Strategy</h1>" in html_content
    assert "Total Trades" in html_content
    assert "<td>0</td>" in html_content
    assert "Win Rate" in html_content
    assert "0.00%" in html_content
