"""
Unit tests for backtester report module.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import BacktestConfig, BacktestResult, Trade
from src.backtester.report_pkg.report import (
    BacktestReport,
    PerformanceMetrics,
    calculate_metrics,
    calculate_sortino_ratio,
    generate_report,
)


@pytest.fixture
def sample_backtest_result() -> BacktestResult:
    """Create sample BacktestResult for testing."""
    dates = np.array([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
    equity_curve = np.array([1000000.0, 1010000.0, 1020000.0])

    trades = [
        Trade(
            ticker="KRW-BTC",
            entry_date=date(2024, 1, 1),
            entry_price=50000.0,
            exit_date=date(2024, 1, 2),
            exit_price=51000.0,
            amount=0.01,
            pnl=10.0,
            pnl_pct=0.02,
        )
    ]

    return BacktestResult(
        total_return=0.02,
        cagr=0.10,
        mdd=0.01,
        calmar_ratio=10.0,
        sharpe_ratio=1.5,
        win_rate=0.6,
        profit_factor=1.5,
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        avg_trade_return=0.001,
        equity_curve=equity_curve,
        dates=dates,
        trades=trades,
        config=BacktestConfig(),
        strategy_name="TestStrategy",
    )


@pytest.fixture
def sample_performance_metrics(sample_backtest_result: BacktestResult) -> PerformanceMetrics:
    """Create sample PerformanceMetrics for testing."""
    import pandas as pd

    from src.backtester.report_pkg.report import calculate_metrics

    # Create trades DataFrame
    trades_df = pd.DataFrame(
        [
            {
                "ticker": "KRW-BTC",
                "entry_date": date(2024, 1, 1),
                "entry_price": 50000.0,
                "exit_date": date(2024, 1, 2),
                "exit_price": 51000.0,
                "amount": 0.01,
                "pnl": 10.0,
                "pnl_pct": 0.02,
            }
        ]
    )

    return calculate_metrics(
        equity_curve=sample_backtest_result.equity_curve,
        dates=sample_backtest_result.dates,
        trades_df=trades_df,
        initial_capital=1000000.0,
    )


class TestCalculateSortinoRatio:
    """Test cases for calculate_sortino_ratio function."""

    def test_calculate_sortino_ratio(self) -> None:
        """Test calculate_sortino_ratio with valid returns."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
        result = calculate_sortino_ratio(returns)

        assert isinstance(result, (int, float))
        assert result >= 0  # Sortino ratio should be non-negative for reasonable inputs

    def test_calculate_sortino_ratio_zero_downside_std(self) -> None:
        """Test calculate_sortino_ratio when downside_std is 0."""
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])  # All positive returns
        result = calculate_sortino_ratio(returns)

        assert result == 0.0

    def test_calculate_sortino_ratio_with_negative_returns(self) -> None:
        """Test calculate_sortino_ratio with negative returns to trigger mean_excess calculation."""
        returns = np.array([0.01, -0.02, 0.01, -0.01, 0.02])
        risk_free_rate = 0.001
        result = calculate_sortino_ratio(
            returns, risk_free_rate=risk_free_rate, annualization_factor=252
        )

        assert isinstance(result, (int, float))


class TestCalculateMetrics:
    """Test cases for calculate_metrics function."""

    def test_calculate_metrics_zero_cagr_case(self) -> None:
        """Test calculate_metrics when CAGR calculation should return 0.0 (line 126)."""
        equity_curve = np.array([1000000.0, 1010000.0])
        dates = np.array([date(2024, 1, 1), date(2024, 1, 1)])  # total_days = 0
        trades_df = pd.DataFrame()

        metrics = calculate_metrics(equity_curve, dates, trades_df, initial_capital=1000000.0)

        assert metrics.cagr_pct == 0.0

    def test_calculate_metrics_zero_sharpe_case(self) -> None:
        """Test calculate_metrics when Sharpe ratio should return 0.0 (line 146)."""
        equity_curve = np.array([1000000.0, 1000000.0, 1000000.0])  # No variance
        dates = np.array([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
        trades_df = pd.DataFrame()

        metrics = calculate_metrics(equity_curve, dates, trades_df, initial_capital=1000000.0)

        assert metrics.sharpe_ratio == 0.0

    def test_calculate_metrics_empty_trades_df(self) -> None:
        """Test calculate_metrics with empty trades_df (lines 177-183)."""
        equity_curve = np.array([1000000.0, 1010000.0, 1020000.0])
        dates = np.array([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
        trades_df = pd.DataFrame()  # Empty DataFrame

        metrics = calculate_metrics(equity_curve, dates, trades_df, initial_capital=1000000.0)

        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate_pct == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.avg_profit_pct == 0.0
        assert metrics.avg_loss_pct == 0.0
        assert metrics.avg_trade_pct == 0.0

    def test_calculate_metrics_no_closed_trades(self) -> None:
        """Test calculate_metrics with trades_df but no closed trades (lines 177-179)."""
        equity_curve = np.array([1000000.0, 1010000.0, 1020000.0])
        dates = np.array([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
        trades_df = pd.DataFrame(
            [
                {
                    "ticker": "KRW-BTC",
                    "entry_date": date(2024, 1, 1),
                    "exit_date": None,  # No exit date (not closed)
                    "pnl": 10.0,
                    "pnl_pct": 0.02,
                }
            ]
        )

        metrics = calculate_metrics(equity_curve, dates, trades_df, initial_capital=1000000.0)

        assert metrics.total_trades == 0  # No closed trades
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics dataclass."""

    def test_performance_metrics_structure(
        self, sample_performance_metrics: PerformanceMetrics
    ) -> None:
        """Test PerformanceMetrics structure."""
        assert hasattr(sample_performance_metrics, "total_return_pct")
        assert hasattr(sample_performance_metrics, "total_trades")
        assert hasattr(sample_performance_metrics, "win_rate_pct")

    def test_performance_metrics_custom(
        self, sample_performance_metrics: PerformanceMetrics
    ) -> None:
        """Test PerformanceMetrics with custom values."""
        assert sample_performance_metrics.total_trades >= 0
        assert isinstance(sample_performance_metrics.total_return_pct, (int, float))


class TestBacktestReport:
    """Test cases for BacktestReport class."""

    def test_initialization(self, sample_backtest_result: BacktestResult) -> None:
        """Test BacktestReport initialization."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )

        assert report.equity_curve is not None
        assert report.metrics is not None

    @patch("src.backtester.report_pkg.report.plt.savefig")
    @patch("src.backtester.report_pkg.report.plt.close")
    def test_generate_report(
        self,
        mock_close: MagicMock,
        mock_savefig: MagicMock,
        sample_backtest_result: BacktestResult,
        tmp_path: Path,
    ) -> None:
        """Test generating report."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )
        output_path = tmp_path / "backtest_report.png"

        report.plot_full_report(save_path=output_path, show=False)

        # Check that savefig was called
        assert mock_savefig.called

    @patch("src.backtester.report_pkg.report.plt.show")
    @patch("src.backtester.report_pkg.report.plt.savefig")
    @patch("src.backtester.report_pkg.report.plt.close")
    @patch("src.backtester.report_pkg.report.plt.subplots")
    @patch("src.backtester.report_pkg.report.plt.colorbar")
    def test_plot_full_report_show(
        self,
        mock_colorbar: MagicMock,
        mock_subplots: MagicMock,
        mock_close: MagicMock,
        mock_savefig: MagicMock,
        mock_show: MagicMock,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test plot_full_report with show=True (covers line 584)."""
        # Mock subplots to return a figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )

        report.plot_full_report(show=True)

        # Check that show was called (line 584)
        mock_show.assert_called_once()

    @patch("src.backtester.report_pkg.report.plt.tight_layout")
    @patch("src.backtester.report_pkg.report.plt.subplots")
    @patch("src.backtester.report_pkg.report.plt.colorbar")
    @patch("src.backtester.report_pkg.report.plt.imshow")
    def test_plot_monthly_heatmap_no_ax(
        self,
        mock_imshow: MagicMock,
        mock_colorbar: MagicMock,
        mock_subplots: MagicMock,
        mock_tight_layout: MagicMock,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test plot_monthly_heatmap with ax=None (covers lines 497, 533)."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_imshow.return_value = MagicMock()

        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )

        fig = report.plot_monthly_heatmap(ax=None)

        # Check that subplots was called (line 497)
        mock_subplots.assert_called_once_with(figsize=(14, 6))
        # Check that tight_layout was called (line 533)
        mock_tight_layout.assert_called_once()
        # Check that figure is returned
        assert fig is not None

    def test_generate_report_metrics(self, sample_backtest_result: BacktestResult) -> None:
        """Test that report calculates metrics correctly."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )

        assert report.metrics is not None
        # Report calculates metrics from trades list, not from BacktestResult fields
        assert report.metrics.total_trades == len(sample_backtest_result.trades)

    def test_initialization_with_dataframe_trades(
        self, sample_backtest_result: BacktestResult
    ) -> None:
        """Test BacktestReport initialization with DataFrame trades (line 319)."""
        trades_df = pd.DataFrame(
            [
                {
                    "ticker": "KRW-BTC",
                    "entry_date": date(2024, 1, 1),
                    "entry_price": 50000.0,
                    "exit_date": date(2024, 1, 2),
                    "exit_price": 51000.0,
                    "amount": 0.01,
                    "pnl": 10.0,
                    "pnl_pct": 0.02,
                    "is_whipsaw": False,
                }
            ]
        )
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=trades_df,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )
        assert report.trades_df is not None
        assert len(report.trades_df) == 1

    def test_initialization_with_empty_trades(self, sample_backtest_result: BacktestResult) -> None:
        """Test BacktestReport initialization with empty trades (line 338)."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=[],
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )
        assert report.trades_df is not None
        assert len(report.trades_df) == 0

    @patch("src.backtester.report_pkg.report_summary.logger.info")
    def test_print_summary(
        self, mock_logger_info: MagicMock, sample_backtest_result: BacktestResult
    ) -> None:
        """Test print_summary method (lines 347-383)."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )
        report.print_summary()

        # Verify logger.info was called multiple times (for different sections)
        assert mock_logger_info.called
        # Check that key sections are printed - check call args
        call_args_strs = [
            str(call[0][0]) if call[0] else "" for call in mock_logger_info.call_args_list
        ]
        all_calls_str = " ".join(call_args_strs)
        assert "BACKTEST REPORT" in all_calls_str or "BACKTEST" in all_calls_str

    def test_to_dataframe(self, sample_backtest_result: BacktestResult) -> None:
        """Test to_dataframe method (lines 631-632)."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )
        df = report.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Metric" in df.columns

    @patch("src.backtester.report_pkg.report.plt.show")
    def test_plot_equity_curve(
        self,
        mock_show: MagicMock,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test plotting equity curve."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )

        # plot_equity_curve should return a figure or None
        fig = report.plot_equity_curve(show_drawdown=False)
        # Just verify the method runs without error
        assert fig is None or hasattr(fig, "savefig")

    @patch("src.backtester.report_pkg.report.plt.tight_layout")
    @patch("src.backtester.report_pkg.report.plt.show")
    def test_plot_equity_curve_with_fig_tight_layout(
        self,
        mock_show: MagicMock,
        mock_tight_layout: MagicMock,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test plot_equity_curve calls tight_layout when fig is created (covers line 451)."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )

        # When ax is None, fig is created, so tight_layout should be called (line 451)
        fig = report.plot_equity_curve(show_drawdown=False)
        # Verify tight_layout was called when fig exists
        if fig is not None:
            mock_tight_layout.assert_called_once()

    @patch("src.backtester.report_pkg.report.plt.tight_layout")
    @patch("src.backtester.report_pkg.report.plt.show")
    def test_plot_drawdown_with_fig_tight_layout(
        self,
        mock_show: MagicMock,
        mock_tight_layout: MagicMock,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test plot_drawdown calls tight_layout when fig is created (covers line 495)."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )

        # When ax is None, fig is created, so tight_layout should be called (line 495)
        fig = report.plot_drawdown()
        # Verify tight_layout was called when fig exists
        if fig is not None:
            mock_tight_layout.assert_called_once()

    @patch("src.backtester.report_pkg.report.plt.show")
    def test_plot_drawdown(
        self,
        mock_show: MagicMock,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test plotting drawdown."""
        report = BacktestReport(
            equity_curve=sample_backtest_result.equity_curve,
            dates=sample_backtest_result.dates,
            trades=sample_backtest_result.trades,
            strategy_name=sample_backtest_result.strategy_name,
            initial_capital=sample_backtest_result.config.initial_capital
            if sample_backtest_result.config
            else 1.0,
        )

        # plot_drawdown should return a figure or None
        fig = report.plot_drawdown()
        # Just verify the method runs without error
        assert fig is None or hasattr(fig, "savefig")


class TestGenerateReport:
    """Test cases for generate_report convenience function."""

    @patch("src.backtester.report_pkg.report.BacktestReport.plot_full_report")
    @patch("src.backtester.report_pkg.report.BacktestReport.print_summary")
    def test_generate_report_function(
        self,
        mock_print: MagicMock,
        mock_plot: MagicMock,
        sample_backtest_result: BacktestResult,
        tmp_path: Path,
    ) -> None:
        """Test generate_report convenience function."""
        output_path = tmp_path / "report.png"

        report = generate_report(sample_backtest_result, save_path=output_path, show=False)

        assert report is not None
        mock_print.assert_called_once()
        mock_plot.assert_called_once()
