"""
Integration tests for complete trading pipeline.

Tests end-to-end workflows from data collection to report generation.
"""

from pathlib import Path

import pandas as pd

from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.backtester.models import BacktestConfig, BacktestResult
from src.backtester.report_pkg.report import BacktestReport
from src.strategies.volatility_breakout.vbo_v1 import VBOV1


class TestCompleteBacktestPipeline:
    """Integration tests for complete backtest pipeline."""

    def test_full_pipeline_data_to_report(
        self,
        sample_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test complete pipeline: data -> backtest -> report."""
        # Step 1: Prepare data
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        sample_ohlcv_data.to_parquet(filepath)

        # Step 2: Create strategy
        strategy = VBOV1(ma_short=5, btc_ma=20)

        # Step 3: Run backtest
        engine = VectorizedBacktestEngine(default_backtest_config)
        result = engine.run(strategy, {"KRW-BTC": filepath})

        # Step 4: Generate report using correct API
        report = BacktestReport(
            equity_curve=result.equity_curve,
            dates=result.dates,
            trades=result.trades,
            strategy_name="VBOV1",
            initial_capital=default_backtest_config.initial_capital,
        )

        # Assertions
        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert report is not None
        assert report.metrics is not None

    def test_pipeline_multiple_tickers(
        self,
        multiple_tickers_data: dict[str, pd.DataFrame],
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test pipeline with multiple tickers."""
        # Setup data files
        files = {}
        for ticker, df in multiple_tickers_data.items():
            filepath = temp_data_dir / f"{ticker}_day.parquet"
            df.to_parquet(filepath)
            files[ticker] = filepath

        # Run backtest
        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)
        result = engine.run(strategy, files)

        # Generate report using correct API
        report = BacktestReport(
            equity_curve=result.equity_curve,
            dates=result.dates,
            trades=result.trades,
            strategy_name="VBOV1",
            initial_capital=default_backtest_config.initial_capital,
        )

        assert isinstance(result, BacktestResult)
        assert report is not None
        assert report.metrics is not None

    def test_pipeline_with_date_range(
        self,
        large_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test pipeline with specific date range."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        large_ohlcv_data.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)

        # Get date range from data
        start_date = large_ohlcv_data.index[100].date()
        end_date = large_ohlcv_data.index[500].date()

        result = engine.run(
            strategy,
            {"KRW-BTC": filepath},
            start_date=start_date,
            end_date=end_date,
        )

        assert isinstance(result, BacktestResult)
        # Result should have equity curve
        assert len(result.equity_curve) > 0

    def test_pipeline_result_consistency(
        self,
        sample_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test that running the same backtest twice gives consistent results."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        sample_ohlcv_data.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)

        result1 = engine.run(strategy, {"KRW-BTC": filepath})
        result2 = engine.run(strategy, {"KRW-BTC": filepath})

        # Results should be identical
        assert result1.total_return == result2.total_return
        assert result1.total_trades == result2.total_trades
        assert result1.win_rate == result2.win_rate


class TestRiskAnalysisPipeline:
    """Integration tests for risk analysis pipeline."""

    def test_risk_metrics_calculation(
        self,
        sample_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test risk metrics are calculated correctly."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        sample_ohlcv_data.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)
        result = engine.run(strategy, {"KRW-BTC": filepath})

        # Verify risk metrics exist
        assert result.mdd is not None
        assert result.sharpe_ratio is not None
        assert result.total_return is not None

    def test_drawdown_analysis(
        self,
        downtrend_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test drawdown analysis in downtrending market."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        downtrend_ohlcv_data.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)
        result = engine.run(strategy, {"KRW-BTC": filepath})

        # MDD is positive percentage in this system (e.g., 20.5 means 20.5% drawdown)
        assert result.mdd is not None
        assert result.mdd >= 0  # MDD stored as positive percentage


class TestStrategyComparisonPipeline:
    """Integration tests for strategy comparison."""

    def test_compare_multiple_strategies(
        self,
        sample_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test comparing multiple strategy configurations on same data."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        sample_ohlcv_data.to_parquet(filepath)

        strategies = [
            VBOV1(ma_short=3, btc_ma=10),
            VBOV1(ma_short=5, btc_ma=20),
            VBOV1(ma_short=10, btc_ma=40),
        ]

        engine = VectorizedBacktestEngine(default_backtest_config)
        results = []

        for strategy in strategies:
            result = engine.run(strategy, {"KRW-BTC": filepath})
            results.append(result)

        # All results should be valid
        assert len(results) == 3
        for result in results:
            assert isinstance(result, BacktestResult)
            assert result.equity_curve is not None

    def test_strategy_parameter_sensitivity(
        self,
        sample_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test strategy parameter sensitivity analysis."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        sample_ohlcv_data.to_parquet(filepath)

        engine = VectorizedBacktestEngine(default_backtest_config)

        # Test different MA periods
        ma_periods = [3, 4, 5, 6]
        results = []

        for ma_period in ma_periods:
            strategy = VBOV1(ma_short=ma_period, btc_ma=ma_period * 4)
            result = engine.run(strategy, {"KRW-BTC": filepath})
            results.append(result)

        # All configurations should produce valid results
        assert len(results) == 4
        for result in results:
            assert isinstance(result, BacktestResult)


class TestDataQualityPipeline:
    """Integration tests for data quality handling."""

    def test_handle_missing_values(
        self,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test handling of data with missing values."""
        # Create data with some NaN values
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, None, 103.0, 104.0],
                "high": [102.0, 103.0, 104.0, None, 106.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "volume": [1000.0] * 5,
            },
            index=pd.date_range("2024-01-01", periods=5),
        )
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        df.to_parquet(filepath)

        strategy = VBOV1(ma_short=3, btc_ma=10)
        engine = VectorizedBacktestEngine(default_backtest_config)

        # Should handle gracefully (either run or raise specific error)
        try:
            result = engine.run(strategy, {"KRW-BTC": filepath})
            assert isinstance(result, BacktestResult)
        except (ValueError, KeyError):
            # Expected for data quality issues
            pass

    def test_handle_extreme_prices(
        self,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test handling of extreme price movements."""
        # Create data with extreme price movements
        df = pd.DataFrame(
            {
                "open": [100.0, 200.0, 50.0, 150.0, 75.0],  # 100% swings
                "high": [110.0, 220.0, 55.0, 165.0, 82.5],
                "low": [90.0, 180.0, 45.0, 135.0, 67.5],
                "close": [105.0, 190.0, 52.0, 155.0, 78.0],
                "volume": [1000.0] * 5,
            },
            index=pd.date_range("2024-01-01", periods=5),
        )
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        df.to_parquet(filepath)

        strategy = VBOV1(ma_short=3, btc_ma=10)
        engine = VectorizedBacktestEngine(default_backtest_config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)
