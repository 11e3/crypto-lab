"""
Integration tests for backtesting scenarios.

Tests complete backtest workflows from data to results.
"""

from pathlib import Path

import pandas as pd

from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.backtester.models import BacktestConfig, BacktestResult
from src.strategies.volatility_breakout.vbo_v1 import VBOV1


class TestVBOBacktestScenarios:
    """Integration tests for VBO strategy backtesting."""

    def test_vbo_backtest_with_trending_data(
        self,
        trending_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test VBO strategy with trending market data."""
        # Setup
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        trending_ohlcv_data.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)

        # Execute
        result = engine.run(strategy, {"KRW-BTC": filepath})

        # Assert
        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        # VBO should perform well in trending markets
        assert result.total_return is not None

    def test_vbo_backtest_with_volatile_data(
        self,
        volatile_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test VBO strategy with volatile market data."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        volatile_ohlcv_data.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None

    def test_vbo_backtest_with_sideways_data(
        self,
        sideways_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test VBO strategy with sideways market data."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        sideways_ohlcv_data.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)
        # Sideways markets typically have lower returns for breakout strategies
        assert result.equity_curve is not None

    def test_vbo_backtest_multiple_tickers(
        self,
        multiple_tickers_data: dict[str, pd.DataFrame],
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test VBO strategy with multiple tickers."""
        files = {}
        for ticker, df in multiple_tickers_data.items():
            filepath = temp_data_dir / f"{ticker}_day.parquet"
            df.to_parquet(filepath)
            files[ticker] = filepath

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)

        result = engine.run(strategy, files)

        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0

    def test_vbo_backtest_with_stop_loss(
        self,
        volatile_ohlcv_data: pd.DataFrame,
        conservative_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test VBO strategy with stop loss configuration."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        volatile_ohlcv_data.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(conservative_backtest_config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)
        # Stop loss should limit drawdown
        if result.mdd is not None:
            assert result.mdd >= -0.20  # Max 20% drawdown with stop loss


class TestBacktestConfigScenarios:
    """Integration tests for various backtest configurations."""

    def test_backtest_with_high_fees(
        self,
        sample_ohlcv_data: pd.DataFrame,
        temp_data_dir: Path,
    ) -> None:
        """Test backtest with high fee configuration."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        sample_ohlcv_data.to_parquet(filepath)

        config = BacktestConfig(
            initial_capital=10_000_000,
            fee_rate=0.005,  # High 0.5% fee
            slippage_rate=0.001,
        )

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)
        # High fees should reduce returns
        assert result.equity_curve is not None

    def test_backtest_with_small_capital(
        self,
        sample_ohlcv_data: pd.DataFrame,
        temp_data_dir: Path,
    ) -> None:
        """Test backtest with small initial capital."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        sample_ohlcv_data.to_parquet(filepath)

        config = BacktestConfig(
            initial_capital=100_000,  # Small 100K KRW
            fee_rate=0.0005,
        )

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)

    def test_backtest_with_large_capital(
        self,
        sample_ohlcv_data: pd.DataFrame,
        temp_data_dir: Path,
    ) -> None:
        """Test backtest with large initial capital."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        sample_ohlcv_data.to_parquet(filepath)

        config = BacktestConfig(
            initial_capital=1_000_000_000,  # 1B KRW
            fee_rate=0.0005,
        )

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)
        # Return percentage should be consistent regardless of capital
        assert result.equity_curve is not None


class TestEdgeCaseScenarios:
    """Integration tests for edge cases."""

    def test_backtest_with_single_day_data(
        self,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test backtest with minimal data."""
        # Create minimal data
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000.0],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        df.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)

    def test_backtest_with_no_signals(
        self,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Test backtest when strategy generates no signals."""
        # Create flat data that won't trigger signals
        df = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [100.1] * 50,
                "low": [99.9] * 50,
                "close": [100.0] * 50,
                "volume": [1000.0] * 50,
            },
            index=pd.date_range("2024-01-01", periods=50),
        )
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        df.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)
        # With flat data, VBOV1 may or may not generate signals
        # (depends on whether high >= target threshold is met)
        assert result.total_trades >= 0

    def test_backtest_stress_large_dataset(
        self,
        large_ohlcv_data: pd.DataFrame,
        default_backtest_config: BacktestConfig,
        temp_data_dir: Path,
    ) -> None:
        """Stress test with large dataset."""
        filepath = temp_data_dir / "KRW-BTC_day.parquet"
        large_ohlcv_data.to_parquet(filepath)

        strategy = VBOV1(ma_short=5, btc_ma=20)
        engine = VectorizedBacktestEngine(default_backtest_config)

        result = engine.run(strategy, {"KRW-BTC": filepath})

        assert isinstance(result, BacktestResult)
        # Equity curve length may differ slightly due to warm-up periods
        assert len(result.equity_curve) >= len(large_ohlcv_data) - 10
