from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import (
    BacktestConfig,
    BacktestResult,
    VectorizedBacktestEngine,
    run_backtest,
)
from src.strategies.base import Strategy
from src.strategies.pair_trading import PairTradingStrategy

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> BacktestConfig:
    return BacktestConfig(
        initial_capital=10000.0,
        fee_rate=0.001,
        slippage_rate=0.001,
        max_slots=2,
        position_sizing="equal",
        use_cache=False,
    )


@pytest.fixture
def engine(mock_config: BacktestConfig) -> VectorizedBacktestEngine:
    return VectorizedBacktestEngine(config=mock_config)


@pytest.fixture
def mock_strategy() -> MagicMock:
    strategy = MagicMock(spec=Strategy)
    strategy.name = "TestStrategy"
    # Basic attributes needed for cache params
    strategy.entry_conditions = MagicMock()
    strategy.entry_conditions.conditions = []
    strategy.exit_conditions = MagicMock()
    strategy.exit_conditions.conditions = []
    strategy.required_indicators.return_value = []
    return strategy


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame."""
    # Use 100 periods to ensure enough history for lookbacks
    periods = 100
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="D")
    df = pd.DataFrame(
        {
            "open": np.full(periods, 100.0),
            "high": np.full(periods, 105.0),
            "low": np.full(periods, 95.0),
            "close": np.full(periods, 100.0),
            "volume": np.full(periods, 1000.0),
            "entry_signal": np.zeros(periods, dtype=bool),
            "exit_signal": np.zeros(periods, dtype=bool),
            # Add these to prevent NoneType errors in engine comparisons (smas[t_idx] can be NaN but not None)
            # engine checks: smas[t_idx, d_idx] if not np.isnan(...) else None.
            # If SMA column missing, array is all NaNs -> None -> crash on comparison.
            "target": np.full(periods, 100.0),
            "sma": np.full(periods, 100.0),
            "short_noise": np.full(periods, 0.5),
        },
        index=dates,
    )
    return df


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


class TestVectorizedBacktestEngine:
    def test_load_data_success(
        self, engine: VectorizedBacktestEngine, tmp_path: Path, sample_data: pd.DataFrame
    ) -> None:
        file_path = tmp_path / "test_data.parquet"
        sample_data.to_parquet(file_path)

        df = engine.load_data(file_path)
        assert len(df) == 100
        assert "close" in df.columns

    def test_load_data_file_not_found(self, engine: VectorizedBacktestEngine) -> None:
        with pytest.raises(FileNotFoundError):
            engine.load_data(Path("non_existent_file.parquet"))

    def test_add_price_columns(
        self, engine: VectorizedBacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        sample_data["entry_signal"] = True
        sample_data["exit_signal"] = True
        # Set specific target to verify calculation (VBO logic)
        sample_data["target"] = 100.5

        df = engine._add_price_columns(sample_data)

        assert df["is_whipsaw"].all()
        # VBO logic: target * (1 + slippage)
        # 100.5 * 1.001 = 100.6005
        expected_entry = 100.5 * (1 + 0.001)
        assert df["entry_price"].iloc[0] == pytest.approx(expected_entry)

        # Test without target (fallback to close logic)
        sample_data_no_target = sample_data.drop(columns=["target"])
        df2 = engine._add_price_columns(sample_data_no_target)
        # Close logic: close * (1 + slippage) -> 100.0 * 1.001
        expected_entry_close = 100.0 * (1 + 0.001)
        assert df2["entry_price"].iloc[0] == pytest.approx(expected_entry_close)

    def test_calculate_metrics_vectorized(self, engine: VectorizedBacktestEngine) -> None:
        dates = np.array([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)])
        equity_curve = np.array([10000.0, 11000.0, 9900.0])
        trades_df = pd.DataFrame()

        result = engine._calculate_metrics_vectorized(equity_curve, dates, trades_df)

        assert result.total_return == pytest.approx(-1.0)
        assert result.mdd == pytest.approx(10.0)

    def test_run_basic_flow(
        self,
        engine: VectorizedBacktestEngine,
        mock_strategy: MagicMock,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test standard backtest run with sufficient data."""
        fpath = tmp_path / "KRW-BTC_day.parquet"

        # Signal at index 30 (Date: 2023-01-31)
        # Exit at index 35
        sample_data.loc[sample_data.index[30], "entry_signal"] = True
        sample_data.loc[sample_data.index[35], "exit_signal"] = True

        # Explicit boolean conversion
        sample_data["entry_signal"] = sample_data["entry_signal"].astype(bool)
        sample_data["exit_signal"] = sample_data["exit_signal"].astype(bool)

        sample_data.to_parquet(fpath)
        data_files = {"KRW-BTC": fpath}

        mock_strategy.calculate_indicators.return_value = sample_data
        mock_strategy.generate_signals.return_value = sample_data

        # Patch optimize_dtypes to avoid issues with float32 vs float64 in tests
        with patch("src.backtester.engine.optimize_dtypes", side_effect=lambda x: x):
            result = engine.run(mock_strategy, data_files)

        assert isinstance(result, BacktestResult)
        assert result.total_trades == 1
        assert result.trades[0].ticker == "KRW-BTC"
        assert result.trades[0].is_closed
        assert result.trades[0].exit_date == sample_data.index[35].date()

    def test_run_whipsaw_logic(
        self,
        engine: VectorizedBacktestEngine,
        mock_strategy: MagicMock,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test whipsaw logic (entry and close < sma same day)."""
        fpath = tmp_path / "KRW-ETH_day.parquet"

        idx = 30
        sample_data.loc[sample_data.index[idx], "entry_signal"] = True
        sample_data.loc[sample_data.index[idx], "close"] = 100.0
        sample_data.loc[sample_data.index[idx], "sma"] = 9999.0  # SMA > Close -> Whipsaw

        sample_data.to_parquet(fpath)
        data_files = {"KRW-ETH": fpath}

        mock_strategy.calculate_indicators.return_value = sample_data
        mock_strategy.generate_signals.return_value = sample_data

        with patch("src.backtester.engine.optimize_dtypes", side_effect=lambda x: x):
            result = engine.run(mock_strategy, data_files)

        assert len(result.trades) == 1
        assert result.trades[0].is_whipsaw
        assert result.trades[0].entry_date == result.trades[0].exit_date

    @patch("src.backtester.engine.calculate_position_size")
    def test_position_sizing_integration(
        self,
        mock_calc_size: MagicMock,
        engine: VectorizedBacktestEngine,
        mock_strategy: MagicMock,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test position sizing integration for single asset."""
        engine.config.position_sizing = "volatility"

        fpath = tmp_path / "KRW-BTC_day.parquet"
        sample_data.loc[sample_data.index[30], "entry_signal"] = True
        sample_data.to_parquet(fpath)
        data_files = {"KRW-BTC": fpath}

        mock_strategy.calculate_indicators.return_value = sample_data
        mock_strategy.generate_signals.return_value = sample_data

        mock_calc_size.return_value = 5000.0

        with patch("src.backtester.engine.optimize_dtypes", side_effect=lambda x: x):
            engine.run(mock_strategy, data_files)

        mock_calc_size.assert_called()


# -------------------------------------------------------------------------
# Test BacktestResult & Metrics
# -------------------------------------------------------------------------


class TestBacktestResult:
    def test_summary_generation(self) -> None:
        """Test summary string generation."""
        result = BacktestResult(total_return=10.0, cagr=5.0, mdd=2.0, strategy_name="TestStrat")
        result.risk_metrics = MagicMock()
        result.risk_metrics.var_95 = 0.05
        result.risk_metrics.cvar_95 = 0.07
        result.risk_metrics.portfolio_volatility = 0.15
        result.equity_curve = np.array([100.0, 110.0])

        summary = result.summary()
        assert "TestStrat" in summary
        assert "CAGR: 5.00%" in summary
        assert "VaR (95%): 5.00%" in summary

    def test_calculate_metrics_edge_cases(self, engine: VectorizedBacktestEngine) -> None:
        """Test metrics calculation with insufficient data."""
        dates = np.array([date(2023, 1, 1)])
        equity = np.array([100.0])
        result = engine._calculate_metrics_vectorized(equity, dates, pd.DataFrame())
        assert result.total_return == 0.0

    def test_risk_metrics_failure_handling(self, engine: VectorizedBacktestEngine) -> None:
        """Test exception handling during risk metrics calculation."""
        dates = np.array([date(2023, 1, 1), date(2023, 1, 2)])
        equity = np.array([100.0, 110.0])

        with patch(
            "src.backtester.engine.calculate_portfolio_risk_metrics",
            side_effect=Exception("Risk error"),
        ):
            result = engine._calculate_metrics_vectorized(equity, dates, pd.DataFrame())
            assert result.risk_metrics is None

    def test_position_value_normalization(self, engine: VectorizedBacktestEngine) -> None:
        """Test position value normalization when value > equity."""
        dates = np.array([date(2023, 1, 1), date(2023, 1, 2)])
        equity = np.array([100.0, 50.0])

        trades_df = pd.DataFrame(
            [
                {
                    "ticker": "BTC",
                    "entry_date": date(2023, 1, 1),
                    "entry_price": 100.0,
                    "amount": 2.0,
                    "exit_date": None,
                    "exit_price": None,
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                    "is_whipsaw": False,
                }
            ]
        )

        with patch("src.backtester.engine.calculate_portfolio_risk_metrics") as mock_calc:
            engine._calculate_metrics_vectorized(equity, dates, trades_df)
            call_kwargs = mock_calc.call_args.kwargs
            pos_values = call_kwargs["position_values"]
            assert pos_values["BTC"] == pytest.approx(50.0)


# -------------------------------------------------------------------------
# Test Data Handling & Caching
# -------------------------------------------------------------------------


class TestEngineDataHandling:
    def test_load_data_generic_exception(
        self, engine: VectorizedBacktestEngine, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "bad_data.parquet"
        fpath.touch()
        with (
            patch("pandas.read_parquet", side_effect=Exception("Corrupt")),
            pytest.raises(ValueError, match="Error loading data"),
        ):
            engine.load_data(fpath)

    def test_caching_logic(
        self,
        engine: VectorizedBacktestEngine,
        mock_strategy: MagicMock,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        engine.config.use_cache = True
        fpath = tmp_path / "KRW-BTC_day.parquet"
        sample_data.to_parquet(fpath)
        data_files = {"KRW-BTC": fpath}

        mock_strategy.calculate_indicators.return_value = sample_data
        mock_strategy.generate_signals.return_value = sample_data

        with patch("src.backtester.engine.get_cache") as mock_get_cache:
            mock_cache_instance = MagicMock()
            mock_get_cache.return_value = mock_cache_instance

            # 1. Test Cache Miss -> Set
            mock_cache_instance.get.return_value = None
            engine.run(mock_strategy, data_files)
            mock_cache_instance.set.assert_called_once()

            # 2. Test Cache Hit
            mock_cache_instance.get.return_value = sample_data
            mock_cache_instance.set.reset_mock()
            engine.run(mock_strategy, data_files)
            mock_cache_instance.set.assert_not_called()


# -------------------------------------------------------------------------
# Test Advanced Trading Logic (Portfolio Opt, Noise)
# -------------------------------------------------------------------------


class TestEngineTradingLogic:
    def test_noise_sorting(
        self,
        engine: VectorizedBacktestEngine,
        mock_strategy: MagicMock,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test sorting candidates by short_noise."""
        idx = 30  # Use index > 20 to ensure valid data (past lookback/skips)
        sample_data.loc[sample_data.index[idx], "entry_signal"] = True
        sample_data["short_noise"] = 0.5

        df1 = sample_data.copy()
        df1["short_noise"] = 0.8  # Higher noise

        df2 = sample_data.copy()
        df2["short_noise"] = 0.2  # Lower noise -> Should be picked first

        fpath1 = tmp_path / "HIGH_NOISE_day.parquet"
        fpath2 = tmp_path / "LOW_NOISE_day.parquet"
        df1.to_parquet(fpath1)
        df2.to_parquet(fpath2)

        engine.config.max_slots = 1

        mock_strategy.calculate_indicators.side_effect = lambda x: x
        mock_strategy.generate_signals.side_effect = lambda x: x

        with patch("src.backtester.engine.optimize_dtypes", side_effect=lambda x: x):
            result = engine.run(mock_strategy, {"HIGH": fpath1, "LOW": fpath2})

        # Should pick LOW noise asset
        assert len(result.trades) == 1
        assert result.trades[0].ticker == "LOW"

    def test_portfolio_optimization_mpt(
        self,
        engine: VectorizedBacktestEngine,
        mock_strategy: MagicMock,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test MPT optimization branch."""
        engine.config.position_sizing = "mpt"
        engine.config.max_slots = 2

        # Signal late enough to have enough history (50 > 20 lookback)
        sample_data.loc[sample_data.index[50], "entry_signal"] = True

        fpath1 = tmp_path / "A_day.parquet"
        fpath2 = tmp_path / "B_day.parquet"
        sample_data.to_parquet(fpath1)
        sample_data.to_parquet(fpath2)

        mock_strategy.calculate_indicators.side_effect = lambda x: x
        mock_strategy.generate_signals.side_effect = lambda x: x

        with (
            patch("src.backtester.engine.optimize_portfolio") as mock_opt,
            patch("src.backtester.engine.optimize_dtypes", side_effect=lambda x: x),
        ):
            mock_weights = MagicMock()
            mock_weights.weights = {"A": 0.4, "B": 0.6}
            mock_opt.return_value = mock_weights

            result = engine.run(mock_strategy, {"A": fpath1, "B": fpath2})

            mock_opt.assert_called()
            assert len(result.trades) == 2

    def test_portfolio_optimization_kelly(
        self,
        engine: VectorizedBacktestEngine,
        mock_strategy: MagicMock,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test Kelly criterion fallback logic."""
        engine.config.position_sizing = "kelly"
        engine.config.max_slots = 2

        sample_data.loc[sample_data.index[50], "entry_signal"] = True
        fpath = tmp_path / "A_day.parquet"
        sample_data.to_parquet(fpath)

        mock_strategy.calculate_indicators.side_effect = lambda x: x
        mock_strategy.generate_signals.side_effect = lambda x: x

        with (
            patch("src.risk.portfolio_optimization.PortfolioOptimizer"),
            patch("src.backtester.engine.optimize_dtypes", side_effect=lambda x: x),
        ):
            # Verify that it runs without crashing and produces a trade (via fallback)
            result = engine.run(mock_strategy, {"A": fpath})
            assert len(result.trades) == 1

    def test_equity_curve_nan_fill(
        self,
        engine: VectorizedBacktestEngine,
        mock_strategy: MagicMock,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test equity curve forward fill on NaN."""
        sample_data.loc[sample_data.index[40], "close"] = np.nan
        sample_data.loc[sample_data.index[30], "entry_signal"] = True

        fpath = tmp_path / "GAP_day.parquet"
        sample_data.to_parquet(fpath)

        mock_strategy.calculate_indicators.side_effect = lambda x: x
        mock_strategy.generate_signals.side_effect = lambda x: x

        with patch("src.backtester.engine.optimize_dtypes", side_effect=lambda x: x):
            result = engine.run(mock_strategy, {"GAP": fpath})
        assert not np.isnan(result.equity_curve).any()


# -------------------------------------------------------------------------
# Test Pair Trading Edge Cases
# -------------------------------------------------------------------------


# Define dummy class at module level for correct patching behavior
class DummyPairStrategy(PairTradingStrategy):
    def __init__(self) -> None:
        super().__init__("PairStrat")
        self.lookback_period = 10

    def required_indicators(self) -> list[str]:
        return []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def calculate_spread_for_pair(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        m = pd.DataFrame(index=df1.index)
        m["entry_signal"] = False
        m["exit_signal"] = False
        # Create signals
        if len(m) > 30:
            m.iloc[30, m.columns.get_loc("entry_signal")] = True
        return m


class TestPairTradingExtended:
    def test_pair_trading_wrong_file_count(self, engine: VectorizedBacktestEngine) -> None:
        """Test pair trading validates 2 files."""
        # Since import is now inside run(), we need to patch before run() is called
        # But the isinstance check will still work since we're importing inside run()
        strat = DummyPairStrategy()
        with pytest.raises(ValueError, match="requires exactly 2 tickers"):
            engine.run(strat, {"A": Path("a")})

    def test_pair_trading_insufficient_slots(
        self, engine: VectorizedBacktestEngine, tmp_path: Path
    ) -> None:
        """Test pair trading logic when slots < 2."""
        strat = DummyPairStrategy()

        # Setup data (reuse standard layout)
        periods = 50
        dates = pd.date_range("2023-01-01", periods=periods)
        df = pd.DataFrame(
            {
                "close": np.full(periods, 100.0),
                "open": np.full(periods, 100.0),
                "high": np.full(periods, 105.0),
                "low": np.full(periods, 95.0),
                "volume": np.full(periods, 1000.0),
                "entry_signal": np.zeros(periods, dtype=bool),
                "exit_signal": np.zeros(periods, dtype=bool),
                "entry_price": np.full(periods, 100.0),
                "exit_price": np.full(periods, 100.0),
            },
            index=dates,
        )

        f1, f2 = tmp_path / "A.parquet", tmp_path / "B.parquet"
        df.to_parquet(f1)
        df.to_parquet(f2)

        engine.config.max_slots = 1

        with patch("src.backtester.engine.optimize_dtypes", side_effect=lambda x: x):
            result = engine.run(strat, {"A": f1, "B": f2})

        # Should enter NO trades because need at least 2 slots
        assert len(result.trades) == 0


# -------------------------------------------------------------------------
# Test run_backtest Convenience Function
# -------------------------------------------------------------------------


class TestConvenienceFunction:
    @patch("src.backtester.engine.VectorizedBacktestEngine")
    @patch("src.backtester.engine.DataCollectorFactory")
    def test_run_backtest_logic(
        self, mock_factory: MagicMock, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Test run_backtest wrapper logic."""

        # Setup Mocks
        mock_collector = MagicMock()
        mock_factory.create.return_value = mock_collector

        mock_engine_instance = MagicMock()
        mock_engine_cls.return_value = mock_engine_instance
        mock_engine_instance.run.return_value = BacktestResult()

        strategy = MagicMock(spec=Strategy)

        # Use side_effect to create file physically during collection
        # This satisfies the check `if v.exists()` later in the function
        def create_file_side_effect(ticker: str, *args: object, **kwargs: object) -> None:
            (tmp_path / "data").mkdir(exist_ok=True, parents=True)
            (tmp_path / "data" / f"{ticker}_day.parquet").touch()

        mock_collector.collect.side_effect = create_file_side_effect

        ticker = "MISSING_TICKER"
        data_dir = tmp_path / "data"

        run_backtest(strategy=strategy, tickers=[ticker], interval="day", data_dir=data_dir)

        mock_collector.collect.assert_called()
        mock_engine_instance.run.assert_called()

    @patch("src.backtester.engine.VectorizedBacktestEngine")
    def test_run_backtest_no_files_found(self, mock_engine: MagicMock, tmp_path: Path) -> None:
        """Test exception when data collection fails/no files found."""
        data_dir = tmp_path / "empty_data"
        with patch("src.backtester.engine.DataCollectorFactory"), pytest.raises(FileNotFoundError):
            run_backtest(strategy=MagicMock(), tickers=["GHOST"], data_dir=data_dir)
