"""Tests for bt_backtest_runner service (native CQS engine)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.web.services.bt_backtest_runner import (
    BtBacktestResult,
    _convert_result,
    _create_strategy,
    _get_data_files,
    get_available_bt_symbols,
    get_default_model_path,
)


class TestGetDefaultModelPath:
    """Test model path resolution."""

    def test_returns_path(self) -> None:
        path = get_default_model_path()
        assert isinstance(path, Path)
        assert path.name.endswith(".joblib")


class TestGetAvailableBtSymbols:
    """Test symbol discovery."""

    @patch("src.web.services.bt_backtest_runner.DATA_DIR")
    def test_returns_sorted_symbols(self, mock_data_dir: MagicMock) -> None:
        mock_data_dir.exists.return_value = True
        mock_data_dir.glob.return_value = [
            Path("KRW-ETH_day.parquet"),
            Path("KRW-BTC_day.parquet"),
            Path("KRW-XRP_day.parquet"),
        ]
        symbols = get_available_bt_symbols("day")
        assert symbols == ["BTC", "ETH", "XRP"]

    @patch("src.web.services.bt_backtest_runner.DATA_DIR")
    def test_empty_when_no_data(self, mock_data_dir: MagicMock) -> None:
        mock_data_dir.exists.return_value = False
        symbols = get_available_bt_symbols()
        assert symbols == []


class TestGetDataFiles:
    """Test data file resolution."""

    @patch("src.web.services.bt_backtest_runner.DATA_DIR")
    def test_builds_correct_paths(self, mock_data_dir: MagicMock, tmp_path: Path) -> None:
        mock_data_dir.__truediv__ = lambda self, name: tmp_path / name

        # Create a mock parquet file
        (tmp_path / "KRW-BTC_day.parquet").touch()

        files = _get_data_files(["BTC"], "day")
        assert "KRW-BTC" in files
        assert files["KRW-BTC"] == tmp_path / "KRW-BTC_day.parquet"

    @patch("src.web.services.bt_backtest_runner.DATA_DIR")
    def test_skips_missing_symbols(self, mock_data_dir: MagicMock, tmp_path: Path) -> None:
        mock_data_dir.__truediv__ = lambda self, name: tmp_path / name

        files = _get_data_files(["NONEXISTENT"], "day")
        assert len(files) == 0


class TestBtBacktestResult:
    """Test BtBacktestResult dataclass."""

    def test_creation(self) -> None:
        result = BtBacktestResult(
            total_return=10.0,
            cagr=5.0,
            mdd=-15.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            win_rate=0.6,
            profit_factor=1.8,
            num_trades=50,
            avg_win_pct=3.0,
            avg_loss_pct=-2.0,
            final_equity=11_000_000.0,
            equity_curve=[10_000_000.0, 10_500_000.0, 11_000_000.0],
            dates=[datetime(2023, 1, 1), datetime(2023, 6, 1), datetime(2024, 1, 1)],
            yearly_returns={2023: 10.0},
            trades=[],
        )
        assert result.total_return == 10.0
        assert result.num_trades == 50
        assert len(result.equity_curve) == 3


class TestConvertResult:
    """Test BacktestResult to BtBacktestResult conversion."""

    def test_converts_basic_result(self) -> None:
        """Test conversion with minimal BacktestResult."""
        mock_result = MagicMock()
        mock_result.equity_curve = [10_000_000.0, 10_500_000.0, 11_000_000.0]
        mock_result.dates = np.array(
            ["2023-01-01", "2023-06-01", "2024-01-01"], dtype="datetime64[D]"
        )
        mock_result.trades = []
        mock_result.total_return = 10.0
        mock_result.cagr = 5.0
        mock_result.mdd = -15.0
        mock_result.sharpe_ratio = 1.5
        mock_result.win_rate = 0.6
        mock_result.profit_factor = 1.8
        mock_result.total_trades = 0

        bt_result = _convert_result(mock_result)

        assert isinstance(bt_result, BtBacktestResult)
        assert bt_result.total_return == 10.0
        assert bt_result.num_trades == 0
        assert bt_result.final_equity == 11_000_000.0

    def test_converts_result_with_trades(self) -> None:
        """Test conversion with trades."""
        mock_trade = MagicMock()
        mock_trade.pnl_pct = 5.0
        mock_trade.ticker = "KRW-BTC"
        mock_trade.entry_date = datetime(2023, 1, 1)
        mock_trade.exit_date = datetime(2023, 1, 5)
        mock_trade.entry_price = 50_000_000.0
        mock_trade.exit_price = 52_500_000.0
        mock_trade.amount = 1_000_000.0
        mock_trade.pnl = 50_000.0

        mock_result = MagicMock()
        mock_result.equity_curve = [10_000_000.0, 10_050_000.0]
        mock_result.dates = np.array(["2023-01-01", "2023-01-05"], dtype="datetime64[D]")
        mock_result.trades = [mock_trade]
        mock_result.total_return = 0.5
        mock_result.cagr = 1.0
        mock_result.mdd = -2.0
        mock_result.sharpe_ratio = 1.0
        mock_result.win_rate = 1.0
        mock_result.profit_factor = 999.0
        mock_result.total_trades = 1

        bt_result = _convert_result(mock_result)

        assert bt_result.num_trades == 1
        assert len(bt_result.trades) == 1
        assert bt_result.trades[0]["symbol"] == "KRW-BTC"
        assert bt_result.avg_win_pct == 5.0
        assert bt_result.avg_loss_pct == 0.0

    def test_converts_empty_equity(self) -> None:
        """Edge case: empty equity curve."""
        mock_result = MagicMock()
        mock_result.equity_curve = []
        mock_result.dates = None
        mock_result.trades = []
        mock_result.total_return = 0.0
        mock_result.cagr = 0.0
        mock_result.mdd = 0.0
        mock_result.sharpe_ratio = 0.0
        mock_result.win_rate = 0.0
        mock_result.profit_factor = 0.0
        mock_result.total_trades = 0

        bt_result = _convert_result(mock_result)
        assert bt_result.final_equity == 0.0
        assert bt_result.sortino_ratio == 0.0


class TestCreateStrategy:
    """Test strategy factory function."""

    def test_momentum_strategy(self) -> None:
        strategy = _create_strategy("momentum", lookback=14)
        assert strategy is not None
        assert strategy.name == "Momentum"

    def test_buy_and_hold(self) -> None:
        strategy = _create_strategy("buy_and_hold")
        assert strategy is not None
        assert strategy.name == "BuyAndHold"

    def test_vbo_portfolio(self) -> None:
        strategy = _create_strategy("vbo_portfolio", ma_short=5, btc_ma=20)
        assert strategy is not None
        assert strategy.name == "VBOPortfolio"

    def test_vbo_single_coin(self) -> None:
        strategy = _create_strategy("vbo_single_coin", ma_short=5, btc_ma=20)
        assert strategy is not None
        assert strategy.name == "VBOSingleCoin"

    def test_vbo_alias(self) -> None:
        strategy = _create_strategy("vbo", lookback=5, multiplier=2)
        assert strategy is not None
        assert strategy.name == "VBOPortfolio"

    def test_unknown_strategy(self) -> None:
        strategy = _create_strategy("nonexistent")
        assert strategy is None
