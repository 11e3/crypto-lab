"""bt library backtest execution service.

Service for running backtests using the external bt library.
Integrates bt library's VBO strategy with crypto-quant-system dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from bt.domain.models import PerformanceMetrics  # type: ignore[import-untyped]

logger = get_logger(__name__)

__all__ = [
    "BtBacktestResult",
    "run_bt_backtest_service",
    "is_bt_available",
    "get_available_bt_symbols",
]

# Data directory for crypto-quant-system
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


@dataclass
class BtBacktestResult:
    """Result container for bt library backtest."""

    # Performance metrics
    total_return: float
    cagr: float
    mdd: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_win: float
    avg_loss: float
    final_equity: float

    # Time series data
    equity_curve: list[float]
    dates: list[datetime]
    yearly_returns: dict[int, float]

    # Trade data
    trades: list[dict]


def is_bt_available() -> bool:
    """Check if bt library is installed and available."""
    try:
        from bt.engine.backtest import BacktestEngine  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def get_available_bt_symbols(interval: str = "day") -> list[str]:
    """Get list of symbols available for bt backtest.

    Args:
        interval: Time interval (default: "day")

    Returns:
        List of available symbol names (e.g., ["BTC", "ETH", "XRP"])
    """
    symbols = []
    if DATA_DIR.exists():
        for file in DATA_DIR.glob(f"KRW-*_{interval}.parquet"):
            # Extract symbol from KRW-{symbol}_{interval}.parquet
            symbol = file.stem.replace(f"_{interval}", "").replace("KRW-", "")
            symbols.append(symbol)
    return sorted(symbols)


def _load_data_for_bt(symbol: str, interval: str = "day") -> pd.DataFrame:
    """Load data from crypto-quant-system for bt library.

    Args:
        symbol: Trading symbol (e.g., "BTC")
        interval: Time interval

    Returns:
        DataFrame with OHLCV data

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    file_path = DATA_DIR / f"KRW-{symbol}_{interval}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_parquet(file_path)

    # Ensure datetime column
    if "datetime" not in df.columns:
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
        elif df.index.name == "datetime" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])

    return df


def _convert_bt_metrics_to_result(metrics: PerformanceMetrics) -> BtBacktestResult:
    """Convert bt library PerformanceMetrics to BtBacktestResult.

    Args:
        metrics: bt library PerformanceMetrics

    Returns:
        BtBacktestResult
    """
    # Convert trades to dict format
    trades_list = []
    for trade in metrics.trades:
        trades_list.append(
            {
                "symbol": trade.symbol,
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price),
                "quantity": float(trade.quantity),
                "pnl": float(trade.pnl),
                "return_pct": float(trade.return_pct),
            }
        )

    # Convert yearly returns
    yearly_returns_dict = {year: float(ret) for year, ret in metrics.yearly_returns.items()}

    return BtBacktestResult(
        total_return=float(metrics.total_return),
        cagr=float(metrics.cagr),
        mdd=float(metrics.mdd),
        sortino_ratio=float(metrics.sortino_ratio),
        win_rate=float(metrics.win_rate),
        profit_factor=float(metrics.profit_factor),
        num_trades=metrics.num_trades,
        avg_win=float(metrics.avg_win),
        avg_loss=float(metrics.avg_loss),
        final_equity=float(metrics.final_equity),
        equity_curve=[float(e) for e in metrics.equity_curve],
        dates=list(metrics.dates),
        yearly_returns=yearly_returns_dict,
        trades=trades_list,
    )


def run_bt_backtest_service(
    symbols: tuple[str, ...],
    interval: str = "day",
    initial_cash: int = 10_000_000,
    fee: float = 0.0005,
    slippage: float = 0.0005,
    multiplier: int = 2,
    lookback: int = 5,
) -> BtBacktestResult | None:
    """Run VBO backtest using bt library.

    Args:
        symbols: Tuple of symbols to trade (tuple for caching)
        interval: Time interval
        initial_cash: Initial capital in KRW
        fee: Trading fee (0.0005 = 0.05%)
        slippage: Slippage (0.0005 = 0.05%)
        multiplier: Multiplier for long-term indicators
        lookback: Lookback period for short-term indicators

    Returns:
        BtBacktestResult or None on failure
    """
    if not is_bt_available():
        logger.error("bt library is not installed")
        return None

    try:
        # Import bt library components
        from bt.framework.facade import BacktestFacade  # type: ignore[import-untyped]

        symbols_list = list(symbols)

        # Initialize facade
        facade = BacktestFacade()

        # Load data for all symbols into dict
        data: dict[str, pd.DataFrame] = {}
        for symbol in symbols_list:
            try:
                df = _load_data_for_bt(symbol, interval)
                data[symbol] = df
                logger.info(f"Loaded {symbol}: {len(df)} rows")
            except FileNotFoundError:
                logger.warning(f"Data not found for {symbol}")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")

        if not data:
            logger.error("No data loaded for any symbol")
            return None

        loaded_symbols = list(data.keys())

        # Run backtest with VBO strategy
        logger.info(f"Running bt backtest with {len(loaded_symbols)} symbols...")
        backtest_config = {
            "initial_cash": initial_cash,
            "fee": fee,
            "slippage": slippage,
            "lookback": lookback,
            "multiplier": multiplier,
        }
        results = facade.run_backtest(
            strategy="vbo",
            symbols=loaded_symbols,
            data=data,
            config=backtest_config,
        )

        # Extract metrics from results (key is "performance", not "metrics")
        metrics = results.get("performance")
        if not metrics:
            logger.error("No metrics in backtest results")
            return None

        logger.info(
            f"bt backtest completed: CAGR={float(metrics.cagr):.2f}%, "
            f"MDD={float(metrics.mdd):.2f}%, Trades={metrics.num_trades}"
        )

        return _convert_bt_metrics_to_result(metrics)

    except Exception as e:
        logger.exception(f"bt backtest failed: {e}")
        return None
