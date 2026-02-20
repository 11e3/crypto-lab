"""Backtest execution service for VBO strategies.

Runs VBO strategies using the native crypto-lab vectorized engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.backtester.models import BacktestConfig, BacktestResult
from src.config import parquet_filename
from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.utils.metrics_core import calculate_sortino_ratio

logger = get_logger(__name__)

__all__ = [
    "VboBacktestResult",
    "run_vbo_backtest_service",
    "run_vbo_backtest_generic_service",
    "get_available_symbols",
    "get_default_model_path",
]

# Data directory for crypto-lab
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
BINANCE_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "binance"

# Models directory
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

# Default regime model
DEFAULT_MODEL_NAME = "regime_classifier_xgb_ultra5.joblib"


def get_default_model_path() -> Path:
    """Get the default regime model path."""
    return MODELS_DIR / DEFAULT_MODEL_NAME


@dataclass
class VboBacktestResult:
    """Result container for VBO strategy backtests."""

    # Performance metrics
    total_return: float
    cagr: float
    mdd: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    final_equity: float

    # Time series data
    equity_curve: list[float]
    dates: list[datetime]
    yearly_returns: dict[int, float]

    # Trade data
    trades: list[dict[str, Any]]


def get_available_symbols(interval: str = "day", exchange: str = "upbit") -> list[str]:
    """Get list of symbols available for backtest.

    Args:
        interval: Time interval (default: "day")
        exchange: Exchange name ("upbit" or "binance")

    Returns:
        List of available symbol names (e.g., ["BTC", "ETH", "XRP"] for Upbit,
        ["BTCUSDT", "ETHUSDT"] for Binance)
    """
    symbols = []
    if exchange == "binance":
        if BINANCE_DATA_DIR.exists():
            for file in BINANCE_DATA_DIR.glob(f"*_{interval}.parquet"):
                symbol = file.stem.replace(f"_{interval}", "")
                symbols.append(symbol)
    else:
        if DATA_DIR.exists():
            for file in DATA_DIR.glob(f"KRW-*_{interval}.parquet"):
                symbol = file.stem.replace(f"_{interval}", "").replace("KRW-", "")
                symbols.append(symbol)
    return sorted(symbols)


def _get_data_files(
    symbols: list[str], interval: str = "day", exchange: str = "upbit"
) -> dict[str, Path]:
    """Build data file paths for given symbols."""
    data_files: dict[str, Path] = {}
    for symbol in symbols:
        if exchange == "binance":
            # Binance symbols are used as-is (e.g., BTCUSDT)
            file_path = BINANCE_DATA_DIR / parquet_filename(symbol, interval)
            if file_path.exists():
                data_files[symbol] = file_path
            else:
                logger.warning(f"Data not found for {symbol}: {file_path}")
        else:
            ticker = f"KRW-{symbol}"
            file_path = DATA_DIR / parquet_filename(ticker, interval)
            if file_path.exists():
                data_files[ticker] = file_path
            else:
                logger.warning(f"Data not found for {symbol}: {file_path}")
    return data_files


def _to_datetime(d: date | datetime | None) -> datetime | None:
    """Convert date to datetime if needed, pass through datetime/None."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d
    return datetime.combine(d, datetime.min.time())


def _convert_result(result: BacktestResult) -> VboBacktestResult:
    """Convert native BacktestResult to VboBacktestResult."""
    # Calculate sortino ratio from equity curve
    equity = np.array(result.equity_curve)
    if len(equity) > 1:
        daily_returns = np.diff(equity) / equity[:-1]
        sortino = calculate_sortino_ratio(daily_returns)
    else:
        sortino = 0.0

    # Calculate avg win/loss percentages
    win_returns: list[float] = []
    loss_returns: list[float] = []
    trades_list: list[dict[str, Any]] = []

    for trade in result.trades:
        return_pct = trade.pnl_pct
        trades_list.append(
            {
                "symbol": trade.ticker,
                "entry_date": _to_datetime(trade.entry_date),
                "exit_date": _to_datetime(trade.exit_date),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price or 0.0,
                "quantity": (trade.amount / trade.entry_price if trade.entry_price > 0 else 0.0),
                "pnl": trade.pnl,
                "return_pct": return_pct,
            }
        )

        if return_pct > 0:
            win_returns.append(return_pct)
        elif return_pct < 0:
            loss_returns.append(return_pct)

    avg_win_pct = sum(win_returns) / len(win_returns) if win_returns else 0.0
    avg_loss_pct = sum(loss_returns) / len(loss_returns) if loss_returns else 0.0

    # Convert dates to datetime objects
    dates_list: list[datetime] = []
    if result.dates is not None and len(result.dates) > 0:
        dates_array = pd.to_datetime(result.dates)
        dates_list = [d.to_pydatetime() for d in dates_array]

    # Calculate yearly returns
    yearly_returns: dict[int, float] = {}
    if len(equity) > 1 and len(dates_list) > 0:
        df_equity = pd.DataFrame(
            {"equity": equity[: len(dates_list)]},
            index=pd.DatetimeIndex(dates_list),
        )
        dt_index = pd.DatetimeIndex(df_equity.index)
        for year, group in df_equity.groupby(dt_index.year):
            if len(group) >= 2:
                year_return = (group["equity"].iloc[-1] / group["equity"].iloc[0] - 1) * 100
                yearly_returns[int(str(year))] = float(year_return)

    return VboBacktestResult(
        total_return=result.total_return,
        cagr=result.cagr,
        mdd=result.mdd,
        sharpe_ratio=result.sharpe_ratio,
        sortino_ratio=sortino,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        num_trades=result.total_trades,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        final_equity=float(equity[-1]) if len(equity) > 0 else 0.0,
        equity_curve=[float(e) for e in equity],
        dates=dates_list,
        yearly_returns=yearly_returns,
        trades=trades_list,
    )


def _run_native_backtest(
    strategy: Strategy,
    symbols: list[str],
    interval: str,
    initial_cash: int,
    fee: float,
    slippage: float,
    start_date: date | None,
    end_date: date | None,
) -> VboBacktestResult | None:
    """Run a backtest using the native vectorized engine."""
    data_files = _get_data_files(symbols, interval)
    if not data_files:
        logger.error("No data loaded for any symbol")
        return None

    config = BacktestConfig(
        initial_capital=float(initial_cash),
        fee_rate=fee,
        slippage_rate=slippage,
        max_slots=len(data_files),
        use_cache=False,
    )

    engine = VectorizedBacktestEngine(config)

    try:
        result = engine.run(
            strategy=strategy,
            data_files=data_files,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(
            f"Backtest completed: CAGR={result.cagr:.2f}%, "
            f"MDD={result.mdd:.2f}%, Trades={result.total_trades}"
        )
        return _convert_result(result)
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return None


def run_vbo_backtest_service(
    symbols: tuple[str, ...],
    interval: str = "day",
    initial_cash: int = 10_000_000,
    fee: float = 0.0005,
    slippage: float = 0.0005,
    multiplier: int = 2,
    lookback: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
) -> VboBacktestResult | None:
    """Run VBO backtest with BTC MA filter.

    Args:
        symbols: Tuple of symbols to trade
        interval: Time interval
        initial_cash: Initial capital in KRW
        fee: Trading fee (0.0005 = 0.05%)
        slippage: Slippage (0.0005 = 0.05%)
        multiplier: Multiplier for long-term MA
        lookback: Lookback period for short-term MA
        start_date: Backtest start date (inclusive)
        end_date: Backtest end date (inclusive)

    Returns:
        VboBacktestResult or None on failure
    """
    from src.strategies.registry import registry

    strategy = registry.create(
        "VBO",
        ma_short=lookback,
        btc_ma=lookback * multiplier,
        data_dir=DATA_DIR,
        interval=interval,
    )

    return _run_native_backtest(
        strategy=strategy,
        symbols=list(symbols),
        interval=interval,
        initial_cash=initial_cash,
        fee=fee,
        slippage=slippage,
        start_date=start_date,
        end_date=end_date,
    )


def run_vbo_backtest_generic_service(
    strategy_type: str,
    symbols: tuple[str, ...],
    interval: str = "day",
    initial_cash: int = 10_000_000,
    fee: float = 0.0005,
    slippage: float = 0.0005,
    start_date: date | None = None,
    end_date: date | None = None,
    **strategy_params: int | float | str,
) -> VboBacktestResult | None:
    """Run backtest with any VBO strategy variant.

    Args:
        strategy_type: Strategy type (currently only vbo supported)
        symbols: Tuple of symbols to trade
        interval: Time interval
        initial_cash: Initial capital in KRW
        fee: Trading fee
        slippage: Slippage
        start_date: Backtest start date
        end_date: Backtest end date
        **strategy_params: Strategy-specific parameters

    Returns:
        VboBacktestResult or None on failure
    """
    strategy = _create_strategy(strategy_type, interval, **strategy_params)
    if strategy is None:
        logger.error(f"Unknown strategy type: {strategy_type}")
        return None

    return _run_native_backtest(
        strategy=strategy,
        symbols=list(symbols),
        interval=interval,
        initial_cash=initial_cash,
        fee=fee,
        slippage=slippage,
        start_date=start_date,
        end_date=end_date,
    )


def _create_strategy(
    strategy_type: str,
    interval: str = "day",
    **params: int | float | str,
) -> Strategy | None:
    """Create a Strategy instance from strategy type string.

    Looks up the strategy in the central registry. Falls back to VBO
    for legacy "vbo" / "vbov1" inputs.
    """
    from src.strategies.registry import registry

    canonical = strategy_type.upper()

    # Normalise legacy aliases
    if canonical == "VBOV1":
        canonical = "VBO"

    if canonical not in registry:
        logger.error(
            f"Unknown strategy type: '{strategy_type}'. Available: {registry.list_names()}"
        )
        return None

    return registry.create(canonical, data_dir=DATA_DIR, interval=interval, **params)
