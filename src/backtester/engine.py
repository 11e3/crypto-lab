"""
Vectorized backtesting engine for trading strategies.

Uses pandas/numpy for high-performance backtesting with support for:
- Multiple assets with portfolio management
- Slotted position management
- Transaction costs and slippage
- Detailed performance metrics
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ANNUALIZATION_FACTOR,
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MAX_SLOTS,
    DEFAULT_SLIPPAGE_RATE,
    RAW_DATA_DIR,
)
from src.data.cache import get_cache
from src.data.collector import Interval
from src.data.collector_factory import DataCollectorFactory
from src.risk.metrics import (
    PortfolioRiskMetrics,
    calculate_portfolio_risk_metrics,
)
from src.risk.portfolio_optimization import optimize_portfolio
from src.risk.position_sizing import (
    calculate_multi_asset_position_sizes,
    calculate_position_size,
)
from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.utils.memory import get_float_dtype, optimize_dtypes

# Import for type checking
try:
    from src.strategies.pair_trading import PairTradingStrategy
except ImportError:
    PairTradingStrategy = None  # type: ignore

from src.execution.advanced_orders import AdvancedOrderManager

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    fee_rate: float = DEFAULT_FEE_RATE
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE
    max_slots: int = DEFAULT_MAX_SLOTS
    position_sizing: str = "equal"  # "equal", "volatility", "fixed-risk", "inverse-volatility", "mpt", "risk_parity", "kelly"
    position_sizing_risk_pct: float = 0.02  # Target risk per position (for fixed-risk method)
    position_sizing_lookback: int = 20  # Lookback period for volatility calculation
    use_cache: bool = True  # Cache indicator calculations

    # Advanced order settings
    stop_loss_pct: float | None = None  # Stop loss as percentage (e.g., 0.05 = 5%)
    take_profit_pct: float | None = None  # Take profit as percentage (e.g., 0.10 = 10%)
    trailing_stop_pct: float | None = None  # Trailing stop as percentage (e.g., 0.05 = 5%)

    # Portfolio optimization settings
    portfolio_optimization_method: str | None = (
        None  # "mpt", "risk_parity", "kelly" (None = use position_sizing)
    )
    risk_free_rate: float = 0.0  # Risk-free rate for MPT (annualized)
    max_kelly: float = 0.25  # Maximum Kelly percentage (fractional Kelly)


@dataclass
class Trade:
    """Record of a single trade."""

    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    amount: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_whipsaw: bool = False

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_date is not None


@dataclass
class BacktestResult:
    """Results from backtesting."""

    # Performance metrics
    total_return: float = 0.0
    cagr: float = 0.0
    mdd: float = 0.0
    calmar_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_return: float = 0.0

    # Time series data
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    dates: np.ndarray = field(default_factory=lambda: np.array([]))
    trades: list[Trade] = field(default_factory=list)

    # Additional info
    config: BacktestConfig | None = None
    strategy_name: str = ""
    interval: str = "day"  # Data interval (day, minute240, week)

    # Portfolio risk metrics
    risk_metrics: PortfolioRiskMetrics | None = None

    def summary(self) -> str:
        """Generate summary string."""
        final_equity = self.equity_curve[-1] if len(self.equity_curve) > 0 else 0
        summary = (
            f"\n{'=' * 50}\n"
            f"Strategy: {self.strategy_name}\n"
            f"{'=' * 50}\n"
            f"CAGR: {self.cagr:.2f}%\n"
            f"MDD: {self.mdd:.2f}%\n"
            f"Calmar Ratio: {self.calmar_ratio:.2f}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Win Rate: {self.win_rate:.2f}%\n"
            f"Total Trades: {self.total_trades}\n"
            f"Final Equity: {final_equity:.4f}\n"
        )

        # Add risk metrics if available
        if self.risk_metrics:
            summary += (
                f"\n--- Risk Metrics ---\n"
                f"VaR (95%): {self.risk_metrics.var_95 * 100:.2f}%\n"
                f"CVaR (95%): {self.risk_metrics.cvar_95 * 100:.2f}%\n"
                f"Portfolio Volatility: {self.risk_metrics.portfolio_volatility * 100:.2f}%\n"
            )

        summary += f"{'=' * 50}"
        return summary


class VectorizedBacktestEngine:
    """
    Vectorized backtesting engine using pandas/numpy.

    Pre-computes all signals and uses array operations for simulation.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.advanced_order_manager = AdvancedOrderManager()

    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load OHLCV data from parquet file.

        Args:
            filepath: Path to parquet file

        Returns:
            DataFrame with OHLCV data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is corrupted or invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            df = pd.read_parquet(filepath)
            df.index = pd.to_datetime(df.index)
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            raise ValueError(f"Error loading data from {filepath}: {e}") from e

    def _add_price_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add entry/exit price columns with slippage.

        Args:
            df: DataFrame with signals

        Returns:
            DataFrame with price columns added
        """
        # Only copy if we need to modify
        if (
            "target" not in df.columns
            or "entry_price" not in df.columns
            or "exit_price" not in df.columns
        ):
            df = df.copy()

        # Whipsaw: entry occurs but close < sma on same bar
        df["is_whipsaw"] = df["entry_signal"] & df["exit_signal"]

        # Entry price: use 'target' if available (VBO strategy), otherwise use 'close'
        if "target" in df.columns:
            # VBO strategy: use target price with slippage
            df["entry_price"] = df["target"] * (1 + self.config.slippage_rate)
        else:
            # Other strategies (momentum, etc.): use close price with slippage
            df["entry_price"] = df["close"] * (1 + self.config.slippage_rate)

        # Exit price (close with slippage)
        df["exit_price"] = df["close"] * (1 - self.config.slippage_rate)

        return df

    def _calculate_metrics_vectorized(
        self,
        equity_curve: np.ndarray,
        dates: np.ndarray,
        trades_df: pd.DataFrame,
        asset_returns: dict[str, list[float]] | None = None,
    ) -> BacktestResult:
        """Calculate performance metrics using vectorized operations."""
        result = BacktestResult(
            equity_curve=equity_curve,
            dates=dates,
            config=self.config,
        )

        if len(equity_curve) < 2:
            return result

        initial = self.config.initial_capital
        final = equity_curve[-1]

        # Total return
        result.total_return = (final / initial - 1) * 100

        # CAGR
        total_days = (dates[-1] - dates[0]).days
        if total_days > 0 and initial > 0 and final > 0:
            result.cagr = ((final / initial) ** (365.0 / total_days) - 1) * 100

        # MDD (vectorized)
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (cummax - equity_curve) / cummax
        result.mdd = np.nanmax(drawdown) * 100

        # Calmar Ratio
        result.calmar_ratio = result.cagr / result.mdd if result.mdd > 0 else 0.0

        # Sharpe Ratio (vectorized)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        daily_returns = np.insert(returns, 0, 0)  # First day has 0 return
        if len(returns) > 0 and np.std(returns) > 0:
            result.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(
                ANNUALIZATION_FACTOR
            )

        # Calculate portfolio risk metrics
        # Convert asset returns to numpy arrays (align by length)
        asset_returns_dict: dict[str, np.ndarray] | None = None
        if asset_returns:
            # Find minimum length to align all returns
            valid_returns = {k: v for k, v in asset_returns.items() if len(v) > 0}
            if valid_returns:
                min_length = min(len(returns) for returns in valid_returns.values())
                if min_length > 0:
                    asset_returns_dict = {
                        ticker: np.array(returns[-min_length:])
                        for ticker, returns in valid_returns.items()
                    }

        # Calculate final position values for concentration analysis
        # Use final equity curve value to estimate position values
        position_values_dict: dict[str, float] | None = None
        if len(trades_df) > 0 and len(equity_curve) > 0:
            # Get open positions from trades (positions that haven't been closed)
            open_trades = trades_df[trades_df["exit_date"].isna()]
            if len(open_trades) > 0:
                position_values_dict = {}
                total_position_value = 0.0
                for _, trade in open_trades.iterrows():
                    ticker = trade["ticker"]
                    if ticker not in position_values_dict:
                        position_values_dict[ticker] = 0.0
                    # Estimate position value (using amount * entry_price as approximation)
                    position_value = trade["amount"] * trade.get("entry_price", 0.0)
                    position_values_dict[ticker] += position_value
                    total_position_value += position_value

                # Normalize position values to portfolio value if total exceeds equity
                final_equity = equity_curve[-1] if len(equity_curve) > 0 else 0.0
                if (
                    total_position_value > 0
                    and final_equity > 0
                    and total_position_value > final_equity
                ):
                    scale_factor = final_equity / total_position_value
                    position_values_dict = {
                        ticker: value * scale_factor
                        for ticker, value in position_values_dict.items()
                    }

        try:
            result.risk_metrics = calculate_portfolio_risk_metrics(
                equity_curve=equity_curve,
                daily_returns=daily_returns,
                asset_returns=asset_returns_dict,
                position_values=position_values_dict,
                total_portfolio_value=equity_curve[-1] if len(equity_curve) > 0 else 0.0,
                benchmark_returns=None,
                annualization_factor=ANNUALIZATION_FACTOR,
            )
        except Exception as e:
            logger.warning(f"Failed to calculate risk metrics: {e}")
            result.risk_metrics = None

        # Trade statistics
        if len(trades_df) > 0:
            closed = trades_df[trades_df["exit_date"].notna()]
            result.total_trades = len(closed)

            if len(closed) > 0:
                result.winning_trades = (closed["pnl"] > 0).sum()
                result.losing_trades = (closed["pnl"] <= 0).sum()
                result.win_rate = (result.winning_trades / len(closed)) * 100
                result.avg_trade_return = closed["pnl_pct"].mean()

                total_profit = closed.loc[closed["pnl"] > 0, "pnl"].sum()
                total_loss = abs(closed.loc[closed["pnl"] <= 0, "pnl"].sum())
                result.profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        return result

    def _get_cache_params(self, strategy: Strategy) -> dict[str, Any]:
        """Extract cache parameters from strategy."""
        params: dict[str, Any] = {"strategy_name": strategy.name}

        # Extract VBO-specific parameters if available
        for attr in ["sma_period", "trend_sma_period", "short_noise_period", "long_noise_period"]:
            if hasattr(strategy, attr):
                params[attr] = getattr(strategy, attr)

        # Include entry/exit conditions in cache key
        if hasattr(strategy, "entry_conditions"):
            params["entry_conditions"] = [c.name for c in strategy.entry_conditions.conditions]
        if hasattr(strategy, "exit_conditions"):
            params["exit_conditions"] = [c.name for c in strategy.exit_conditions.conditions]

        return params

    def run(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult:
        """
        Run vectorized backtest for a strategy on multiple assets.

        Args:
            strategy: Trading strategy to backtest
            data_files: Dictionary mapping ticker to data file path
            start_date: Start date for filtering data (inclusive)
            end_date: End date for filtering data (inclusive)

        Returns:
            BacktestResult with performance metrics
        """
        # Check if this is a pair trading strategy
        is_pair_trading = PairTradingStrategy is not None and isinstance(
            strategy, PairTradingStrategy
        )

        if is_pair_trading:
            # Pair trading requires exactly 2 tickers
            if len(data_files) != 2:
                raise ValueError(
                    f"Pair trading strategy requires exactly 2 tickers, "
                    f"got {len(data_files)}: {list(data_files.keys())}"
                )
            # Use special pair trading processing
            return self._run_pair_trading(
                strategy, data_files, start_date=start_date, end_date=end_date
            )

        # Get cache parameters
        cache_params = self._get_cache_params(strategy)
        cache = get_cache() if self.config.use_cache else None

        # Load and prepare data for all tickers
        ticker_data: dict[str, pd.DataFrame] = {}
        all_dates: set = set()

        # Store original dataframes for position sizing (need historical data)
        ticker_historical_data: dict[str, pd.DataFrame] = {}

        for ticker, filepath in data_files.items():
            try:
                # Extract interval from filename (e.g., KRW-BTC_day.parquet -> day)
                interval = filepath.stem.split("_")[1] if "_" in filepath.stem else "unknown"
                raw_mtime = filepath.stat().st_mtime if filepath.exists() else None

                # Try to load from cache
                cached_df = None
                if cache is not None:
                    cached_df = cache.get(ticker, interval, cache_params, raw_mtime)

                if cached_df is not None:
                    # Use cached data (already has indicators and signals)
                    df = cached_df
                    logger.debug(f"Loaded {ticker} from cache")
                else:
                    # Calculate indicators and signals
                    df = self.load_data(filepath)
                    # Optimize dtypes before storing (memory efficiency)
                    df = optimize_dtypes(df)
                    # Store original data for position sizing (only if needed for position sizing)
                    if self.config.position_sizing != "equal":
                        ticker_historical_data[ticker] = df.copy()
                    df = strategy.calculate_indicators(df)
                    df = strategy.generate_signals(df)

                    # Save to cache
                    if cache is not None:
                        cache.set(ticker, interval, cache_params, df, raw_mtime)
                        logger.debug(f"Saved {ticker} to cache")

                df = self._add_price_columns(df)
                df["ticker"] = ticker
                # Optimize dtypes before storing
                df = optimize_dtypes(df)
                ticker_data[ticker] = df
                # Store historical data if not already stored (for cached data and position sizing)
                if ticker not in ticker_historical_data and self.config.position_sizing != "equal":
                    ticker_historical_data[ticker] = df.copy()
                # Only add dates where indicators are valid (matching legacy: only dates in processed_data)
                # Legacy skips first 10 days per ticker, so only dates with valid indicators exist
                # Check for required indicators based on strategy
                valid_mask = df["close"].notna()
                if "sma" in df.columns:
                    valid_mask = valid_mask & df["sma"].notna()
                if "target" in df.columns:
                    valid_mask = valid_mask & df["target"].notna()
                valid_dates = df.index[valid_mask].date
                all_dates.update(valid_dates)
            except Exception as e:
                logger.error(f"Error processing {ticker} from {filepath}: {e}", exc_info=True)
                continue

        # Create unified timeline
        sorted_dates = np.array(sorted(all_dates))
        n_dates = len(sorted_dates)

        if n_dates == 0:
            logger.warning("No data available for backtesting")
            return BacktestResult(strategy_name=strategy.name)

        # Pre-build lookup arrays for each ticker
        # Shape: (n_tickers, n_dates) for each attribute
        tickers = list(ticker_data.keys())
        n_tickers = len(tickers)

        # Initialize arrays with optimized dtypes (float32 for memory efficiency)
        float_dtype = get_float_dtype()
        opens = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        highs = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        closes = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        targets = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        smas = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        entry_signals = np.zeros((n_tickers, n_dates), dtype=bool)
        exit_signals = np.zeros((n_tickers, n_dates), dtype=bool)
        whipsaws = np.zeros((n_tickers, n_dates), dtype=bool)
        entry_prices = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        exit_prices = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        short_noises = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)

        # Check if any ticker has 'target' column (VBO strategy)
        any("target" in df.columns for df in ticker_data.values())

        # Date to index mapping (using dictionary for O(1) lookup)
        date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

        # Fill arrays using vectorized operations
        for t_idx, ticker in enumerate(tickers):
            df = ticker_data[ticker]

            # Get date indices for this ticker's data
            df_dates = pd.Series(df.index.date)
            df_idx = df_dates.map(date_to_idx)
            valid_mask = df_idx.notna()
            idx = df_idx[valid_mask].astype(int).values

            opens[t_idx, idx] = df.loc[valid_mask.values, "open"].values
            highs[t_idx, idx] = df.loc[valid_mask.values, "high"].values
            closes[t_idx, idx] = df.loc[valid_mask.values, "close"].values
            if "target" in df.columns:
                targets[t_idx, idx] = df.loc[valid_mask.values, "target"].values
            else:
                # For non-VBO strategies, use close as target
                targets[t_idx, idx] = df.loc[valid_mask.values, "close"].values
            if "sma" in df.columns:
                smas[t_idx, idx] = df.loc[valid_mask.values, "sma"].values
            entry_signals[t_idx, idx] = df.loc[valid_mask.values, "entry_signal"].values
            exit_signals[t_idx, idx] = df.loc[valid_mask.values, "exit_signal"].values
            whipsaws[t_idx, idx] = df.loc[valid_mask.values, "is_whipsaw"].values
            entry_prices[t_idx, idx] = df.loc[valid_mask.values, "entry_price"].values
            exit_prices[t_idx, idx] = df.loc[valid_mask.values, "exit_price"].values
            if "short_noise" in df.columns:
                short_noises[t_idx, idx] = df.loc[valid_mask.values, "short_noise"].values

        # Filter dates: Skip initial days where indicators are not available (matching legacy/bt.py)
        # Legacy skips first max(noise_period, trend_sma_period, long_noise_period) = 10 days per ticker
        # Then only processes dates that exist in processed_data (which excludes first 10 days)
        # So we need to filter to dates where at least one ticker has valid indicators
        # AND only process dates that would be in legacy's sorted_dates

        # Find dates where at least one ticker has valid indicators (non-NaN target, sma, etc.)
        # This matches legacy behavior where processed_data only contains dates after skip period
        valid_date_mask = np.zeros(n_dates, dtype=bool)
        for d_idx in range(n_dates):
            # Check if at least one ticker has valid indicators for this date
            # Legacy only includes dates in processed_data, which excludes first 10 days per ticker
            has_valid_target = ~np.isnan(targets[:, d_idx])
            has_valid_data = ~np.isnan(closes[:, d_idx])

            # SMA is optional (some strategies may not use it)
            if np.any(~np.isnan(smas[:, d_idx])):
                has_valid_sma = ~np.isnan(smas[:, d_idx])
            else:
                has_valid_sma = np.ones(n_tickers, dtype=bool)  # No SMA required

            # Date is valid if at least one ticker has all required indicators
            # This matches legacy: dates only exist in market_data if they're in processed_data
            if np.any(has_valid_target & has_valid_sma & has_valid_data):
                valid_date_mask[d_idx] = True

        # Filter to only valid dates (matching legacy: only dates in processed_data)
        valid_indices = np.where(valid_date_mask)[0]
        if len(valid_indices) > 0:
            first_valid_idx = valid_indices[0]
            last_valid_idx = valid_indices[-1] + 1

            logger.debug(
                f"Filtering dates: keeping {len(valid_indices)} valid dates (matching legacy/bt.py)"
            )
            sorted_dates = sorted_dates[first_valid_idx:last_valid_idx]
            n_dates = len(sorted_dates)

            # Filter all arrays to valid date range
            opens = opens[:, first_valid_idx:last_valid_idx]
            highs = highs[:, first_valid_idx:last_valid_idx]
            closes = closes[:, first_valid_idx:last_valid_idx]
            targets = targets[:, first_valid_idx:last_valid_idx]
            smas = smas[:, first_valid_idx:last_valid_idx]
            entry_signals = entry_signals[:, first_valid_idx:last_valid_idx]
            exit_signals = exit_signals[:, first_valid_idx:last_valid_idx]
            whipsaws = whipsaws[:, first_valid_idx:last_valid_idx]
            entry_prices = entry_prices[:, first_valid_idx:last_valid_idx]
            exit_prices = exit_prices[:, first_valid_idx:last_valid_idx]
            short_noises = short_noises[:, first_valid_idx:last_valid_idx]

        # Simulation using numpy arrays
        cash = self.config.initial_capital
        max_slots = self.config.max_slots
        fee_rate = self.config.fee_rate

        # Position tracking: -1 means no position (use float32 for memory efficiency)
        float_dtype = get_float_dtype()
        position_amounts = np.zeros(n_tickers, dtype=float_dtype)
        position_entry_prices = np.zeros(n_tickers, dtype=float_dtype)
        position_entry_dates = np.full(n_tickers, -1, dtype=np.int32)  # int32 sufficient

        equity_curve = np.zeros(n_dates, dtype=float_dtype)
        trades_list: list[dict] = []

        # Track individual asset returns for correlation analysis
        asset_returns: dict[str, list[float]] = {ticker: [] for ticker in tickers}
        previous_closes = np.full(n_tickers, np.nan, dtype=float_dtype)

        for d_idx in range(n_dates):
            current_date = sorted_dates[d_idx]

            # Get valid data mask for this date
            valid_data = ~np.isnan(closes[:, d_idx])

            # Track individual asset returns
            for t_idx in range(n_tickers):
                if valid_data[t_idx] and not np.isnan(closes[t_idx, d_idx]):
                    current_close = closes[t_idx, d_idx]
                    if not np.isnan(previous_closes[t_idx]):
                        # Calculate daily return
                        daily_return = (current_close - previous_closes[t_idx]) / previous_closes[
                            t_idx
                        ]
                        asset_returns[tickers[t_idx]].append(daily_return)
                    previous_closes[t_idx] = current_close

            # ---- PROCESS EXITS ----
            in_position = position_amounts > 0
            should_exit = exit_signals[:, d_idx] & in_position & valid_data

            if np.any(should_exit):
                exit_idx = np.where(should_exit)[0]
                for t_idx in exit_idx:
                    # Execute exit
                    exit_price = exit_prices[t_idx, d_idx]
                    amount = position_amounts[t_idx]
                    revenue = amount * exit_price * (1 - fee_rate)
                    cash += revenue

                    # Record trade
                    entry_price = position_entry_prices[t_idx]
                    pnl = revenue - (amount * entry_price)
                    pnl_pct = (exit_price / entry_price - 1) * 100

                    trades_list.append(
                        {
                            "ticker": tickers[t_idx],
                            "entry_date": sorted_dates[position_entry_dates[t_idx]],
                            "entry_price": entry_price,
                            "exit_date": current_date,
                            "exit_price": exit_price,
                            "amount": amount,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "is_whipsaw": False,
                        }
                    )

                    # Cancel advanced orders for this position
                    self.advanced_order_manager.cancel_all_orders(ticker=tickers[t_idx])

                    # Clear position
                    position_amounts[t_idx] = 0
                    position_entry_prices[t_idx] = 0
                    position_entry_dates[t_idx] = -1

            # ---- PROCESS ENTRIES ----
            not_in_position = position_amounts == 0
            can_enter = entry_signals[:, d_idx] & not_in_position & valid_data

            if np.any(can_enter):
                # Sort by noise (prefer lower noise) if available, otherwise use entry order
                candidate_idx = np.where(can_enter)[0]
                if np.any(~np.isnan(short_noises[candidate_idx, d_idx])):
                    noise_values = short_noises[candidate_idx, d_idx]
                    # Replace NaN with inf to sort them last
                    noise_values = np.where(np.isnan(noise_values), np.inf, noise_values)
                    sorted_order = np.argsort(noise_values)
                    candidate_idx = candidate_idx[sorted_order]

                # Calculate position sizes for all candidates at once if using dynamic sizing
                position_sizes: dict[str, float] = {}
                if self.config.position_sizing != "equal" and len(candidate_idx) > 1:
                    # Check if using portfolio optimization methods
                    optimization_method = (
                        self.config.portfolio_optimization_method or self.config.position_sizing
                    )

                    if optimization_method in ["mpt", "risk_parity"]:
                        # Portfolio optimization methods
                        candidate_tickers = [tickers[idx] for idx in candidate_idx]

                        # Get historical returns for optimization
                        returns_data: dict[str, list[float]] = {}
                        for idx in candidate_idx:
                            ticker = tickers[idx]
                            if (
                                ticker in asset_returns
                                and len(asset_returns[ticker])
                                >= self.config.position_sizing_lookback
                            ):
                                # Use recent returns
                                recent_returns = asset_returns[ticker][
                                    -self.config.position_sizing_lookback :
                                ]
                                returns_data[ticker] = recent_returns

                        if len(returns_data) >= 2:
                            # Create returns DataFrame
                            max_len = max(len(r) for r in returns_data.values())
                            returns_df = pd.DataFrame(
                                {
                                    ticker: r + [np.nan] * (max_len - len(r))
                                    for ticker, r in returns_data.items()
                                }
                            )
                            returns_df = returns_df.dropna()

                            if not returns_df.empty and len(returns_df) >= 10:
                                try:
                                    # Optimize portfolio
                                    weights = optimize_portfolio(
                                        returns_df,
                                        method=optimization_method,
                                        risk_free_rate=self.config.risk_free_rate,
                                    )

                                    # Convert weights to position sizes
                                    for ticker, weight in weights.weights.items():
                                        if ticker in candidate_tickers:
                                            position_sizes[ticker] = cash * weight
                                except Exception as e:
                                    logger.warning(
                                        f"Portfolio optimization failed: {e}, falling back to equal"
                                    )
                                    # Fallback to equal
                                    position_sizes = {}

                    elif optimization_method == "kelly":
                        # Kelly Criterion: need trade history
                        if len(trades_list) >= 10:
                            try:
                                # Convert trades to DataFrame
                                trades_df = pd.DataFrame(trades_list)
                                if "pnl_pct" in trades_df.columns:
                                    # Calculate Kelly allocation
                                    from src.risk.portfolio_optimization import PortfolioOptimizer

                                    optimizer = PortfolioOptimizer()
                                    kelly_allocations = optimizer.optimize_kelly_portfolio(
                                        trades_df,
                                        available_cash=cash,
                                        max_kelly=self.config.max_kelly,
                                    )
                                    position_sizes = kelly_allocations
                            except Exception as e:
                                logger.warning(
                                    f"Kelly optimization failed: {e}, falling back to equal"
                                )
                                position_sizes = {}

                    if not position_sizes:
                        # Fallback to standard multi-asset position sizing
                        candidate_tickers = [tickers[idx] for idx in candidate_idx]
                        candidate_prices = {
                            tickers[idx]: entry_prices[idx, d_idx]
                            for idx in candidate_idx
                            if not np.isnan(entry_prices[idx, d_idx])
                        }

                        # Get historical data up to current date
                        candidate_historical = {}
                        for idx in candidate_idx:
                            ticker = tickers[idx]
                            if ticker in ticker_historical_data:
                                # Get data up to current date
                                hist_df = ticker_historical_data[ticker]
                                # Filter to dates <= current_date
                                if "date" in hist_df.columns:
                                    candidate_historical[ticker] = hist_df[
                                        hist_df["date"] <= current_date
                                    ]
                                elif hist_df.index.name == "date" or isinstance(
                                    hist_df.index, pd.DatetimeIndex
                                ):
                                    candidate_historical[ticker] = hist_df.iloc[: d_idx + 1]
                                else:
                                    candidate_historical[ticker] = hist_df.iloc[: d_idx + 1]

                        position_sizes = calculate_multi_asset_position_sizes(
                            method=self.config.position_sizing,
                            available_cash=cash,
                            tickers=candidate_tickers,
                            current_prices=candidate_prices,
                            historical_data=candidate_historical,
                            target_risk_pct=self.config.position_sizing_risk_pct,
                            lookback_period=self.config.position_sizing_lookback,
                        )

                for t_idx in candidate_idx:
                    # Recalculate available_slots each iteration (matching legacy/bt.py)
                    current_positions = np.sum(position_amounts > 0)
                    available_slots = max_slots - current_positions

                    if (
                        available_slots <= 0
                    ):  # pragma: no cover (difficult to test - requires exact max_slots reached during loop)
                        break

                    sma_price = smas[t_idx, d_idx] if not np.isnan(smas[t_idx, d_idx]) else None
                    close_price = closes[t_idx, d_idx]
                    buy_price = entry_prices[t_idx, d_idx]

                    # Calculate position size based on method
                    if self.config.position_sizing != "equal" and tickers[t_idx] in position_sizes:
                        invest_amount = position_sizes[tickers[t_idx]]
                    elif self.config.position_sizing != "equal":
                        # Single asset or fallback: use individual calculation
                        ticker = tickers[t_idx]
                        if ticker in ticker_historical_data:
                            hist_df = ticker_historical_data[ticker]
                            if "date" in hist_df.columns:
                                hist_up_to_date = hist_df[hist_df["date"] <= current_date]
                            elif hist_df.index.name == "date" or isinstance(
                                hist_df.index, pd.DatetimeIndex
                            ):
                                hist_up_to_date = hist_df.iloc[: d_idx + 1]
                            else:
                                hist_up_to_date = hist_df.iloc[: d_idx + 1]

                            invest_amount = calculate_position_size(
                                method=self.config.position_sizing,
                                available_cash=cash,
                                available_slots=available_slots,
                                ticker=ticker,
                                current_price=buy_price,
                                historical_data=hist_up_to_date,
                                target_risk_pct=self.config.position_sizing_risk_pct,
                                lookback_period=self.config.position_sizing_lookback,
                            )
                        else:
                            # Fallback to equal
                            invest_amount = cash / available_slots
                    else:
                        # Equal sizing (default)
                        invest_amount = cash / available_slots

                    # Check whipsaw: entry condition met but close < sma on same day
                    # (matching legacy/bt.py logic: if row["close"] < row["sma"])
                    is_whipsaw = close_price < sma_price

                    if is_whipsaw:
                        # Whipsaw: buy and sell same day
                        sell_price = exit_prices[t_idx, d_idx]
                        amount = (invest_amount / buy_price) * (1 - fee_rate)
                        return_money = amount * sell_price * (1 - fee_rate)
                        cash = cash - invest_amount + return_money

                        pnl = return_money - invest_amount
                        pnl_pct = (sell_price / buy_price - 1) * 100

                        trades_list.append(
                            {
                                "ticker": tickers[t_idx],
                                "entry_date": current_date,
                                "entry_price": buy_price,
                                "exit_date": current_date,
                                "exit_price": sell_price,
                                "amount": amount,
                                "pnl": pnl,
                                "pnl_pct": pnl_pct,
                                "is_whipsaw": True,
                            }
                        )
                    else:
                        # Normal entry
                        amount = (invest_amount / buy_price) * (1 - fee_rate)

                        position_amounts[t_idx] = amount
                        position_entry_prices[t_idx] = buy_price
                        position_entry_dates[t_idx] = d_idx
                        cash -= invest_amount

                        trades_list.append(
                            {
                                "ticker": tickers[t_idx],
                                "entry_date": current_date,
                                "entry_price": buy_price,
                                "exit_date": None,
                                "exit_price": None,
                                "amount": amount,
                                "pnl": 0.0,
                                "pnl_pct": 0.0,
                                "is_whipsaw": False,
                            }
                        )
                        # Note: available_slots is recalculated each iteration, no need to decrement

            # ---- CALCULATE DAILY EQUITY ----
            # Calculate position values, using previous close if current is NaN
            # This prevents sudden drops when data is missing for some tickers
            positions_value = 0.0
            for t_idx in range(n_tickers):
                if position_amounts[t_idx] > 0:
                    # If we have a position, use current close if valid, otherwise previous close
                    if valid_data[t_idx] and not np.isnan(closes[t_idx, d_idx]):
                        current_price = closes[t_idx, d_idx]
                    elif d_idx > 0 and not np.isnan(closes[t_idx, d_idx - 1]):
                        # Use previous day's close if current is missing
                        current_price = closes[t_idx, d_idx - 1]
                    else:
                        # If no valid price available, use entry price as fallback
                        current_price = position_entry_prices[t_idx]
                    positions_value += position_amounts[t_idx] * current_price

            # Calculate equity
            equity_curve[d_idx] = cash + positions_value
            # Forward fill if equity is NaN or negative (shouldn't happen, but safety check)
            if (np.isnan(equity_curve[d_idx]) or equity_curve[d_idx] < 0) and d_idx > 0:
                equity_curve[d_idx] = equity_curve[d_idx - 1]

        # Convert trades to DataFrame for vectorized metrics and optimize dtypes
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        if not trades_df.empty:
            trades_df = optimize_dtypes(trades_df)

        # Calculate metrics
        result = self._calculate_metrics_vectorized(
            equity_curve, sorted_dates, trades_df, asset_returns=asset_returns
        )
        result.strategy_name = strategy.name

        # Convert trades_df to Trade objects for compatibility
        if len(trades_df) > 0:
            result.trades = [
                Trade(
                    ticker=row["ticker"],
                    entry_date=row["entry_date"],
                    entry_price=row["entry_price"],
                    exit_date=row["exit_date"],
                    exit_price=row["exit_price"],
                    amount=row["amount"],
                    pnl=row["pnl"],
                    pnl_pct=row["pnl_pct"],
                    is_whipsaw=row["is_whipsaw"],
                )
                for _, row in trades_df.iterrows()
            ]

        return result

    def _run_pair_trading(
        self,
        strategy: "PairTradingStrategy",
        data_files: dict[str, Path],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult:
        """
        Run backtest for pair trading strategy.

        Pair trading requires two tickers and calculates spread between them.

        Args:
            strategy: PairTradingStrategy instance
            data_files: Dictionary with exactly 2 tickers

        Returns:
            BacktestResult with performance metrics
        """
        if PairTradingStrategy is None:
            raise ImportError("PairTradingStrategy is not available")

        tickers = list(data_files.keys())
        if len(tickers) != 2:
            raise ValueError(f"Pair trading requires exactly 2 tickers, got {len(tickers)}")

        ticker1, ticker2 = tickers[0], tickers[1]
        filepath1, filepath2 = data_files[ticker1], data_files[ticker2]

        # Load data for both tickers
        df1 = self.load_data(filepath1)
        df2 = self.load_data(filepath2)

        # Filter by date range if specified
        if start_date is not None or end_date is not None:
            df1_index = pd.to_datetime(df1.index)
            df2_index = pd.to_datetime(df2.index)
            if start_date is not None:
                df1 = df1[df1_index >= pd.Timestamp(start_date)]
                df2 = df2[df2_index >= pd.Timestamp(start_date)]
            if end_date is not None:
                df1 = df1[df1_index <= pd.Timestamp(end_date)]
                df2 = df2[df2_index <= pd.Timestamp(end_date)]
            logger.debug(
                f"Filtered pair trading data: "
                f"{len(df1)} rows for {ticker1}, {len(df2)} rows for {ticker2} "
                f"after date filtering ({start_date} to {end_date})"
            )

        # Calculate spread indicators using strategy's helper method
        merged_df = strategy.calculate_spread_for_pair(df1, df2)

        # Generate signals
        merged_df = strategy.generate_signals(merged_df)

        # For pair trading, we need to create signals for both tickers
        # Entry: when spread deviates (Z-score high), buy one and sell the other
        # Exit: when spread reverts (Z-score near zero), close both positions

        # Create ticker-specific dataframes
        df1_signals = df1.copy()
        df2_signals = df2.copy()

        # Align with merged_df dates
        common_dates = merged_df.index.intersection(df1_signals.index).intersection(
            df2_signals.index
        )
        df1_signals = df1_signals.loc[common_dates]
        df2_signals = df2_signals.loc[common_dates]
        merged_signals = merged_df.loc[common_dates]

        # Determine which asset to buy/sell based on Z-score sign
        # Positive Z-score: asset1 is overvalued relative to asset2
        #   -> Sell asset1, Buy asset2
        # Negative Z-score: asset1 is undervalued relative to asset2
        #   -> Buy asset1, Sell asset2

        # For simplicity, we'll use the spread-based entry/exit signals
        # and apply them to both assets (with opposite directions)
        # Copy signals (ensure boolean type)
        df1_signals["entry_signal"] = merged_signals["entry_signal"].astype(bool).values
        df1_signals["exit_signal"] = merged_signals["exit_signal"].astype(bool).values
        df2_signals["entry_signal"] = merged_signals["entry_signal"].astype(bool).values
        df2_signals["exit_signal"] = merged_signals["exit_signal"].astype(bool).values

        # Add price columns (using close price with slippage)
        df1_signals["entry_price"] = df1_signals["close"] * (1 + self.config.slippage_rate)
        df1_signals["exit_price"] = df1_signals["close"] * (1 - self.config.slippage_rate)
        df2_signals["entry_price"] = df2_signals["close"] * (1 + self.config.slippage_rate)
        df2_signals["exit_price"] = df2_signals["close"] * (1 - self.config.slippage_rate)

        # Store ticker data
        ticker_data = {ticker1: df1_signals, ticker2: df2_signals}

        # Get all dates
        all_dates = set(common_dates.date)

        if len(all_dates) == 0:
            logger.warning("No data available for pair trading backtest")
            return BacktestResult(strategy_name=strategy.name)

        # Create unified timeline
        sorted_dates = np.array(sorted(all_dates))
        n_dates = len(sorted_dates)

        # Pre-build lookup arrays for each ticker
        n_tickers = 2
        # Use float32 for memory efficiency
        float_dtype = get_float_dtype()
        opens = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        highs = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        closes = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        entry_signals = np.zeros((n_tickers, n_dates), dtype=bool)
        exit_signals = np.zeros((n_tickers, n_dates), dtype=bool)
        entry_prices = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)
        exit_prices = np.full((n_tickers, n_dates), np.nan, dtype=float_dtype)

        # Date to index mapping
        date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

        # Fill arrays
        for t_idx, ticker in enumerate([ticker1, ticker2]):
            df = ticker_data[ticker]
            # Use date part of datetime index for mapping
            df_dates = pd.Series(df.index.date)
            df_idx = df_dates.map(date_to_idx)
            valid_mask = df_idx.notna()

            if not valid_mask.any():
                logger.warning(f"No valid dates found for {ticker}")
                continue

            idx = df_idx[valid_mask].astype(int).values

            # Ensure indices are within bounds
            valid_idx = idx[idx < n_dates]
            if len(valid_idx) != len(idx):
                logger.warning(f"Some indices out of bounds for {ticker}")
                valid_mask = valid_mask & (df_idx < n_dates)
                idx = df_idx[valid_mask].astype(int).values

            opens[t_idx, idx] = df.loc[valid_mask.values, "open"].values
            highs[t_idx, idx] = df.loc[valid_mask.values, "high"].values
            closes[t_idx, idx] = df.loc[valid_mask.values, "close"].values
            # Ensure boolean type for signals
            entry_signals[t_idx, idx] = (
                df.loc[valid_mask.values, "entry_signal"].astype(bool).values
            )
            exit_signals[t_idx, idx] = df.loc[valid_mask.values, "exit_signal"].astype(bool).values
            entry_prices[t_idx, idx] = df.loc[valid_mask.values, "entry_price"].values
            exit_prices[t_idx, idx] = df.loc[valid_mask.values, "exit_price"].values

        # Filter to valid dates (need enough data for spread calculation)
        # Pair trading needs lookback_period days for spread mean/std
        min_required_days = strategy.lookback_period
        valid_date_mask = np.zeros(n_dates, dtype=bool)
        for d_idx in range(n_dates):
            has_valid_data = ~np.isnan(closes[:, d_idx])
            # Need both assets to have valid data
            if np.all(has_valid_data) and d_idx >= min_required_days:
                valid_date_mask[d_idx] = True

        valid_indices = np.where(valid_date_mask)[0]
        if len(valid_indices) > 0:
            first_valid_idx = valid_indices[0]
            last_valid_idx = valid_indices[-1] + 1
            sorted_dates = sorted_dates[first_valid_idx:last_valid_idx]
            n_dates = len(sorted_dates)
            opens = opens[:, first_valid_idx:last_valid_idx]
            highs = highs[:, first_valid_idx:last_valid_idx]
            closes = closes[:, first_valid_idx:last_valid_idx]
            entry_signals = entry_signals[:, first_valid_idx:last_valid_idx]
            exit_signals = exit_signals[:, first_valid_idx:last_valid_idx]
            entry_prices = entry_prices[:, first_valid_idx:last_valid_idx]
            exit_prices = exit_prices[:, first_valid_idx:last_valid_idx]
        else:
            logger.warning(
                f"No valid dates after filtering (need at least {min_required_days} days)"
            )
            return BacktestResult(strategy_name=strategy.name)

        # Simulate trading
        cash = self.config.initial_capital
        fee_rate = self.config.fee_rate
        max_slots = self.config.max_slots

        # Position tracking (for pair trading, we track positions in both assets)
        # Use float32 for memory efficiency
        float_dtype = get_float_dtype()
        position_amounts = np.zeros(n_tickers, dtype=float_dtype)
        position_entry_prices = np.zeros(n_tickers, dtype=float_dtype)
        position_entry_dates = np.full(n_tickers, -1, dtype=np.int32)

        equity_curve = np.zeros(n_dates, dtype=float_dtype)
        trades_list: list[dict] = []

        # Track individual asset returns for correlation analysis
        asset_returns: dict[str, list[float]] = {ticker: [] for ticker in tickers}
        previous_closes = np.full(n_tickers, np.nan, dtype=float_dtype)

        for d_idx in range(n_dates):
            current_date = sorted_dates[d_idx]
            valid_data = ~np.isnan(closes[:, d_idx])

            # Track individual asset returns
            for t_idx in range(n_tickers):
                if valid_data[t_idx] and not np.isnan(closes[t_idx, d_idx]):
                    current_close = closes[t_idx, d_idx]
                    if not np.isnan(previous_closes[t_idx]):
                        # Calculate daily return
                        daily_return = (current_close - previous_closes[t_idx]) / previous_closes[
                            t_idx
                        ]
                        asset_returns[tickers[t_idx]].append(daily_return)
                    previous_closes[t_idx] = current_close

            # For pair trading, both assets must have valid data
            if not np.all(valid_data):
                # Calculate daily equity even if no valid data
                # Use valid closes only to prevent NaN issues
                valid_closes = np.where(valid_data, closes[:, d_idx], np.nan)
                positions_value = np.nansum(position_amounts * valid_closes)
                # If positions_value is NaN (all closes are NaN), use previous equity or cash only
                if np.isnan(positions_value):
                    positions_value = 0.0
                equity_curve[d_idx] = cash + positions_value
                continue

            # Process exits first
            in_position = position_amounts > 0
            should_exit = exit_signals[:, d_idx] & in_position

            if np.any(should_exit):
                # Exit all positions (pair trading exits both together)
                for t_idx in range(n_tickers):
                    if position_amounts[t_idx] > 0:
                        exit_price = exit_prices[t_idx, d_idx]
                        if np.isnan(exit_price):
                            continue
                        amount = position_amounts[t_idx]
                        revenue = amount * exit_price * (1 - fee_rate)
                        cash += revenue

                        entry_price = position_entry_prices[t_idx]
                        pnl = revenue - (amount * entry_price)
                        pnl_pct = (exit_price / entry_price - 1) * 100

                        # Find corresponding entry trade and update it
                        for trade in trades_list:
                            if (
                                trade["ticker"] == tickers[t_idx]
                                and trade["exit_date"] is None
                                and trade["entry_date"] == sorted_dates[position_entry_dates[t_idx]]
                            ):
                                trade["exit_date"] = current_date
                                trade["exit_price"] = exit_price
                                trade["pnl"] = pnl
                                trade["pnl_pct"] = pnl_pct
                                break

                        position_amounts[t_idx] = 0.0
                        position_entry_prices[t_idx] = 0.0
                        position_entry_dates[t_idx] = -1

            # Process entries
            # For pair trading, we need both assets to have entry signals
            # and both should not be in position
            not_in_position = position_amounts == 0.0
            has_entry_signal = entry_signals[:, d_idx].astype(bool)
            has_entry_signal & not_in_position

            # For pair trading, enter both positions simultaneously when both have signals
            # Simplified: just check if both have entry signals and not in position
            if np.all(has_entry_signal) and np.all(not_in_position):
                # Both assets can enter - this is a pair trade
                current_positions = np.sum(position_amounts > 0)
                available_slots = max_slots - current_positions

                if available_slots >= 2:  # Need at least 2 slots for pair trading
                    # Enter both positions
                    for t_idx in range(n_tickers):
                        buy_price = entry_prices[t_idx, d_idx]
                        if np.isnan(buy_price):
                            logger.debug(
                                f"Pair trading: Skipping {tickers[t_idx]} at d_idx={d_idx} "
                                f"due to NaN entry_price"
                            )
                            continue
                        invest_amount = cash / 2  # Split capital between two assets

                        if invest_amount > 0:
                            amount = invest_amount / buy_price * (1 - fee_rate)
                            cash -= invest_amount

                            position_amounts[t_idx] = amount
                            position_entry_prices[t_idx] = buy_price
                            position_entry_dates[t_idx] = d_idx

                            # Add entry trade record
                            trades_list.append(
                                {
                                    "ticker": tickers[t_idx],
                                    "entry_date": current_date,
                                    "entry_price": buy_price,
                                    "exit_date": None,
                                    "exit_price": None,
                                    "amount": amount,
                                    "pnl": 0.0,
                                    "pnl_pct": 0.0,
                                    "is_whipsaw": False,
                                }
                            )

            # Track individual asset returns
            for t_idx in range(n_tickers):
                if valid_data[t_idx] and not np.isnan(closes[t_idx, d_idx]):
                    current_close = closes[t_idx, d_idx]
                    if not np.isnan(previous_closes[t_idx]):
                        # Calculate daily return
                        daily_return = (current_close - previous_closes[t_idx]) / previous_closes[
                            t_idx
                        ]
                        asset_returns[tickers[t_idx]].append(daily_return)
                    previous_closes[t_idx] = current_close

            # Calculate daily equity
            # Calculate position values only for valid data (non-NaN closes)
            valid_closes = np.where(valid_data, closes[:, d_idx], np.nan)
            positions_value = np.nansum(position_amounts * valid_closes)
            # If positions_value is NaN (all closes are NaN), use previous equity or cash only
            if np.isnan(positions_value):
                positions_value = 0.0
            equity_curve[d_idx] = cash + positions_value

        # Convert trades to DataFrame and optimize dtypes
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        if not trades_df.empty:
            trades_df = optimize_dtypes(trades_df)

        # Calculate metrics
        result = self._calculate_metrics_vectorized(
            equity_curve, sorted_dates, trades_df, asset_returns=asset_returns
        )
        result.strategy_name = strategy.name

        # Convert trades_df to Trade objects
        if len(trades_df) > 0:
            result.trades = [
                Trade(
                    ticker=row["ticker"],
                    entry_date=row["entry_date"],
                    entry_price=row["entry_price"],
                    exit_date=row["exit_date"],
                    exit_price=row["exit_price"],
                    amount=row["amount"],
                    pnl=row["pnl"],
                    pnl_pct=row["pnl_pct"],
                    is_whipsaw=row["is_whipsaw"],
                )
                for _, row in trades_df.iterrows()
            ]

        return result


# Alias for backward compatibility
BacktestEngine = VectorizedBacktestEngine


def run_backtest(
    strategy: Strategy,
    tickers: list[str],
    interval: str = "day",
    data_dir: Path | None = None,
    config: BacktestConfig | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> BacktestResult:
    """
    Convenience function to run backtest.

    Automatically collects missing data if files don't exist or are insufficient.

    Args:
        strategy: Trading strategy
        tickers: List of tickers (e.g., ["KRW-BTC", "KRW-ETH"])
        interval: Data interval (e.g., "day", "minute240")
        data_dir: Data directory (defaults to data/raw/)
        config: Backtest configuration
        start_date: Start date for filtering data (inclusive)
        end_date: End date for filtering data (inclusive)

    Returns:
        BacktestResult

    Raises:
        FileNotFoundError: If data collection fails for all tickers
    """
    data_dir = data_dir or RAW_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build data file paths
    data_files = {ticker: data_dir / f"{ticker}_{interval}.parquet" for ticker in tickers}

    # Check which files are missing or need updates
    missing_tickers = []
    for ticker, filepath in data_files.items():
        if not filepath.exists():
            missing_tickers.append(ticker)
            logger.info(f"Data file not found for {ticker}, will collect...")
        else:
            # Check if file is too old (older than 1 day) or has insufficient data
            try:
                from datetime import datetime, timedelta

                file_age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
                if file_age > timedelta(days=1):
                    missing_tickers.append(ticker)
                    logger.info(f"Data file for {ticker} is older than 1 day, will update...")
                else:
                    # Check if file has data
                    try:
                        df = pd.read_parquet(filepath)
                        if df.empty or len(df) < 10:  # Require at least 10 rows
                            missing_tickers.append(ticker)
                            logger.info(
                                f"Data file for {ticker} has insufficient data, will collect..."
                            )
                    except Exception:
                        missing_tickers.append(ticker)
                        logger.info(f"Data file for {ticker} is corrupted, will re-collect...")
            except Exception as e:
                logger.warning(f"Error checking data file for {ticker}: {e}, will collect...")
                missing_tickers.append(ticker)

    # Collect missing data
    if missing_tickers:
        logger.info(f"Collecting data for {len(missing_tickers)} ticker(s): {missing_tickers}")
        collector = DataCollectorFactory.create(data_dir=data_dir)

        # Map interval string to Interval type
        interval_map: dict[str, Interval] = {
            "minute240": "minute240",
            "day": "day",
            "week": "week",
        }

        interval_type = interval_map.get(interval, "day")  # type: ignore

        for ticker in missing_tickers:
            try:
                collector.collect(ticker, interval_type, full_refresh=False)
                logger.info(f"Successfully collected data for {ticker}")
            except Exception as e:
                logger.error(f"Failed to collect data for {ticker}: {e}")

    # Filter to existing files after collection
    data_files = {k: v for k, v in data_files.items() if v.exists()}

    if not data_files:
        raise FileNotFoundError(
            f"No data files found or collected in {data_dir} for tickers: {tickers}, interval: {interval}"
        )

    engine = VectorizedBacktestEngine(config)
    result = engine.run(strategy, data_files, start_date=start_date, end_date=end_date)
    result.interval = interval  # Store interval for risk metrics display
    return result
