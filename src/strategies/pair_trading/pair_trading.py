"""
Pair Trading Strategy.

Implements statistical arbitrage using spread and Z-score between two correlated assets.
Assumes spread will revert to its mean after deviating.
"""

import pandas as pd

from src.strategies.base import Condition, Strategy
from src.strategies.pair_trading.conditions import (
    SpreadMeanReversionCondition,
    SpreadZScoreCondition,
)
from src.utils.indicators import sma


class PairTradingStrategy(Strategy):
    """
    Pair Trading Strategy.

    Default configuration:
    - Entry: Spread Z-score exceeds threshold (spread deviated from mean)
    - Exit: Spread Z-score reverts to mean (near zero)

    The strategy assumes the spread between two assets will revert to its
    historical mean after deviating. Requires two tickers to be specified.

    Note: This strategy requires special handling in backtest engine as it
    needs both tickers' data simultaneously to calculate spread.
    """

    def __init__(
        self,
        name: str = "PairTradingStrategy",
        lookback_period: int = 60,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        spread_type: str = "ratio",  # "ratio" or "difference"
        entry_conditions: list[Condition] | None = None,
        exit_conditions: list[Condition] | None = None,
        use_default_conditions: bool = True,
    ) -> None:
        """
        Initialize Pair Trading strategy.

        Args:
            name: Strategy name
            lookback_period: Period for spread mean/std calculation
            entry_z_score: Z-score threshold for entry (default 2.0)
            exit_z_score: Z-score threshold for exit (default 0.5)
            spread_type: Type of spread calculation ("ratio" or "difference")
            entry_conditions: Custom entry conditions (optional)
            exit_conditions: Custom exit conditions (optional)
            use_default_conditions: Whether to add default conditions
        """
        # Store parameters
        self.lookback_period = lookback_period
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.spread_type = spread_type

        # Build default conditions
        default_entry: list[Condition] = []
        default_exit: list[Condition] = []

        if use_default_conditions:
            default_entry = [
                SpreadZScoreCondition(
                    z_score_key="z_score", entry_threshold=entry_z_score
                ),
            ]
            default_exit = [
                SpreadMeanReversionCondition(
                    z_score_key="z_score", exit_threshold=exit_z_score
                ),
            ]

        # Merge with custom conditions
        all_entry = (entry_conditions or []) + default_entry
        all_exit = (exit_conditions or []) + default_exit

        super().__init__(
            name=name,
            entry_conditions=all_entry,
            exit_conditions=all_exit,
        )

    def required_indicators(self) -> list[str]:
        """Return list of required indicators."""
        return [
            "spread",
            "spread_mean",
            "spread_std",
            "z_score",
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pair trading indicators.

        Note: This method expects df to have data for both assets.
        In practice, the backtest engine should merge data from both tickers
        before calling this method.

        Args:
            df: DataFrame with OHLCV data for both assets
                 Expected columns: close_asset1, close_asset2 (or similar)

        Returns:
            DataFrame with spread and Z-score indicators
        """
        df = df.copy()

        # For pair trading, we need to identify which columns belong to which asset
        # This is a simplified version - in practice, the engine should handle merging
        if "close_asset1" in df.columns and "close_asset2" in df.columns:
            close1 = df["close_asset1"]
            close2 = df["close_asset2"]
        elif len([c for c in df.columns if c.startswith("close")]) == 2:
            # Assume two close columns exist
            close_cols = [c for c in df.columns if c.startswith("close")]
            close1 = df[close_cols[0]]
            close2 = df[close_cols[1]]
        else:
            # Fallback: use first two close columns or create dummy data
            # This should not happen in normal usage
            if "close" in df.columns:
                # Single asset - cannot calculate spread
                df["spread"] = pd.Series(0.0, index=df.index)
                df["spread_mean"] = pd.Series(0.0, index=df.index)
                df["spread_std"] = pd.Series(1.0, index=df.index)
                df["z_score"] = pd.Series(0.0, index=df.index)
                return df
            else:
                raise ValueError(
                    "Pair trading requires data from two assets. "
                    "Expected columns with close prices for both assets."
                )

        # Calculate spread
        if self.spread_type == "ratio":
            # Ratio spread: asset1 / asset2
            df["spread"] = close1 / close2
        else:
            # Difference spread: asset1 - asset2
            df["spread"] = close1 - close2

        # Calculate spread statistics
        df["spread_mean"] = sma(df["spread"], self.lookback_period)
        df["spread_std"] = (
            df["spread"].rolling(window=self.lookback_period, min_periods=self.lookback_period).std()
        )

        # Calculate Z-score
        # Z-score = (spread - mean) / std
        df["z_score"] = (df["spread"] - df["spread_mean"]) / df["spread_std"].replace(0, 1)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals using vectorized operations.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            DataFrame with 'entry_signal' and 'exit_signal' columns
        """
        df = df.copy()

        # Build entry signal based on configured conditions
        entry_signal = pd.Series(True, index=df.index)

        # Check each entry condition
        for condition in self.entry_conditions.conditions:
            if condition.name == "SpreadZScore":
                entry_threshold = getattr(condition, "entry_threshold", 2.0)
                entry_signal = entry_signal & (df["z_score"].abs() >= entry_threshold)

        # Build exit signal based on configured conditions
        exit_signal = pd.Series(False, index=df.index)

        for condition in self.exit_conditions.conditions:
            if condition.name == "SpreadMeanReversion":
                exit_threshold = getattr(condition, "exit_threshold", 0.5)
                exit_signal = exit_signal | (df["z_score"].abs() <= exit_threshold)

        df["entry_signal"] = entry_signal
        df["exit_signal"] = exit_signal

        return df

    def calculate_spread_for_pair(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate spread indicators for a pair of assets.

        This is a helper method that should be called by the backtest engine
        to merge data from two tickers before calculating indicators.

        Args:
            df1: DataFrame for first asset (ticker1)
            df2: DataFrame for second asset (ticker2)

        Returns:
            DataFrame with merged data and spread indicators
        """
        # Align indices (use intersection of dates)
        common_dates = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_dates].copy()
        df2_aligned = df2.loc[common_dates].copy()

        # Create merged dataframe
        merged_df = pd.DataFrame(index=common_dates)
        merged_df["close_asset1"] = df1_aligned["close"]
        merged_df["close_asset2"] = df2_aligned["close"]

        # Add other columns for reference (if available)
        if "open" in df1_aligned.columns:
            merged_df["open_asset1"] = df1_aligned["open"]
        if "high" in df1_aligned.columns:
            merged_df["high_asset1"] = df1_aligned["high"]
        if "low" in df1_aligned.columns:
            merged_df["low_asset1"] = df1_aligned["low"]
        if "volume" in df1_aligned.columns:
            merged_df["volume_asset1"] = df1_aligned["volume"]
        else:
            merged_df["volume_asset1"] = 0

        if "open" in df2_aligned.columns:
            merged_df["open_asset2"] = df2_aligned["open"]
        if "high" in df2_aligned.columns:
            merged_df["high_asset2"] = df2_aligned["high"]
        if "low" in df2_aligned.columns:
            merged_df["low_asset2"] = df2_aligned["low"]
        if "volume" in df2_aligned.columns:
            merged_df["volume_asset2"] = df2_aligned["volume"]
        else:
            merged_df["volume_asset2"] = 0

        # Calculate indicators
        return self.calculate_indicators(merged_df)
