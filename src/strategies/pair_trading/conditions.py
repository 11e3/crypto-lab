"""
Entry and exit conditions for Pair Trading strategy.

These conditions use spread and Z-score to identify mean reversion
opportunities between two correlated assets.
"""

import pandas as pd

from src.strategies.base import OHLCV, Condition


class SpreadZScoreCondition(Condition):
    """
    Entry condition: Spread Z-score exceeds threshold.

    Indicates spread has deviated significantly from mean.
    """

    def __init__(
        self,
        z_score_key: str = "z_score",
        entry_threshold: float = 2.0,
        name: str = "SpreadZScore",
    ) -> None:
        """
        Initialize spread Z-score condition.

        Args:
            z_score_key: Key for Z-score value in indicators dict
            entry_threshold: Z-score threshold for entry (default 2.0)
            name: Condition name
        """
        super().__init__(name)
        self.z_score_key = z_score_key
        self.entry_threshold = entry_threshold

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if Z-score exceeds threshold."""
        z_score = indicators.get(self.z_score_key)

        if z_score is None:
            return False

        # Enter when spread deviates significantly (Z-score > threshold)
        return abs(z_score) >= self.entry_threshold


class SpreadMeanReversionCondition(Condition):
    """
    Exit condition: Spread has reverted to mean (Z-score near zero).

    Indicates spread has returned to its historical average.
    """

    def __init__(
        self,
        z_score_key: str = "z_score",
        exit_threshold: float = 0.5,
        name: str = "SpreadMeanReversion",
    ) -> None:
        """
        Initialize spread mean reversion condition.

        Args:
            z_score_key: Key for Z-score value in indicators dict
            exit_threshold: Z-score threshold for exit (default 0.5)
            name: Condition name
        """
        super().__init__(name)
        self.z_score_key = z_score_key
        self.exit_threshold = exit_threshold

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if Z-score has reverted to mean."""
        z_score = indicators.get(self.z_score_key)

        if z_score is None:
            return False

        # Exit when spread has reverted (Z-score near zero)
        return abs(z_score) <= self.exit_threshold


class SpreadDeviationCondition(Condition):
    """
    Condition: Spread deviation from mean exceeds percentage threshold.

    Alternative to Z-score for entry/exit signals.
    """

    def __init__(
        self,
        spread_key: str = "spread",
        spread_mean_key: str = "spread_mean",
        min_deviation_pct: float = 0.02,
        name: str = "SpreadDeviation",
    ) -> None:
        """
        Initialize spread deviation condition.

        Args:
            spread_key: Key for spread value in indicators dict
            spread_mean_key: Key for spread mean value in indicators dict
            min_deviation_pct: Minimum deviation percentage (default 2%)
            name: Condition name
        """
        super().__init__(name)
        self.spread_key = spread_key
        self.spread_mean_key = spread_mean_key
        self.min_deviation_pct = min_deviation_pct

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Check if spread has deviated enough from mean."""
        spread = indicators.get(self.spread_key)
        spread_mean = indicators.get(self.spread_mean_key)

        if spread is None or spread_mean is None or spread_mean == 0:
            return False

        # Calculate deviation percentage
        deviation_pct = abs(spread - spread_mean) / abs(spread_mean)
        return deviation_pct >= self.min_deviation_pct
