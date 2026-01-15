"""
Signal combination methods.

Provides various methods to combine alpha signals.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CombinationMethod(str, Enum):
    """Signal combination methods."""

    EQUAL = "equal"  # Equal weights
    IC_WEIGHTED = "ic_weighted"  # Weight by Information Coefficient
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Adjust by signal volatility
    REGIME_BASED = "regime_based"  # Different weights for different regimes
    MACHINE_LEARNING = "machine_learning"  # ML-based combination


@dataclass
class CombinationResult:
    """Result of signal combination."""

    combined_signal: pd.Series
    weights_used: dict[str, float]
    method: CombinationMethod


class SignalCombiner:
    """
    Combines multiple alpha signals.

    Provides various combination methods to create a unified signal.

    Example:
        >>> combiner = SignalCombiner(method="ic_weighted")
        >>> combined = combiner.combine(
        ...     signals={"momentum": mom_signal, "value": val_signal},
        ...     ic_values={"momentum": 0.05, "value": 0.03},
        ... )
    """

    def __init__(
        self,
        method: CombinationMethod = CombinationMethod.EQUAL,
        min_weight: float = 0.05,
        max_weight: float = 0.50,
    ) -> None:
        """
        Initialize signal combiner.

        Args:
            method: Combination method
            min_weight: Minimum weight per signal
            max_weight: Maximum weight per signal
        """
        self.method = method
        self.min_weight = min_weight
        self.max_weight = max_weight

    def combine(
        self,
        signals: dict[str, pd.Series],
        ic_values: dict[str, float] | None = None,
        volatilities: dict[str, float] | None = None,
        regime: str | None = None,
    ) -> CombinationResult:
        """
        Combine multiple signals.

        Args:
            signals: Dict of signal_name -> signal Series
            ic_values: Information Coefficient per signal (for IC weighting)
            volatilities: Signal volatilities (for vol adjustment)
            regime: Current market regime (for regime-based)

        Returns:
            CombinationResult with combined signal
        """
        if not signals:
            return CombinationResult(
                combined_signal=pd.Series(dtype=float),
                weights_used={},
                method=self.method,
            )

        if self.method == CombinationMethod.EQUAL:
            weights = self._equal_weights(signals)
        elif self.method == CombinationMethod.IC_WEIGHTED:
            weights = self._ic_weights(signals, ic_values or {})
        elif self.method == CombinationMethod.VOLATILITY_ADJUSTED:
            weights = self._volatility_weights(signals, volatilities or {})
        elif self.method == CombinationMethod.REGIME_BASED:
            weights = self._regime_weights(signals, regime)
        else:
            weights = self._equal_weights(signals)

        # Apply constraints
        weights = self._apply_constraints(weights)

        # Combine
        combined = self._weighted_sum(signals, weights)

        return CombinationResult(
            combined_signal=combined,
            weights_used=weights,
            method=self.method,
        )

    def _equal_weights(self, signals: dict[str, pd.Series]) -> dict[str, float]:
        """Equal weights for all signals."""
        n = len(signals)
        return {name: 1.0 / n for name in signals}

    def _ic_weights(
        self,
        signals: dict[str, pd.Series],
        ic_values: dict[str, float],
    ) -> dict[str, float]:
        """Weight signals by Information Coefficient."""
        # Use absolute IC (predictive power regardless of direction)
        abs_ic = {name: abs(ic_values.get(name, 0)) for name in signals}

        total_ic = sum(abs_ic.values())
        if total_ic == 0:
            return self._equal_weights(signals)

        return {name: ic / total_ic for name, ic in abs_ic.items()}

    def _volatility_weights(
        self,
        signals: dict[str, pd.Series],
        volatilities: dict[str, float],
    ) -> dict[str, float]:
        """Weight signals inversely to their volatility."""
        # Lower volatility signals get higher weight
        inv_vol = {}
        for name in signals:
            vol = volatilities.get(name, 1.0)
            inv_vol[name] = 1.0 / vol if vol > 0 else 1.0

        total_inv_vol = sum(inv_vol.values())
        if total_inv_vol == 0:
            return self._equal_weights(signals)

        return {name: iv / total_inv_vol for name, iv in inv_vol.items()}

    def _regime_weights(
        self,
        signals: dict[str, pd.Series],
        regime: str | None,
    ) -> dict[str, float]:
        """Weights based on market regime."""
        # Default regime weights
        regime_configs = {
            "trending": {"momentum": 0.5, "breakout": 0.3, "technical": 0.2},
            "mean_reverting": {"mean_reversion": 0.5, "technical": 0.3, "value": 0.2},
            "volatile": {"technical": 0.4, "mean_reversion": 0.3, "momentum": 0.3},
            "default": None,  # Fall back to equal
        }

        regime_weight = regime_configs.get(regime, regime_configs["default"])

        if regime_weight is None:
            return self._equal_weights(signals)

        # Apply regime weights where applicable
        weights = {}
        remaining_weight = 1.0

        for name in signals:
            if name in regime_weight:
                weights[name] = regime_weight[name]
                remaining_weight -= regime_weight[name]
            else:
                weights[name] = 0

        # Distribute remaining weight equally to unassigned signals
        unassigned = [n for n in signals if weights.get(n, 0) == 0]
        if unassigned and remaining_weight > 0:
            per_signal = remaining_weight / len(unassigned)
            for name in unassigned:
                weights[name] = per_signal

        return weights

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply min/max weight constraints."""
        constrained = {}
        for name, weight in weights.items():
            constrained[name] = max(self.min_weight, min(self.max_weight, weight))

        # Renormalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v / total for k, v in constrained.items()}

        return constrained

    def _weighted_sum(
        self,
        signals: dict[str, pd.Series],
        weights: dict[str, float],
    ) -> pd.Series:
        """Calculate weighted sum of signals."""
        # Align all signals
        aligned = pd.DataFrame(signals)

        # Calculate weighted sum
        result = pd.Series(0.0, index=aligned.index)
        for name, weight in weights.items():
            if name in aligned.columns:
                result += aligned[name].fillna(0) * weight

        return result


def detect_regime(
    returns: pd.Series,
    lookback: int = 60,
) -> str:
    """
    Detect current market regime.

    Args:
        returns: Return series
        lookback: Lookback period

    Returns:
        Regime string: "trending", "mean_reverting", "volatile"
    """
    if len(returns) < lookback:
        return "default"

    recent = returns.iloc[-lookback:]

    # Calculate metrics
    volatility = recent.std() * np.sqrt(252)
    trend = recent.mean() * 252
    autocorr = recent.autocorr(lag=1)

    # Classify regime
    if volatility > 0.40:  # High volatility
        return "volatile"
    elif abs(autocorr) > 0.15 and abs(trend) > 0.10:  # Trending
        return "trending"
    elif autocorr < -0.10:  # Mean reverting
        return "mean_reverting"
    else:
        return "default"


__all__ = [
    "SignalCombiner",
    "CombinationMethod",
    "CombinationResult",
    "detect_regime",
]
