"""
Alpha generation pipeline.

Orchestrates multiple alpha signals into a unified prediction.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from src.alpha.signals import AlphaSignal, SignalOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SignalWeight:
    """Weight configuration for a signal."""

    signal_name: str
    weight: float = 1.0
    active: bool = True

    # Dynamic weighting based on performance
    use_dynamic_weight: bool = False
    ic_decay: float = 0.95


@dataclass
class PipelineResult:
    """Result from alpha pipeline."""

    # Combined signals: ticker -> combined alpha (-1 to 1)
    combined_signals: dict[str, float]

    # Individual signal contributions
    signal_contributions: dict[str, dict[str, float]] = field(default_factory=dict)

    # Target weights (signals converted to portfolio weights)
    target_weights: dict[str, float] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    active_signals: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        data = {"combined": self.combined_signals}
        for signal_name, contributions in self.signal_contributions.items():
            data[signal_name] = contributions
        return pd.DataFrame(data)


class AlphaPipeline:
    """
    Alpha generation pipeline.

    Combines multiple alpha signals into unified predictions.

    Example:
        >>> from src.alpha.signals import MomentumSignal, MeanReversionSignal
        >>>
        >>> pipeline = AlphaPipeline(
        ...     signals=[
        ...         MomentumSignal(lookback=20),
        ...         MeanReversionSignal(lookback=60),
        ...     ],
        ...     weights=[
        ...         SignalWeight("momentum", weight=0.6),
        ...         SignalWeight("mean_reversion", weight=0.4),
        ...     ],
        ... )
        >>> result = pipeline.generate(prices)
        >>> weights = result.target_weights
    """

    def __init__(
        self,
        signals: list[AlphaSignal],
        weights: list[SignalWeight] | None = None,
        combination_method: Literal["weighted", "rank", "confidence"] = "weighted",
        long_only: bool = False,
        top_n: int | None = None,
        min_signal_threshold: float = 0.1,
    ) -> None:
        """
        Initialize pipeline.

        Args:
            signals: List of alpha signal generators
            weights: Weight configuration for each signal
            combination_method: How to combine signals
            long_only: Only generate long signals
            top_n: Limit to top N signals by strength
            min_signal_threshold: Minimum signal strength to include
        """
        self.signals = {s.name: s for s in signals}
        self.combination_method = combination_method
        self.long_only = long_only
        self.top_n = top_n
        self.min_signal_threshold = min_signal_threshold

        # Set up weights
        if weights is None:
            n = len(signals)
            self.weights = {
                s.name: SignalWeight(s.name, weight=1.0 / n)
                for s in signals
            }
        else:
            self.weights = {w.signal_name: w for w in weights}

        # IC tracking for dynamic weights
        self._ic_history: dict[str, list[float]] = {
            name: [] for name in self.signals
        }

    def generate(
        self,
        data: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> PipelineResult:
        """
        Generate combined alpha signals.

        Args:
            data: Price data DataFrame
            timestamp: Current timestamp

        Returns:
            PipelineResult with combined signals and target weights
        """
        timestamp = timestamp or datetime.now()

        # Generate individual signals
        signal_outputs: dict[str, SignalOutput] = {}
        for name, signal in self.signals.items():
            weight_config = self.weights.get(name)
            if weight_config and not weight_config.active:
                continue

            try:
                output = signal.generate(data, timestamp)
                signal_outputs[name] = output
            except Exception as e:
                logger.warning(f"Signal {name} failed: {e}")

        if not signal_outputs:
            return PipelineResult(
                combined_signals={},
                timestamp=timestamp,
            )

        # Get all tickers
        all_tickers = set()
        for output in signal_outputs.values():
            all_tickers.update(output.signals.keys())

        # Combine signals
        combined = self._combine_signals(signal_outputs, all_tickers)

        # Apply filters
        filtered = self._apply_filters(combined)

        # Convert to target weights
        target_weights = self._signals_to_weights(filtered)

        # Prepare contributions
        contributions = {
            name: output.signals
            for name, output in signal_outputs.items()
        }

        return PipelineResult(
            combined_signals=combined,
            signal_contributions=contributions,
            target_weights=target_weights,
            timestamp=timestamp,
            active_signals=list(signal_outputs.keys()),
        )

    def _combine_signals(
        self,
        outputs: dict[str, SignalOutput],
        tickers: set[str],
    ) -> dict[str, float]:
        """Combine individual signals."""
        combined = {}

        for ticker in tickers:
            if self.combination_method == "weighted":
                combined[ticker] = self._weighted_combine(outputs, ticker)
            elif self.combination_method == "rank":
                combined[ticker] = self._rank_combine(outputs, ticker)
            elif self.combination_method == "confidence":
                combined[ticker] = self._confidence_combine(outputs, ticker)
            else:
                combined[ticker] = self._weighted_combine(outputs, ticker)

        return combined

    def _weighted_combine(
        self,
        outputs: dict[str, SignalOutput],
        ticker: str,
    ) -> float:
        """Weighted average of signals."""
        total_weight = 0
        weighted_sum = 0

        for name, output in outputs.items():
            weight_config = self.weights.get(name)
            weight = weight_config.weight if weight_config else 1.0

            if ticker in output.signals:
                weighted_sum += output.signals[ticker] * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def _rank_combine(
        self,
        outputs: dict[str, SignalOutput],
        ticker: str,
    ) -> float:
        """Combine using rank averaging."""
        # Get signals for this ticker
        ticker_signals = []
        for name, output in outputs.items():
            if ticker in output.signals:
                ticker_signals.append(output.signals[ticker])

        if not ticker_signals:
            return 0

        return np.mean(ticker_signals)

    def _confidence_combine(
        self,
        outputs: dict[str, SignalOutput],
        ticker: str,
    ) -> float:
        """Combine weighted by confidence."""
        total_confidence = 0
        weighted_sum = 0

        for name, output in outputs.items():
            if ticker in output.signals:
                signal = output.signals[ticker]
                confidence = output.confidence.get(ticker, 0.5)

                weighted_sum += signal * confidence
                total_confidence += confidence

        return weighted_sum / total_confidence if total_confidence > 0 else 0

    def _apply_filters(
        self,
        signals: dict[str, float],
    ) -> dict[str, float]:
        """Apply filters to signals."""
        filtered = {}

        for ticker, signal in signals.items():
            # Long only filter
            if self.long_only and signal < 0:
                continue

            # Minimum threshold
            if abs(signal) < self.min_signal_threshold:
                continue

            filtered[ticker] = signal

        # Top N filter
        if self.top_n and len(filtered) > self.top_n:
            sorted_signals = sorted(
                filtered.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            filtered = dict(sorted_signals[:self.top_n])

        return filtered

    def _signals_to_weights(
        self,
        signals: dict[str, float],
    ) -> dict[str, float]:
        """Convert signals to portfolio weights."""
        if not signals:
            return {}

        # Separate long and short signals
        long_signals = {k: v for k, v in signals.items() if v > 0}
        short_signals = {k: abs(v) for k, v in signals.items() if v < 0}

        weights = {}

        # Normalize long positions
        long_total = sum(long_signals.values())
        if long_total > 0:
            for ticker, signal in long_signals.items():
                weights[ticker] = signal / long_total * 0.5  # 50% long exposure

        # Normalize short positions (if not long only)
        if not self.long_only:
            short_total = sum(short_signals.values())
            if short_total > 0:
                for ticker, signal in short_signals.items():
                    weights[ticker] = -signal / short_total * 0.5  # 50% short exposure

        # If long only, scale to 100%
        if self.long_only:
            weights = {k: v * 2 for k, v in weights.items()}

        return weights

    def update_ic(
        self,
        signal_name: str,
        ic: float,
    ) -> None:
        """Update IC history for dynamic weighting."""
        if signal_name in self._ic_history:
            self._ic_history[signal_name].append(ic)

            # Limit history
            if len(self._ic_history[signal_name]) > 100:
                self._ic_history[signal_name] = self._ic_history[signal_name][-100:]

    def get_signal_stats(self) -> pd.DataFrame:
        """Get signal statistics."""
        stats = []
        for name in self.signals:
            weight = self.weights.get(name)
            ic_history = self._ic_history.get(name, [])

            stats.append({
                "signal": name,
                "weight": weight.weight if weight else 0,
                "active": weight.active if weight else True,
                "avg_ic": np.mean(ic_history) if ic_history else 0,
                "ic_std": np.std(ic_history) if ic_history else 0,
            })

        return pd.DataFrame(stats)


def create_default_pipeline() -> AlphaPipeline:
    """Create default alpha pipeline with common signals."""
    from src.alpha.signals import (
        MomentumSignal,
        MeanReversionSignal,
        TechnicalSignal,
        BreakoutSignal,
    )

    signals = [
        MomentumSignal(lookback=20),
        MeanReversionSignal(lookback=60),
        TechnicalSignal(),
        BreakoutSignal(lookback=20),
    ]

    weights = [
        SignalWeight("momentum", weight=0.30),
        SignalWeight("mean_reversion", weight=0.25),
        SignalWeight("technical", weight=0.25),
        SignalWeight("breakout", weight=0.20),
    ]

    return AlphaPipeline(
        signals=signals,
        weights=weights,
        long_only=True,
        top_n=10,
    )


__all__ = [
    "AlphaPipeline",
    "SignalWeight",
    "PipelineResult",
    "create_default_pipeline",
]
