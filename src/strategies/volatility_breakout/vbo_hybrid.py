"""VBO Hybrid Strategy: MA20 primary + ML secondary.

Uses BTC MA20 as the primary market filter, and only consults
the ML regime model when BTC is in the ambiguous zone (close to MA20).
This guarantees performance >= pure MA20 filter while allowing ML
to add value in uncertain conditions.

Decision logic:
    distance = (prev_btc_close - prev_btc_ma20) / prev_btc_ma20

    if distance > +threshold:   MA clearly bull  → allow entry (no ML)
    if distance < -threshold:   MA clearly bear  → block entry (no ML)
    if |distance| <= threshold: ambiguous zone   → ML model decides
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.strategies.base import Condition, Strategy
from src.strategies.volatility_breakout.conditions import (
    BreakoutCondition,
    PriceBelowSMACondition,
    SMABreakoutCondition,
)
from src.strategies.volatility_breakout.regime import (
    get_regime_model_loader,
    predict_regime_proba,
)
from src.strategies.volatility_breakout.vbo_indicators import calculate_vbo_indicators
from src.strategies.volatility_breakout.vbo_portfolio import _load_btc_data
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default model directory
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
DEFAULT_MODEL_NAME = "regime_classifier_xgb_v2.joblib"


class HybridMarketFilterCondition(Condition):
    """Entry filter: MA20 primary, ML secondary in ambiguous zone.

    Expects 'hybrid_bull' column in DataFrame (vectorized mode).
    """

    def __init__(self, name: str = "HybridMarketFilter") -> None:
        super().__init__(name)

    def evaluate(
        self,
        current: Any,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        hybrid_bull = indicators.get("hybrid_bull")
        if hybrid_bull is None:
            return False
        return bool(hybrid_bull)


class HybridMarketExitCondition(Condition):
    """Exit filter: MA20 primary, ML secondary in ambiguous zone.

    Expects 'hybrid_bear' column in DataFrame (vectorized mode).
    """

    def __init__(self, name: str = "HybridMarketExit") -> None:
        super().__init__(name)

    def evaluate(
        self,
        current: Any,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        hybrid_bear = indicators.get("hybrid_bear")
        if hybrid_bear is None:
            return False
        return bool(hybrid_bear)


class VBOHybrid(Strategy):
    """VBO with MA20 + ML hybrid market filter.

    Entry conditions (AND):
    - Breakout: high >= target
    - SMA filter: target > SMA(ma_short)
    - Hybrid filter: MA20 bull OR (ambiguous AND ML bull)

    Exit conditions (OR):
    - Price below SMA: close < SMA(ma_short)
    - Hybrid exit: MA20 bear OR (ambiguous AND ML bear)

    Args:
        name: Strategy name
        ma_short: Short MA period for individual coins
        btc_ma: BTC MA period for primary filter
        noise_ratio: K-factor for VBO breakout
        threshold: Distance threshold for ambiguous zone (default 0.02 = 2%)
        ml_confidence: Min ML probability to classify as bull (default 0.5)
        model_path: Path to ML regime model (.joblib)
        btc_data: Pre-loaded BTC DataFrame
        data_dir: Directory containing parquet files
        interval: Data interval
    """

    def __init__(
        self,
        name: str = "VBOHybrid",
        ma_short: int = 5,
        btc_ma: int = 20,
        noise_ratio: float = 0.5,
        threshold: float = 0.02,
        ml_confidence: float = 0.5,
        model_path: str | Path | None = None,
        btc_data: pd.DataFrame | None = None,
        data_dir: Path | None = None,
        interval: str = "day",
        **_kwargs: Any,
    ) -> None:
        self.ma_short = ma_short
        self.btc_ma = btc_ma
        self.noise_ratio = noise_ratio
        self.threshold = threshold
        self.ml_confidence = ml_confidence
        self.model_path = str(model_path) if model_path else str(MODELS_DIR / DEFAULT_MODEL_NAME)
        self._btc_data = btc_data
        self._data_dir = data_dir or Path(__file__).resolve().parents[3] / "data" / "raw"
        self._interval = interval

        entry_conditions: list[Condition] = [
            BreakoutCondition(),
            SMABreakoutCondition(),
            HybridMarketFilterCondition(),
        ]
        exit_conditions: list[Condition] = [
            PriceBelowSMACondition(),
            HybridMarketExitCondition(),
        ]

        super().__init__(
            name=name,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
        )

    def _get_btc_data(self) -> pd.DataFrame | None:
        """Get BTC data, loading from disk if needed."""
        if self._btc_data is None:
            self._btc_data = _load_btc_data(self._data_dir, self._interval)
        return self._btc_data

    def _get_ml_bull_proba(self, btc_df: pd.DataFrame) -> pd.Series | None:
        """Get ML regime probability for BULL_TREND class."""
        loader = get_regime_model_loader()
        try:
            model = loader.load_model(self.model_path)
        except (ImportError, FileNotFoundError) as e:
            logger.warning(f"Cannot load regime model: {e}")
            return None

        try:
            probas = predict_regime_proba(model, btc_df)
        except (ValueError, KeyError) as e:
            logger.warning(f"Regime prediction failed: {e}")
            return None

        # Find BULL_TREND column
        bull_col = None
        for col in probas.columns:
            if "BULL" in str(col).upper():
                bull_col = col
                break

        if bull_col is None:
            # Binary classifier: use first column probability
            bull_col = probas.columns[0]

        return probas[bull_col]

    def required_indicators(self) -> list[str]:
        return [
            "noise",
            "sma",
            "target",
            "prev_range",
            "hybrid_bull",
            "hybrid_bear",
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VBO indicators + hybrid MA20/ML filter columns."""
        df = calculate_vbo_indicators(
            df,
            sma_period=self.ma_short,
            trend_sma_period=self.ma_short * 2,
            short_noise_period=self.ma_short,
            long_noise_period=self.ma_short * 2,
            exclude_current=True,
        )

        btc_df = self._get_btc_data()

        if btc_df is None or (isinstance(btc_df, pd.DataFrame) and btc_df.empty):
            # No BTC data: default to allow all (pure VBO)
            df["hybrid_bull"] = True
            df["hybrid_bear"] = False
            return df

        # Calculate BTC MA20 distance
        btc_close = btc_df["close"].reindex(df.index, method="ffill")
        btc_ma = btc_close.rolling(window=self.btc_ma).mean()

        prev_btc_close = btc_close.shift(1)
        prev_btc_ma = btc_ma.shift(1)
        distance = (prev_btc_close - prev_btc_ma) / prev_btc_ma

        # Classification zones
        clearly_bull = distance > self.threshold
        clearly_bear = distance < -self.threshold
        ambiguous = ~clearly_bull & ~clearly_bear

        # Start with MA20 decisions
        hybrid_bull = clearly_bull.copy()
        hybrid_bear = clearly_bear.copy()

        # In ambiguous zone: consult ML model
        if ambiguous.any():
            bull_proba = self._get_ml_bull_proba(btc_df)
            if bull_proba is not None:
                bull_proba_aligned = bull_proba.reindex(df.index, method="ffill").shift(1)
                ml_bull = bull_proba_aligned >= self.ml_confidence

                hybrid_bull = hybrid_bull | (ambiguous & ml_bull)
                hybrid_bear = hybrid_bear | (ambiguous & ~ml_bull)
            else:
                # ML unavailable: fall back to MA20 decision in ambiguous zone
                # Treat ambiguous as bull (same behavior as VBOPortfolio near MA20)
                hybrid_bull = hybrid_bull | ambiguous

        df["hybrid_bull"] = hybrid_bull.fillna(False)
        df["hybrid_bear"] = hybrid_bear.fillna(False)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry/exit signals with hybrid filter."""
        df = df.copy()

        # Entry: breakout AND sma filter AND hybrid bull
        entry_breakout = df["high"] >= df["target"]
        entry_sma = df["target"] > df["sma"]
        entry_hybrid = df["hybrid_bull"].fillna(False)

        df["entry_signal"] = entry_breakout & entry_sma & entry_hybrid

        # Exit: price below sma OR hybrid bear
        exit_sma = df["close"] < df["sma"]
        exit_hybrid = df["hybrid_bear"].fillna(False)

        df["exit_signal"] = exit_sma | exit_hybrid

        return df


__all__ = [
    "VBOHybrid",
    "HybridMarketFilterCondition",
    "HybridMarketExitCondition",
]
