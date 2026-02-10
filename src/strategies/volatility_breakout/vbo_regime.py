"""VBO Regime Strategy with ML-based market filter.

Uses a pre-trained ML model to classify BTC market regime
(BULL_TREND vs NOT_BULL) instead of simple BTC MA20 filter.

Ported from bt framework's VBORegimeStrategy.
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
    predict_regime,
)
from src.strategies.volatility_breakout.vbo_indicators import calculate_vbo_indicators
from src.strategies.volatility_breakout.vbo_portfolio import _load_btc_data
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default model directory
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
DEFAULT_MODEL_NAME = "regime_classifier_xgb_ultra5.joblib"


class RegimeFilterCondition(Condition):
    """Entry filter: only trade when BTC regime is BULL_TREND.

    In vectorized mode, expects 'regime_bull' column in DataFrame.
    """

    def __init__(self, name: str = "RegimeFilter") -> None:
        super().__init__(name)

    def evaluate(
        self,
        current: Any,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        regime_bull = indicators.get("regime_bull")
        if regime_bull is None:
            return False
        return bool(regime_bull)


class RegimeExitCondition(Condition):
    """Exit filter: sell when BTC regime is NOT_BULL.

    In vectorized mode, expects 'regime_bear' column in DataFrame.
    """

    def __init__(self, name: str = "RegimeExit") -> None:
        super().__init__(name)

    def evaluate(
        self,
        current: Any,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        regime_bear = indicators.get("regime_bear")
        if regime_bear is None:
            return False
        return bool(regime_bear)


class VBORegime(Strategy):
    """VBO with ML regime model market filter.

    Entry conditions (AND):
    - Breakout: high >= target
    - SMA filter: target > SMA(ma_short)
    - Regime filter: BTC regime == BULL_TREND

    Exit conditions (OR):
    - Price below SMA: close < SMA(ma_short)
    - Regime exit: BTC regime != BULL_TREND

    Args:
        name: Strategy name
        ma_short: Short MA period
        noise_ratio: K-factor for VBO breakout
        model_path: Path to regime model (.joblib)
        btc_data: Pre-loaded BTC DataFrame
        data_dir: Directory containing parquet files
        interval: Data interval
    """

    def __init__(
        self,
        name: str = "VBORegime",
        ma_short: int = 5,
        noise_ratio: float = 0.5,
        model_path: str | Path | None = None,
        btc_data: pd.DataFrame | None = None,
        data_dir: Path | None = None,
        interval: str = "day",
        **_kwargs: Any,
    ) -> None:
        self.ma_short = ma_short
        self.noise_ratio = noise_ratio
        self.model_path = str(model_path) if model_path else str(MODELS_DIR / DEFAULT_MODEL_NAME)
        self._btc_data = btc_data
        self._data_dir = data_dir or Path(__file__).resolve().parents[3] / "data" / "raw"
        self._interval = interval

        entry_conditions: list[Condition] = [
            BreakoutCondition(),
            SMABreakoutCondition(),
            RegimeFilterCondition(),
        ]
        exit_conditions: list[Condition] = [
            PriceBelowSMACondition(),
            RegimeExitCondition(),
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

    def _predict_btc_regime(self, btc_df: pd.DataFrame) -> pd.Series | None:
        """Predict regime for BTC data."""
        loader = get_regime_model_loader()
        try:
            model = loader.load_model(self.model_path)
        except (ImportError, FileNotFoundError) as e:
            logger.warning(f"Cannot load regime model: {e}")
            return None

        try:
            return predict_regime(model, btc_df)
        except (ValueError, KeyError) as e:
            logger.warning(f"Regime prediction failed: {e}")
            return None

    def required_indicators(self) -> list[str]:
        return [
            "noise",
            "sma",
            "target",
            "prev_range",
            "regime_bull",
            "regime_bear",
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VBO indicators + ML regime filter columns."""
        df = calculate_vbo_indicators(
            df,
            sma_period=self.ma_short,
            trend_sma_period=self.ma_short * 2,
            short_noise_period=self.ma_short,
            long_noise_period=self.ma_short * 2,
            exclude_current=True,
        )

        # Add regime prediction columns
        btc_df = self._get_btc_data()
        if btc_df is not None:
            regime_series = self._predict_btc_regime(btc_df)
            if regime_series is not None:
                # Reindex to match target DataFrame and shift by 1 day
                regime_aligned = regime_series.reindex(df.index, method="ffill")
                prev_regime = regime_aligned.shift(1)

                df["regime_bull"] = prev_regime == "BULL_TREND"
                df["regime_bear"] = prev_regime != "BULL_TREND"
            else:
                df["regime_bull"] = True
                df["regime_bear"] = False
        else:
            # No BTC data: don't filter
            df["regime_bull"] = True
            df["regime_bear"] = False

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry/exit signals with ML regime filter."""
        df = df.copy()

        # Entry: breakout AND sma filter AND regime bull
        entry_breakout = df["high"] >= df["target"]
        entry_sma = df["target"] > df["sma"]
        entry_regime = df["regime_bull"].fillna(False)

        df["entry_signal"] = entry_breakout & entry_sma & entry_regime

        # Exit: price below sma OR regime bear (OR logic)
        exit_sma = df["close"] < df["sma"]
        exit_regime = df["regime_bear"].fillna(False)

        df["exit_signal"] = exit_sma | exit_regime

        return df


__all__ = [
    "VBORegime",
    "RegimeFilterCondition",
    "RegimeExitCondition",
]
