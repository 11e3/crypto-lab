"""ML Regime detection for VBO strategies.

Provides regime classification using a pre-trained ML model to
determine market state (BULL_TREND vs NOT_BULL). Used as a market
filter instead of simple BTC MA20.

Ported from bt framework's regime module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RegimeModelLoader:
    """Singleton loader for regime classification model.

    Loads the ML model once and caches it for reuse.
    """

    _instance: RegimeModelLoader | None = None
    _model: dict[str, Any] | None = None
    _model_path: str | None = None

    def __new__(cls) -> RegimeModelLoader:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str | Path) -> dict[str, Any]:
        """Load regime classifier model from joblib file.

        Args:
            model_path: Path to the model file (.joblib)

        Returns:
            Dict containing model, scaler, label_encoder, feature_names

        Raises:
            ImportError: If joblib is not installed
            FileNotFoundError: If model file does not exist
        """
        model_path_str = str(model_path)
        if self._model is not None and self._model_path == model_path_str:
            return self._model

        try:
            import joblib  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "ML dependencies not installed. Install with: "
                "pip install joblib scikit-learn"
            ) from e

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self._model = joblib.load(path)
        self._model_path = model_path_str
        logger.info(f"Loaded regime model from {path}")
        return self._model

    def get_model(self) -> dict[str, Any] | None:
        """Get cached model if loaded."""
        return self._model

    def clear_cache(self) -> None:
        """Clear cached model."""
        self._model = None
        self._model_path = None


def get_regime_model_loader() -> RegimeModelLoader:
    """Get singleton instance of RegimeModelLoader."""
    return RegimeModelLoader()


def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 5 features for regime classification.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with 5 features:
        - return_20d: 20-day return (momentum)
        - volatility: 20-day rolling volatility
        - rsi: 14-day RSI
        - ma_alignment: MA trend alignment score (-2 to +2)
        - volume_ratio_20: Volume vs 20-day average
    """
    result = pd.DataFrame(index=df.index)

    # 1. 20-day return
    result["return_20d"] = df["close"].pct_change(20)

    # 2. 20-day rolling volatility
    daily_returns = df["close"].pct_change()
    result["volatility"] = daily_returns.rolling(window=20).std()

    # 3. 14-day RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    result["rsi"] = 100 - (100 / (1 + rs))

    # 4. MA alignment score
    ma_5 = df["close"].rolling(window=5).mean()
    ma_20 = df["close"].rolling(window=20).mean()
    ma_60 = df["close"].rolling(window=60).mean()

    alignment = pd.Series(0, index=df.index, dtype=float)
    alignment = alignment + (ma_5 > ma_20).astype(int)
    alignment = alignment + (ma_20 > ma_60).astype(int)
    alignment = alignment - (ma_5 < ma_20).astype(int)
    alignment = alignment - (ma_20 < ma_60).astype(int)
    result["ma_alignment"] = alignment

    # 5. Volume ratio
    volume_ma_20 = df["volume"].rolling(window=20).mean()
    result["volume_ratio_20"] = df["volume"] / volume_ma_20

    return result


def predict_regime(clf_data: dict[str, Any], ohlcv_df: pd.DataFrame) -> pd.Series:
    """Predict regime from OHLCV data.

    Args:
        clf_data: Loaded classifier dict from joblib
        ohlcv_df: OHLCV DataFrame

    Returns:
        Series with regime predictions ("BULL_TREND" or "NOT_BULL")
    """
    features = calculate_regime_features(ohlcv_df)
    features = features.dropna()

    if len(features) == 0:
        raise ValueError("Not enough data to calculate features (need at least 60 rows)")

    x_scaled = clf_data["scaler"].transform(features[clf_data["feature_names"]])
    pred_encoded = clf_data["model"].predict(x_scaled)
    predictions = clf_data["label_encoder"].inverse_transform(pred_encoded)

    return pd.Series(predictions, index=features.index, name="regime")


__all__ = [
    "RegimeModelLoader",
    "calculate_regime_features",
    "get_regime_model_loader",
    "predict_regime",
]
