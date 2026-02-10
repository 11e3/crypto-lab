"""Tests for VBORegime strategy and regime detection module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.volatility_breakout.regime import (
    RegimeModelLoader,
    calculate_regime_features,
    predict_regime,
    predict_regime_proba,
)
from src.strategies.volatility_breakout.vbo_regime import (
    RegimeExitCondition,
    RegimeFilterCondition,
    VBORegime,
)


@pytest.fixture()
def ohlcv_data() -> pd.DataFrame:
    """Generate OHLCV data with enough history for regime features."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    prices = 50000 + np.cumsum(np.random.randn(n) * 500)
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.97,
            "close": prices,
            "volume": np.random.rand(n) * 10000 + 1000,
        },
        index=dates,
    )


class TestRegimeModelLoader:
    """Test RegimeModelLoader singleton."""

    def test_singleton_pattern(self) -> None:
        """Two instances should be the same object."""
        loader1 = RegimeModelLoader()
        loader2 = RegimeModelLoader()
        assert loader1 is loader2

    def test_load_model_with_invalid_path(self) -> None:
        """Should raise FileNotFoundError for missing model."""
        loader = RegimeModelLoader()
        # Reset internal state
        RegimeModelLoader._model = None
        RegimeModelLoader._model_path = None

        with pytest.raises(FileNotFoundError):
            loader.load_model("/nonexistent/model.joblib")


class TestCalculateRegimeFeatures:
    """Test regime feature calculation."""

    def test_returns_dataframe(self, ohlcv_data: pd.DataFrame) -> None:
        features = calculate_regime_features(ohlcv_data)
        assert isinstance(features, pd.DataFrame)

    def test_feature_columns(self, ohlcv_data: pd.DataFrame) -> None:
        features = calculate_regime_features(ohlcv_data)
        expected_cols = [
            "return_20d",
            "volatility",
            "rsi",
            "ma_alignment",
            "volume_ratio_20",
        ]
        for col in expected_cols:
            assert col in features.columns, f"Missing feature column: {col}"

    def test_features_have_values_after_warmup(self, ohlcv_data: pd.DataFrame) -> None:
        """After 20-day warmup, features should have non-NaN values."""
        features = calculate_regime_features(ohlcv_data)
        # After 20 periods, should have valid values
        valid_rows = features.iloc[25:].dropna()
        assert len(valid_rows) > 0

    def test_rsi_range(self, ohlcv_data: pd.DataFrame) -> None:
        """RSI should be between 0 and 100."""
        features = calculate_regime_features(ohlcv_data)
        rsi_valid = features["rsi"].dropna()
        if len(rsi_valid) > 0:
            assert rsi_valid.min() >= 0
            assert rsi_valid.max() <= 100

    def test_small_data(self) -> None:
        """Should handle data smaller than lookback period."""
        np.random.seed(42)
        small_df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [103, 104, 105],
                "low": [97, 98, 99],
                "close": [101, 102, 103],
                "volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="D"),
        )
        features = calculate_regime_features(small_df)
        assert isinstance(features, pd.DataFrame)


class TestPredictRegime:
    """Test regime prediction with mocked model."""

    _FEATURE_NAMES = [
        "return_20d", "volatility", "rsi", "ma_alignment", "volume_ratio_20",
    ]

    def _make_clf_data(self, n_valid: int, regime: str = "BULL_TREND") -> dict:
        """Create mock classifier data for n_valid non-NaN rows."""
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_label_encoder = MagicMock()

        mock_scaler.transform.return_value = np.zeros((n_valid, 5))
        mock_model.predict.return_value = np.zeros(n_valid, dtype=int)
        mock_label_encoder.inverse_transform.return_value = np.array(
            [regime] * n_valid
        )

        return {
            "model": mock_model,
            "scaler": mock_scaler,
            "label_encoder": mock_label_encoder,
            "feature_names": self._FEATURE_NAMES,
        }

    def test_predict_returns_series(self, ohlcv_data: pd.DataFrame) -> None:
        """Predict should return a pandas Series."""
        # Calculate features first to determine valid count
        features = calculate_regime_features(ohlcv_data).dropna()
        n_valid = len(features)
        assert n_valid > 0, "Need at least some valid features"

        clf_data = self._make_clf_data(n_valid, "BULL_TREND")
        result = predict_regime(clf_data, ohlcv_data)

        assert isinstance(result, pd.Series)
        assert len(result) == n_valid

    def test_predict_handles_nan_features(self, ohlcv_data: pd.DataFrame) -> None:
        """Prediction returns only non-NaN rows (NaN rows are dropped)."""
        features = calculate_regime_features(ohlcv_data).dropna()
        n_valid = len(features)

        clf_data = self._make_clf_data(n_valid, "NOT_BULL")
        result = predict_regime(clf_data, ohlcv_data)

        assert isinstance(result, pd.Series)
        # Result should only contain non-NaN feature rows
        assert len(result) == n_valid
        assert (result == "NOT_BULL").all()


class TestPredictRegimeProba:
    """Test probability-based regime prediction."""

    _FEATURE_NAMES = [
        "return_20d", "volatility", "rsi", "ma_alignment", "volume_ratio_20",
    ]

    def test_returns_dataframe(self, ohlcv_data: pd.DataFrame) -> None:
        """predict_regime_proba should return a DataFrame."""
        features = calculate_regime_features(ohlcv_data).dropna()
        n_valid = len(features)

        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_label_encoder = MagicMock()

        mock_scaler.transform.return_value = np.zeros((n_valid, 5))
        mock_model.predict_proba.return_value = np.column_stack([
            np.full(n_valid, 0.7),
            np.full(n_valid, 0.3),
        ])
        mock_label_encoder.classes_ = ["BULL_TREND", "NOT_BULL"]

        clf_data = {
            "model": mock_model,
            "scaler": mock_scaler,
            "label_encoder": mock_label_encoder,
            "feature_names": self._FEATURE_NAMES,
        }

        result = predict_regime_proba(clf_data, ohlcv_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_valid
        assert "BULL_TREND" in result.columns
        assert "NOT_BULL" in result.columns

    def test_probabilities_sum_to_one(self, ohlcv_data: pd.DataFrame) -> None:
        """Each row's probabilities should sum to ~1.0."""
        features = calculate_regime_features(ohlcv_data).dropna()
        n_valid = len(features)

        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_label_encoder = MagicMock()

        mock_scaler.transform.return_value = np.zeros((n_valid, 5))
        mock_model.predict_proba.return_value = np.column_stack([
            np.full(n_valid, 0.6),
            np.full(n_valid, 0.4),
        ])
        mock_label_encoder.classes_ = ["BULL_TREND", "NOT_BULL"]

        clf_data = {
            "model": mock_model,
            "scaler": mock_scaler,
            "label_encoder": mock_label_encoder,
            "feature_names": self._FEATURE_NAMES,
        }

        result = predict_regime_proba(clf_data, ohlcv_data)
        row_sums = result.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums.values, 1.0)

    def test_uses_classes_from_clf_data(self, ohlcv_data: pd.DataFrame) -> None:
        """If clf_data has 'classes' key, use it for column names."""
        features = calculate_regime_features(ohlcv_data).dropna()
        n_valid = len(features)

        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_label_encoder = MagicMock()

        mock_scaler.transform.return_value = np.zeros((n_valid, 5))
        mock_model.predict_proba.return_value = np.column_stack([
            np.full(n_valid, 0.5),
            np.full(n_valid, 0.5),
        ])

        clf_data = {
            "model": mock_model,
            "scaler": mock_scaler,
            "label_encoder": mock_label_encoder,
            "feature_names": self._FEATURE_NAMES,
            "classes": ["CLASS_A", "CLASS_B"],
        }

        result = predict_regime_proba(clf_data, ohlcv_data)
        assert "CLASS_A" in result.columns
        assert "CLASS_B" in result.columns


class TestRegimeFilterCondition:
    """Test RegimeFilterCondition."""

    def test_true_when_regime_bull(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = RegimeFilterCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"regime_bull": True}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is True

    def test_false_when_regime_not_bull(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = RegimeFilterCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"regime_bull": False}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is False

    def test_false_when_missing_indicator(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = RegimeFilterCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is False


class TestRegimeExitCondition:
    """Test RegimeExitCondition."""

    def test_exit_when_regime_bear(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = RegimeExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"regime_bear": True}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is True

    def test_no_exit_when_regime_bull(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = RegimeExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"regime_bear": False}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is False


class TestVBORegimeInit:
    """Test VBORegime initialization."""

    def test_default_parameters(self) -> None:
        strategy = VBORegime(btc_data=pd.DataFrame(), model_path=None)
        assert strategy.name == "VBORegime"
        assert strategy.ma_short == 5
        assert strategy.noise_ratio == 0.5

    def test_custom_parameters(self) -> None:
        strategy = VBORegime(
            name="CustomRegime",
            ma_short=10,
            noise_ratio=0.7,
            btc_data=pd.DataFrame(),
            model_path=None,
        )
        assert strategy.name == "CustomRegime"
        assert strategy.ma_short == 10
        assert strategy.noise_ratio == 0.7

    def test_required_indicators(self) -> None:
        strategy = VBORegime(btc_data=pd.DataFrame(), model_path=None)
        indicators = strategy.required_indicators()
        assert "sma" in indicators
        assert "target" in indicators
        assert "regime_bull" in indicators
        assert "regime_bear" in indicators


class TestVBORegimeIndicators:
    """Test VBORegime indicator calculation."""

    def test_without_model_falls_back(self, ohlcv_data: pd.DataFrame) -> None:
        """Without a model, regime indicators should default to False."""
        strategy = VBORegime(btc_data=None, model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())

        assert "sma" in df.columns
        assert "target" in df.columns
        assert "regime_bull" in df.columns
        assert "regime_bear" in df.columns

    def test_signals_without_model(self, ohlcv_data: pd.DataFrame) -> None:
        """Signals should still work without model (no entries expected)."""
        strategy = VBORegime(btc_data=None, model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())
        df = strategy.generate_signals(df)

        assert "entry_signal" in df.columns
        assert "exit_signal" in df.columns
        assert df["entry_signal"].dtype == bool
        assert df["exit_signal"].dtype == bool


class TestVBORegimeWithMockedModel:
    """Test VBORegime with mocked ML model."""

    @patch("src.strategies.volatility_breakout.vbo_regime.predict_regime")
    @patch("src.strategies.volatility_breakout.vbo_regime.get_regime_model_loader")
    def test_indicators_with_mock_model(
        self,
        mock_get_loader: MagicMock,
        mock_predict: MagicMock,
        ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test indicator calculation with mocked regime prediction."""
        # Mock the model loader
        mock_loader = MagicMock()
        mock_get_loader.return_value = mock_loader
        mock_loader.load_model.return_value = {"model": MagicMock()}

        # Mock predict_regime to return half BULL, half NOT_BULL
        n = len(ohlcv_data)
        regimes = ["BULL_TREND"] * (n // 2) + ["NOT_BULL"] * (n - n // 2)
        mock_predict.return_value = pd.Series(
            regimes, index=ohlcv_data.index, name="regime"
        )

        strategy = VBORegime(
            btc_data=ohlcv_data,  # Provide BTC data so it won't try to load from disk
            model_path="/fake/model.joblib",
        )

        df = strategy.calculate_indicators(ohlcv_data.copy())
        assert "regime_bull" in df.columns
        assert "regime_bear" in df.columns
        # Should have a mix of True/False regime values
        assert df["regime_bull"].any()
        assert df["regime_bear"].any()
